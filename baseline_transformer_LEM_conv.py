import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from conllu import parse_incr
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import math
from sklearn.metrics import f1_score

#import les données
def load_data(conllu_file):
    sentences, lemmas = [], []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            sentences.append([token['form'].lower() for token in sentence])
            lemmas.append([token['lemma'] for token in sentence])
    return sentences, lemmas

# Define load_data function
def load_data_preprocess(conllu_file, char_to_ix, max_word_len):
    sentences, lemmas = [], []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            char_sentences = []
            for token in sentence:
                chars = [char_to_ix.get(c, char_to_ix['<UNK>']) for c in token['form'].lower()]
                chars = chars[:max_word_len] + [char_to_ix['<PAD>']] * (max_word_len - len(chars))
                char_sentences.append(chars)
            sentences.append(char_sentences)
            lemmas.append([token['lemma'] for token in sentence])
    return sentences, lemmas

class CharCNNEmbedding(nn.Module):
    def __init__(self, num_chars, embedding_dim, num_filters, kernel_size):
        super(CharCNNEmbedding, self).__init__()
        self.char_embedding = nn.Embedding(num_chars, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, num_filters, kernel_size, padding=kernel_size // 2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.embedding_dim = num_filters

    def forward(self, x):
        # x: [batch_size, seq_len, word_len]
        batch_size, seq_len, word_len = x.size(0), x.size(1), x.size(2)
        x = self.char_embedding(x)  # [batch_size, seq_len, word_len, embedding_dim]
        x = x.view(batch_size * seq_len, word_len, -1).transpose(1, 2)  # [batch_size * seq_len, embedding_dim, word_len]
        x = self.conv(x)  # Apply Conv1d
        x = self.pool(x).squeeze(-1)  # [batch_size * seq_len, num_filters]
        x = x.view(batch_size, seq_len, -1)  # Reshape back to [batch_size, seq_len, num_filters]
        return x

class LEMDataset(Dataset):
    def __init__(self, sentences, lemmas, lemma_to_ix, max_word_len, char_to_ix):
        # Convert sentences to PyTorch tensors
        self.sentences = [torch.tensor([self.pad_word(word, max_word_len, char_to_ix['<PAD>']) for word in sentence], dtype=torch.long) for sentence in sentences]
        
        self.lemmas = [torch.tensor([lemma_to_ix.get(lemma, 0) for lemma in lemma_list], dtype=torch.long) for lemma_list in lemmas]
       

    def pad_word(self, word, max_word_len, pad_index):
        # Function to pad or truncate each word to the same length
        return word[:max_word_len] + [pad_index] * (max_word_len - len(word))
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.lemmas[idx]
    
#Padding
def collate_fn(batch):
    sentences, lemmas = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    lemmas_padded = pad_sequence(lemmas, batch_first=True, padding_value=0)
    return sentences_padded, lemmas_padded

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LEMTransformerModel(nn.Module):
    def __init__(self, num_chars, char_embedding_dim, num_filters, kernel_size, nhead, nhid, nlayers, lemma_size):
        super(LEMTransformerModel, self).__init__()
        self.char_cnn_embedding = CharCNNEmbedding(num_chars, char_embedding_dim, num_filters, kernel_size)
        self.pos_encoder = PositionalEncoding(num_filters)
        transformer_layers = nn.TransformerEncoderLayer(num_filters, nhead, nhid, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layers, nlayers)
        self.hidden2lem = nn.Linear(num_filters, lemma_size)

    def forward(self, sentence):
        embeds = self.char_cnn_embedding(sentence)  # Replace word embedding
        embeds = self.pos_encoder(embeds)
        transformer_out = self.transformer_encoder(embeds)
        lem_space = self.hidden2lem(transformer_out)
        lem_scores = torch.log_softmax(lem_space, dim=2)
        return lem_scores

def calculate_accuracy(true, predicted):
    correct = sum(t == p for t, p in zip(true, predicted))
    return correct / len(true) if len(true) > 0 else 0

def evaluate_model(model, data_loader, loss_function,device, lemma_to_ix):
    model.eval()
    total_loss = 0
    all_true_lem = []
    all_predicted_lem = []

    with torch.no_grad():
        for sentence_in, targets in data_loader:
            sentence_in, targets = sentence_in.to(device), targets.to(device)
            lem_scores = model(sentence_in)
            loss = loss_function(lem_scores.view(-1, len(lemma_to_ix)), targets.view(-1))
            total_loss += loss.item()

            # Récupération des prédictions
            predicted = torch.argmax(lem_scores, dim=2)
            all_true_lem.extend(targets.flatten().tolist())
            all_predicted_lem.extend(predicted.flatten().tolist())

    # Filtrer les étiquettes de padding
    filtered_true_lem = [lem for lem in all_true_lem if lem != -1]
    filtered_predicted_lem = [all_predicted_lem[i] for i, lem in enumerate(all_true_lem) if lem != -1]

    # Calculer l'accuracy
    accuracy = calculate_accuracy(filtered_true_lem, filtered_predicted_lem)
    f1 = f1_score(filtered_true_lem, filtered_predicted_lem, average='macro')
    return total_loss / len(data_loader), accuracy, f1