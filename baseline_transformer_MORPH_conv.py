import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from conllu import parse_incr
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import math

#import les données
def load_data(conllu_file):
    sentences, morph_traits = [], []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            sentences.append([token['form'].lower() for token in sentence])
            morph_traits.append(['|'.join([f"{k}={v}" for k, v in token['feats'].items()]) if token['feats'] else '_' for token in sentence])
    return sentences, morph_traits

# Define load_data function
def load_data_preprocess(conllu_file, char_to_ix, max_word_len):
    sentences, morph_traits = [], []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            char_sentences = []
            for token in sentence:
                chars = [char_to_ix.get(c, char_to_ix['<UNK>']) for c in token['form'].lower()]
                chars = chars[:max_word_len] + [char_to_ix['<PAD>']] * (max_word_len - len(chars))
                char_sentences.append(chars)
            sentences.append(char_sentences)
            morph_traits.append(['|'.join([f"{k}={v}" for k, v in token['feats'].items()]) if token['feats'] else '_' for token in sentence])
    return sentences, morph_traits

class MORPHDataset(Dataset):
    def __init__(self, sentences, morph_traits, morph_to_ix, max_word_len, char_to_ix):
        # Convert sentences to PyTorch tensors
        self.sentences = [torch.tensor([self.pad_word(word, max_word_len, char_to_ix['<PAD>']) for word in sentence], dtype=torch.long) for sentence in sentences]
        
        self.morph_traits = [torch.tensor([morph_to_ix.get(trait, 0) for trait in trait_list], dtype=torch.long) for trait_list in morph_traits]
       

    def pad_word(self, word, max_word_len, pad_index):
        # Function to pad or truncate each word to the same length
        return word[:max_word_len] + [pad_index] * (max_word_len - len(word))
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.morph_traits[idx]

#Padding
def collate_fn(batch):
    sentences, morph_traits = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    morph_traits_padded = pad_sequence(morph_traits, batch_first=True, padding_value=0)
    return sentences_padded, morph_traits_padded

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

#Positional Encoding pour les morphologies 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#Transformer pour morphs seulement
class MorphTransformerModel(nn.Module):
    def __init__(self, num_chars, char_embedding_dim, num_filters, kernel_size, nhead, nhid, nlayers, morph_trait_size):
        super(MorphTransformerModel, self).__init__()
        self.char_cnn_embedding = CharCNNEmbedding(num_chars, char_embedding_dim, num_filters, kernel_size)
        self.pos_encoder = PositionalEncoding(num_filters)
        transformer_layers = nn.TransformerEncoderLayer(num_filters, nhead, nhid, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layers, nlayers)
        self.hidden2morph = nn.Linear(num_filters, morph_trait_size)  

    def forward(self, sentence):
        embeds = self.char_cnn_embedding(sentence)
        embeds = self.pos_encoder(embeds)
        transformer_out = self.transformer_encoder(embeds)
        morph_trait_space = self.hidden2morph(transformer_out) 
        morph_trait_scores = torch.log_softmax(morph_trait_space, dim=2)
        return morph_trait_scores
    
#accuracy, evaluation
def calculate_accuracy(true, predicted):
    correct = sum(t == p for t, p in zip(true, predicted))
    return correct / len(true) if len(true) > 0 else 0

def calculate_f1(true_morph, predicted_morphs):
    true_positives = sum(t1 == t2 and t1 != 0 for t1, t2 in zip(true_morph, predicted_morphs))
    false_positives = sum(t1 != 0 and t2 == 0 for t1, t2 in zip(true_morph, predicted_morphs))
    false_negatives = sum(t1 == 0 and t2 != 0 for t1, t2 in zip(true_morph, predicted_morphs))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def evaluate_model(model, data_loader, loss_function,device, morph_to_ix):
    model.eval()
    total_loss = 0
    all_true_morph = []
    all_predicted_morph = []

    with torch.no_grad():
        for sentence_in, targets in data_loader:
            sentence_in, targets = sentence_in.to(device), targets.to(device)
            morph_scores = model(sentence_in)
            loss = loss_function(morph_scores.view(-1, len(morph_to_ix)), targets.view(-1))
            total_loss += loss.item()

            # Récupération des prédictions
            predicted = torch.argmax(morph_scores, dim=2)
            all_true_morph.extend(targets.flatten().tolist())
            all_predicted_morph.extend(predicted.flatten().tolist())

    # Filtrer les étiquettes de padding
    filtered_true_morph = [morph for morph in all_true_morph if morph != -1]
    filtered_predicted_morph = [all_predicted_morph[i] for i, morph in enumerate(all_true_morph) if morph != -1]

    # Calculer l'accuracy
    accuracy = calculate_accuracy(filtered_true_morph, filtered_predicted_morph)
    precision, recall, f1 = calculate_f1(filtered_true_morph, filtered_predicted_morph)
    return total_loss / len(data_loader), accuracy, precision, recall, f1