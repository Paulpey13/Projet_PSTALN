#Base line pour le POS (transformer) 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from conllu import parse_incr
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import math
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device='cpu'
print(torch.cuda.is_available())  # Renvoie True si un GPU est disponible
print(device)  # Renvoie le périphérique actuel (CPU ou GPU
#Lecture+traitement des data
def load_data(conllu_file):
    sentences, pos_tags = [], []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            sentences.append([token['form'].lower() for token in sentence])
            pos_tags.append([token['upos'] for token in sentence])
    return sentences, pos_tags

#Init le POS
class POSDataset(Dataset):
    def __init__(self, sentences, tags, word_to_ix, tag_to_ix):
        # Add the <UNK> token to the vocabulary
        self.sentences = [torch.tensor([word_to_ix.get(word, word_to_ix['<UNK>']) for word in sentence], dtype=torch.long) for sentence in sentences]
        self.tags = [torch.tensor([tag_to_ix[tag] for tag in tag_list], dtype=torch.long) for tag_list in tags]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]

#Padding
def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=-1)
    return sentences_padded, tags_padded

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

#Modele tranformer pour le POS
class POSTransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, nhid, nlayers, tagset_size):
        super(POSTransformerModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        transformer_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, nhid, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layers, nlayers)
        self.hidden2tag = nn.Linear(embedding_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence) * math.sqrt(self.embedding_dim)
        embeds = self.pos_encoder(embeds)
        transformer_out = self.transformer_encoder(embeds)
        tag_space = self.hidden2tag(transformer_out)
        tag_scores = torch.log_softmax(tag_space, dim=2)
        return tag_scores


#Evaluations du modele : ici accuracy mais voir si y'en a pas un mieux
def calculate_accuracy(true_tags, pred_tags):
    correct = sum(t1 == t2 for t1, t2 in zip(true_tags, pred_tags))
    return correct / len(true_tags)

def evaluate_model(model, data_loader, loss_function, tag_to_ix):
    model.eval()
    total_loss = 0
    all_true_tags = []
    all_predicted_tags = []
    with torch.no_grad():
        for sentence_in, targets in data_loader:
            sentence_in, targets = sentence_in.to(device), targets.to(device)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores.view(-1, len(tag_to_ix)), targets.view(-1))
            total_loss += loss.item()

            # Récupération des prédictions
            predicted = torch.argmax(tag_scores, dim=2)
            all_true_tags.extend(targets.flatten().tolist())
            all_predicted_tags.extend(predicted.flatten().tolist())

    # Filtrer les étiquettes de padding
    filtered_true_tags = [tag for tag in all_true_tags if tag != -1]
    filtered_predicted_tags = [all_predicted_tags[i] for i, tag in enumerate(all_true_tags) if tag != -1]

    # Calculer l'accuracy
    accuracy = calculate_accuracy(filtered_true_tags, filtered_predicted_tags)
    f1 = f1_score(filtered_true_tags, filtered_predicted_tags, average='macro')
    return total_loss / len(data_loader), accuracy, f1
