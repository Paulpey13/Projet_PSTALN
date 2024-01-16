import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from conllu import parse_incr
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

def load_data(conllu_file):
    sentences, morphs = [], []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            sentences.append([token['form'].lower() for token in sentence])
            morphs.append(['|'.join([f"{k}={v}" for k, v in token['feats'].items()]) if token['feats'] else '_' for token in sentence])
    return sentences, morphs

def word_to_index(word, word_to_ix):
    return word_to_ix[word] if word in word_to_ix else 0

class MORPHDataset(Dataset):
    def __init__(self, sentences, morphs, word_to_ix, morph_to_ix):
        self.sentences = [torch.tensor([word_to_ix.get(word, 0) for word in sentence], dtype=torch.long) for sentence in sentences]
        self.morphs = [torch.tensor([morph_to_ix.get(trait, 0) for trait in trait_list], dtype=torch.long) for trait_list in morphs]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.morphs[idx]
    
def collate_fn(batch):
    sentences, morphs = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    morphs_padded = pad_sequence(morphs, batch_first=True, padding_value=0)
    return sentences_padded, morphs_padded

class MORPH_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, morphs_size):
        super(MORPH_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2morph = nn.Linear(hidden_dim, morphs_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        morph_space = self.hidden2morph(lstm_out)
        morph_scores = torch.log_softmax(morph_space, dim=2)
        return morph_scores

def calculate_accuracy(true_morph, predicted_morphs):
    correct = sum(t1 == t2 for t1, t2 in zip(true_morph, predicted_morphs))
    return correct / len(true_morph)

def evaluate_model(model, data_loader, loss_function, morph_to_ix):
    model.eval()
    total_loss = 0
    all_true_morphs = []
    all_predicted_morphs = []
    with torch.no_grad():
        for sentences, morphs in data_loader:
            morph_scores = model(sentences)
            predicted = torch.argmax(morph_scores, dim=2)
            all_true_morphs.extend(morphs.flatten().tolist())
            all_predicted_morphs.extend(predicted.flatten().tolist())
            loss = loss_function(morph_scores.view(-1, len(morph_to_ix)), morphs.view(-1))
            total_loss += loss.item()
    filtered_true_morphs = [morph for morph in all_true_morphs if morph != -1]
    filtered_predicted_morphs = [all_predicted_morphs[i] for i, morph in enumerate(all_true_morphs) if morph != -1]
    accuracy = calculate_accuracy(filtered_true_morphs, filtered_predicted_morphs)
    f1 = f1_score(filtered_true_morphs, filtered_predicted_morphs, average='macro')
    return total_loss / len(data_loader), accuracy, f1