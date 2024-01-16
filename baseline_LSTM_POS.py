import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from conllu import parse_incr
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

def load_data(conllu_file):
    sentences, pos_tags = [], []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            sentences.append([token['form'].lower() for token in sentence])
            pos_tags.append([token['upos'] for token in sentence])
    return sentences, pos_tags

def word_to_index(word, word_to_ix):
    return word_to_ix[word] if word in word_to_ix else word_to_ix['<OOV>']

class POSDataset(Dataset):
    def __init__(self, sentences, tags, word_to_ix, tag_to_ix):
        self.sentences = [torch.tensor([word_to_index(word, word_to_ix) for word in sentence], dtype=torch.long) for sentence in sentences]
        self.tags = [torch.tensor([tag_to_ix[tag] for tag in tag_list], dtype=torch.long) for tag_list in tags]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]

def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=-1)
    return sentences_padded, tags_padded

class POSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
        super(POSModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = torch.log_softmax(tag_space, dim=2)
        return tag_scores

def calculate_accuracy(true_tags, pred_tags):
    correct = sum(t1 == t2 for t1, t2 in zip(true_tags, pred_tags))
    return correct / len(true_tags)

def evaluate_model(model, data_loader, loss_function, tag_to_ix):
    model.eval()
    total_loss = 0
    all_true_pos = []
    all_predicted_pos = []
    with torch.no_grad():
        for sentences, pos_tags in data_loader:
            pos_scores = model(sentences)
            predicted = torch.argmax(pos_scores, dim=2)
            loss = loss_function(pos_scores.view(-1, len(tag_to_ix)), pos_tags.view(-1))
            total_loss += loss.item()
            all_true_pos.extend(pos_tags.flatten().tolist())
            all_predicted_pos.extend(predicted.flatten().tolist())
    filtered_true_pos = [tag for tag in all_true_pos if tag != -1]
    filtered_predicted_pos = [all_predicted_pos[i] for i, tag in enumerate(all_true_pos) if tag != -1]
    accuracy = calculate_accuracy(filtered_true_pos, filtered_predicted_pos)
    f1 = f1_score(filtered_true_pos, filtered_predicted_pos, average='macro')
    return total_loss / len(data_loader), accuracy, f1
