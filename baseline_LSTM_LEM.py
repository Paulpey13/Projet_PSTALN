import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from conllu import parse_incr
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

def load_data(conllu_file):
    sentences, lemmas = [], []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            sentences.append([token['form'].lower() for token in sentence])
            lemmas.append([token['lemma'] for token in sentence])
    return sentences, lemmas

def word_to_index(word, word_to_ix):
    return word_to_ix[word] if word in word_to_ix else word_to_ix['<OOV>']

class LEMDataset(Dataset):
    def __init__(self, sentences, lemmas, word_to_ix, lemma_to_ix):
        self.sentences = [torch.tensor([word_to_ix.get(word, 0) for word in sentence], dtype=torch.long) for sentence in sentences]
        self.lemmas = [torch.tensor([lemma_to_ix.get(lemma, 0) for lemma in lemma_list], dtype=torch.long) for lemma_list in lemmas]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.lemmas[idx]
    
def collate_fn(batch):
    sentences, lemmas = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    lemmas_padded = pad_sequence(lemmas, batch_first=True, padding_value=0)
    return sentences_padded, lemmas_padded

class LEM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, lemmas_size):
        super(LEM_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2lem = nn.Linear(hidden_dim, lemmas_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        lem_space = self.hidden2lem(lstm_out)
        lem_scores = torch.log_softmax(lem_space, dim=2)
        return lem_scores

def calculate_accuracy(true_lemma, predicted_lemmas):
    correct = sum(t1 == t2 for t1, t2 in zip(true_lemma, predicted_lemmas))
    return correct / len(true_lemma)

def calculate_f1(true_lemma, predicted_lemmas):
    true_positives = sum(t1 == t2 and t1 != 0 for t1, t2 in zip(true_lemma, predicted_lemmas))
    false_positives = sum(t1 != 0 and t2 == 0 for t1, t2 in zip(true_lemma, predicted_lemmas))
    false_negatives = sum(t1 == 0 and t2 != 0 for t1, t2 in zip(true_lemma, predicted_lemmas))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def evaluate_model(model, data_loader, loss_function, lem_to_ix):
    model.eval()
    total_loss = 0
    all_true_lemmas = []
    all_predicted_lemmas = []
    with torch.no_grad():
        for sentences, lemmas in data_loader:
            lemma_scores = model(sentences)
            predicted = torch.argmax(lemma_scores, dim=2)
            loss = loss_function(lemma_scores.view(-1, len(lem_to_ix)), lemmas.view(-1))
            total_loss += loss.item()
            all_true_lemmas.extend(lemmas.flatten().tolist())
            all_predicted_lemmas.extend(predicted.flatten().tolist())
    filtered_true_lemmas = [lem for lem in all_true_lemmas if lem != -1]
    filtered_predicted_lemmas = [all_predicted_lemmas[i] for i, lem in enumerate(all_true_lemmas) if lem != -1]
    accuracy = calculate_accuracy(filtered_true_lemmas, filtered_predicted_lemmas)
    precision, recall, f1 = calculate_f1(filtered_true_lemmas, filtered_predicted_lemmas)
    return total_loss / len(data_loader), accuracy, precision, recall, f1