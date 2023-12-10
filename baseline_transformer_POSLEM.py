#Base line pour le POS+lemme (transformer) 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from conllu import parse_incr
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import math

#Lecture+traitement des data
#Modification par rapport à la baseline pour ajouter les lemmes
def load_data(conllu_file):
    sentences, pos_tags, lemmas = [], [], []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            sentences.append([token['form'].lower() for token in sentence])
            pos_tags.append([token['upos'] for token in sentence])
            lemmas.append([token['lemma'] for token in sentence])
    return sentences, pos_tags, lemmas


#handle le dataset
class POSLemDataset(Dataset):
    def __init__(self, sentences, pos_tags, lemmas, word_to_ix, tag_to_ix, lemma_to_ix):
        self.sentences = [torch.tensor([word_to_ix.get(word, 0) for word in sentence], dtype=torch.long) for sentence in sentences]
        self.pos_tags = [torch.tensor([tag_to_ix.get(tag, 0) for tag in tag_list], dtype=torch.long) for tag_list in pos_tags]
        self.lemmas = [torch.tensor([lemma_to_ix.get(lemma, 0) for lemma in lemma_list], dtype=torch.long) for lemma_list in lemmas]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.pos_tags[idx], self.lemmas[idx]


#Padding
def collate_fn(batch):
    sentences, pos_tags, lemmas = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    pos_tags_padded = pad_sequence(pos_tags, batch_first=True, padding_value=-1)
    lemmas_padded = pad_sequence(lemmas, batch_first=True, padding_value=-1)
    return sentences_padded, pos_tags_padded, lemmas_padded


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
class POSLemTransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, nhid, nlayers, pos_tagset_size, lemma_vocab_size):
        super(POSLemTransformerModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        transformer_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, nhid, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layers, nlayers)  # Assurez-vous que cette ligne est correcte
        self.hidden2pos_tag = nn.Linear(embedding_dim, pos_tagset_size)
        self.hidden2lemma = nn.Linear(embedding_dim, lemma_vocab_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence) * math.sqrt(self.embedding_dim)
        embeds = self.pos_encoder(embeds)
        transformer_out = self.transformer_encoder(embeds)
        # Pour POS tagging
        pos_tag_space = self.hidden2pos_tag(transformer_out)
        pos_tag_scores = torch.log_softmax(pos_tag_space, dim=2)

        # Pour lemmatisation
        lemma_space = self.hidden2lemma(transformer_out)
        lemma_scores = torch.log_softmax(lemma_space, dim=2)

        return pos_tag_scores, lemma_scores

#load data
sentences, pos_tags, lemmas = load_data("UD_French-Sequoia/fr_sequoia-ud-dev.conllu")

#init le vocab
word_counts = Counter(word for sentence in sentences for word in sentence)
word_to_ix = {word: i+1 for i, word in enumerate(word_counts)}  # +1 pour le padding
word_to_ix['<PAD>'] = 0

#pos tags
tag_counts = Counter(tag for tags in pos_tags for tag in tags)
tag_to_ix = {tag: i for i, tag in enumerate(tag_counts)}

#lemma_to_ix
lemma_counts = Counter(lemma for lemma_list in lemmas for lemma in lemma_list)
lemma_to_ix = {lemma: i+1 for i, lemma in enumerate(lemma_counts)}  # +1 pour le padding
lemma_to_ix['<PAD>'] = 0

#params
embedding_dim = 256
nhead = 4
nhid = 512
nlayers = 2
batch_size = 2
epochs = 50

#init data et dataloader
dataset = POSLemDataset(sentences, pos_tags, lemmas, word_to_ix, tag_to_ix, lemma_to_ix)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

#init transformer, loss et optimizer
#changer les parametres en CV pour optimiser tout ça
model = POSLemTransformerModel(len(word_to_ix), embedding_dim, nhead, nhid, nlayers, len(tag_to_ix),len(lemma_to_ix))
loss_function = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

#Training
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for sentence_in, pos_targets, lemma_targets in data_loader:
        optimizer.zero_grad()
        pos_tag_scores, lemma_scores = model(sentence_in)
        pos_loss = loss_function(pos_tag_scores.view(-1, len(tag_to_ix)), pos_targets.view(-1))
        lemma_loss = loss_function(lemma_scores.view(-1, len(lemma_to_ix)), lemma_targets.view(-1))
        loss = pos_loss + lemma_loss  # Combinez les pertes
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Total Loss: {total_loss / len(data_loader)}")


# Exemple de prédiction + affichage des prédictions et des vraies valeurs
def calculate_accuracy(true, predicted):
    correct = sum(t == p for t, p in zip(true, predicted))
    return correct / len(true) if len(true) > 0 else 0

# Évaluation du modèle
model.eval()
all_true_pos, all_predicted_pos = [], []
all_true_lemmas, all_predicted_lemmas = [], []
with torch.no_grad():
    for sentences, pos_tags, lemmas in data_loader:
        pos_tag_scores, lemma_scores = model(sentences)
        pos_predicted = pos_tag_scores.argmax(dim=2)
        lemma_predicted = lemma_scores.argmax(dim=2)

        for i in range(pos_tags.shape[0]):  # Parcourir chaque phrase du batch
            valid_indices = pos_tags[i] != -1  # Indices où les tags ne sont pas padding
            all_true_pos.extend(pos_tags[i][valid_indices].tolist())
            all_predicted_pos.extend(pos_predicted[i][valid_indices].tolist())
            all_true_lemmas.extend(lemmas[i][valid_indices].tolist())
            all_predicted_lemmas.extend(lemma_predicted[i][valid_indices].tolist())

# Calcul de l'accuracy pour le POS tagging et la lemmatisation
accuracy_pos = calculate_accuracy(all_true_pos, all_predicted_pos)
accuracy_lemmas = calculate_accuracy(all_true_lemmas, all_predicted_lemmas)

# Calcul du score total (moyenne des deux accuracies)
total_score = (accuracy_pos + accuracy_lemmas) / 2

print(f"POS Tagging Accuracy: {accuracy_pos:.4f}")
print(f"Lemmatization Accuracy: {accuracy_lemmas:.4f}")
print(f"Total Score: {total_score:.4f}")


