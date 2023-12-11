#Test morph tout seul pour avoir une meilleur vision de ce qui se passe
# ça permettra aussi de comparer apres la dépendance


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

class MorphDataset(Dataset):
    def __init__(self, sentences, morph_traits, word_to_ix, morph_to_ix):
        self.sentences = [torch.tensor([word_to_ix.get(word, 0) for word in sentence], dtype=torch.long) for sentence in sentences]
        self.morph_traits = [torch.tensor([morph_to_ix.get(trait, 0) for trait in trait_list], dtype=torch.long) for trait_list in morph_traits]

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
    def __init__(self, vocab_size, embedding_dim, nhead, nhid, nlayers, morph_trait_size):
        super(MorphTransformerModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        transformer_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, nhid, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layers, nlayers)
        self.hidden2morph = nn.Linear(embedding_dim, morph_trait_size)  

    def forward(self, sentence):
        embeds = self.embedding(sentence) * math.sqrt(self.embedding_dim)
        embeds = self.pos_encoder(embeds)
        transformer_out = self.transformer_encoder(embeds)
        morph_trait_space = self.hidden2morph(transformer_out) 
        morph_trait_scores = torch.log_softmax(morph_trait_space, dim=2)
        return morph_trait_scores

#import data
sentences, morph_traits = load_data("UD_French-Sequoia/fr_sequoia-ud-dev.conllu")

#dico mots+morphs
word_counts = Counter(word for sentence in sentences for word in sentence)
word_to_ix = {word: i+1 for i, word in enumerate(word_counts)}
word_to_ix['<PAD>'] = 0

#vérifier que c'est bien séparé
unique_morph_traits = set(trait for trait_list in morph_traits for trait in trait_list if trait != '_')
morph_to_ix = {trait: i+1 for i, trait in enumerate(unique_morph_traits)}
morph_to_ix['_'] = 0

#params
embedding_dim = 256
nhead = 4
nhid = 512
nlayers = 2
batch_size = 2
epochs = 10

#dataset et dataloader
dataset = MorphDataset(sentences, morph_traits, word_to_ix, morph_to_ix)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

#init du model
model = MorphTransformerModel(len(word_to_ix), embedding_dim, nhead, nhid, nlayers, len(morph_to_ix))
loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=0.01)

#training
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for sentence_in, morph_targets in data_loader:
        optimizer.zero_grad()
        morph_trait_scores = model(sentence_in)
        loss = loss_function(morph_trait_scores.view(-1, len(morph_to_ix)), morph_targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Total Loss: {total_loss / len(data_loader)}")

#accuracy, evaluation
def calculate_accuracy(true, predicted):
    correct = sum(t == p for t, p in zip(true, predicted))
    return correct / len(true) if len(true) > 0 else 0

model.eval()
all_true_morph, all_predicted_morph = [], []

with torch.no_grad():
    for sentences, morphs in data_loader:
        morph_trait_scores = model(sentences)
        morph_predicted = morph_trait_scores.argmax(dim=2)

        for i in range(morphs.shape[0]):
            valid_indices = morphs[i] != 0
            all_true_morph.extend(morphs[i][valid_indices].tolist())
            all_predicted_morph.extend(morph_predicted[i][valid_indices].tolist())

accuracy_morph = calculate_accuracy(all_true_morph, all_predicted_morph)
print(f"Morphological Traits Accuracy: {accuracy_morph:.4f}")

#obtenir moprhologies à partir des indices
def get_morphologies_from_indices(indices, ix_to_morph):
    return [ix_to_morph.get(index, '_') for index in indices]

#inversion du dico
ix_to_morph = {i: morph for morph, i in morph_to_ix.items()}

#j'ai refait un test la pour afficher predictions et voir si cest bien
model.eval()
with torch.no_grad():
    for sentences, true_morphs in data_loader:
        predicted_morph_scores = model(sentences)
        predicted_morphs = predicted_morph_scores.argmax(dim=2)

        for i in range(sentences.shape[0]):  #phrase par phrase
            print(f"Sentence {i+1}:")
            for j in range(sentences[i].shape[0]):  #mot par mot
                if sentences[i][j] == word_to_ix['<PAD>']:  #ignore padding
                    continue
                word = list(word_to_ix.keys())[list(word_to_ix.values()).index(sentences[i][j])]
                true_morph_idx = true_morphs[i][j].item()
                predicted_morph_idx = predicted_morphs[i][j].item()
                
                true_morph = get_morphologies_from_indices([true_morph_idx], ix_to_morph)[0]
                predicted_morph = get_morphologies_from_indices([predicted_morph_idx], ix_to_morph)[0]

                print(f"Word: {word} - True Morphology: {true_morph} - Predicted Morphology: {predicted_morph}")
