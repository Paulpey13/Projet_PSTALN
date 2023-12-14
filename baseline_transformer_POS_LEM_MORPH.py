#Base line pour le POS+lemme+morphologies (transformer) 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from conllu import parse_incr
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import math
from utils import load_data



#handle le dataset
class POSLemDataset(Dataset):
    def __init__(self, sentences, pos_tags, lemmas, morph_traits, word_to_ix, tag_to_ix, lemma_to_ix, morph_to_ix):
        self.sentences = [torch.tensor([word_to_ix.get(word, 0) for word in sentence], dtype=torch.long) for sentence in sentences]
        self.pos_tags = [torch.tensor([tag_to_ix.get(tag, 0) for tag in tag_list], dtype=torch.long) for tag_list in pos_tags]
        self.lemmas = [torch.tensor([lemma_to_ix.get(lemma, 0) for lemma in lemma_list], dtype=torch.long) for lemma_list in lemmas]
        self.morph_traits = [torch.tensor([morph_to_ix.get(trait, 0) for trait in trait_list], dtype=torch.long) for trait_list in morph_traits]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.pos_tags[idx], self.lemmas[idx], self.morph_traits[idx]



#Padding
def collate_fn(batch):
    sentences, pos_tags, lemmas, morph_traits = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    pos_tags_padded = pad_sequence(pos_tags, batch_first=True, padding_value=-1)
    lemmas_padded = pad_sequence(lemmas, batch_first=True, padding_value=-1)
    morph_traits_padded = pad_sequence(morph_traits, batch_first=True, padding_value=0)
    return sentences_padded, pos_tags_padded, lemmas_padded, morph_traits_padded



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)

        # before merging morph :
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

#Modele tranformer pour le POS
class POSLemTransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, nhid, nlayers, pos_tagset_size, lemma_vocab_size, morph_trait_size):
        super(POSLemTransformerModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        transformer_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, nhid, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layers, nlayers)
        self.hidden2pos_tag = nn.Linear(embedding_dim, pos_tagset_size)
        self.hidden2lemma = nn.Linear(embedding_dim, lemma_vocab_size)
        self.hidden2morph = nn.Linear(embedding_dim, morph_trait_size)  

    def forward(self, sentence):
        embeds = self.embedding(sentence) * math.sqrt(self.embedding_dim)
        embeds = self.pos_encoder(embeds)
        transformer_out = self.transformer_encoder(embeds)

        pos_tag_space = self.hidden2pos_tag(transformer_out)
        pos_tag_scores = torch.log_softmax(pos_tag_space, dim=2)

        lemma_space = self.hidden2lemma(transformer_out)
        lemma_scores = torch.log_softmax(lemma_space, dim=2)

        morph_trait_space = self.hidden2morph(transformer_out) 
        morph_trait_scores = torch.log_softmax(morph_trait_space, dim=2)

        return pos_tag_scores, lemma_scores, morph_trait_scores


#load data
sentences, pos_tags, lemmas, morph_traits = load_data("UD_French-Sequoia/fr_sequoia-ud-dev.conllu",True,True,True)

#init le vocab
word_counts = Counter(word for sentence in sentences for word in sentence)
word_to_ix = {word: i+1 for i, word in enumerate(word_counts)}  #+1 pour le padding
word_to_ix['<PAD>'] = 0

#pos tags
tag_counts = Counter(tag for tags in pos_tags for tag in tags)
tag_to_ix = {tag: i for i, tag in enumerate(tag_counts)}

#lemmes
lemma_counts = Counter(lemma for lemma_list in lemmas for lemma in lemma_list)
lemma_to_ix = {lemma: i+1 for i, lemma in enumerate(lemma_counts)}  #+1 pour le padding
lemma_to_ix['<PAD>'] = 0

#morphologies
unique_morph_traits = set(trait for trait_list in morph_traits for trait in trait_list if trait != '_')
morph_to_ix = {trait: i+1 for i, trait in enumerate(unique_morph_traits)}
morph_to_ix['_'] = 0


#params
embedding_dim = 256
nhead = 4
nhid = 512
nlayers = 2
batch_size = 2
epochs = 1

#init data et dataloader
# Initialisation de POSLemDataset avec tous les arguments nécessaires
dataset = POSLemDataset(sentences, pos_tags, lemmas, morph_traits, word_to_ix, tag_to_ix, lemma_to_ix, morph_to_ix)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

#init transformer, loss et optimizer
#changer les parametres en CV pour optimiser tout ça
model = POSLemTransformerModel(len(word_to_ix), embedding_dim, nhead, nhid, nlayers, len(tag_to_ix),len(lemma_to_ix),len(morph_to_ix))
loss_function = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

#Training
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for sentence_in, pos_targets, lemma_targets, morph_targets in data_loader:
        optimizer.zero_grad()
        pos_tag_scores, lemma_scores, morph_trait_scores = model(sentence_in)
        pos_loss = loss_function(pos_tag_scores.view(-1, len(tag_to_ix)), pos_targets.view(-1))
        lemma_loss = loss_function(lemma_scores.view(-1, len(lemma_to_ix)), lemma_targets.view(-1))
        morph_loss = loss_function(morph_trait_scores.view(-1, len(morph_to_ix)), morph_targets.view(-1))
        loss = pos_loss + lemma_loss + morph_loss  #combine les pertes (à vori si cest vraiment comme ça qu'il faut faire)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Total Loss: {total_loss / len(data_loader)}")



def calculate_accuracy(true, predicted):
    correct = sum(t == p for t, p in zip(true, predicted))
    return correct / len(true) if len(true) > 0 else 0

#evaluation
model.eval()
all_true_pos, all_predicted_pos = [], []
all_true_lemmas, all_predicted_lemmas = [], []
all_true_morph, all_predicted_morph = [], []

with torch.no_grad():
    for sentences, pos_tags, lemmas, morphs in data_loader:
        pos_tag_scores, lemma_scores, morph_trait_scores = model(sentences)
        pos_predicted = pos_tag_scores.argmax(dim=2)
        lemma_predicted = lemma_scores.argmax(dim=2)
        morph_predicted = morph_trait_scores.argmax(dim=2)

        for i in range(pos_tags.shape[0]):  #Parcourir chaque phrase du batch
            valid_indices = pos_tags[i] != -1  #Indices où les tags ne sont pas padding
            all_true_pos.extend(pos_tags[i][valid_indices].tolist())
            all_predicted_pos.extend(pos_predicted[i][valid_indices].tolist())
            all_true_lemmas.extend(lemmas[i][valid_indices].tolist())
            all_predicted_lemmas.extend(lemma_predicted[i][valid_indices].tolist())
            all_true_morph.extend(morphs[i][valid_indices].tolist())
            all_predicted_morph.extend(morph_predicted[i][valid_indices].tolist())

#Accuracy poru chaque tâche
accuracy_pos = calculate_accuracy(all_true_pos, all_predicted_pos)
accuracy_lemmas = calculate_accuracy(all_true_lemmas, all_predicted_lemmas)
accuracy_morph = calculate_accuracy(all_true_morph, all_predicted_morph)

#print scores
print(f"POS Tagging Accuracy: {accuracy_pos:.4f}")
print(f"Lemmatization Accuracy: {accuracy_lemmas:.4f}")
print(f"Morphological Traits Accuracy: {accuracy_morph:.4f}")