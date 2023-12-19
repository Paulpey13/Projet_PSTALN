#Base line pour le POS (transformer) 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from conllu import parse_incr
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())  # Renvoie True si un GPU est disponible

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
def evaluate_model(model, data_loader, loss_function):
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
    return total_loss / len(data_loader), accuracy


#load data
sentences, pos_tags = load_data("UD_French-Sequoia/fr_sequoia-ud-train.conllu")

#init le vocab
word_counts = Counter(word for sentence in sentences for word in sentence)
word_to_ix = {word: i+1 for i, word in enumerate(word_counts)}  # +1 pour le padding
word_to_ix['<UNK>'] = 1

tag_counts = Counter(tag for tags in pos_tags for tag in tags)
tag_to_ix = {tag: i for i, tag in enumerate(tag_counts)}

#params
embedding_dim = 256
nhead = 4
nhid = 512
nlayers = 2
batch_size = 1
epochs = 20



#init data et dataloader
dataset = POSDataset(sentences, pos_tags, word_to_ix, tag_to_ix)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

#init transformer, loss et optimizer
#changer les parametres en CV pour optimiser tout ça
model = POSTransformerModel(len(word_to_ix), embedding_dim, nhead, nhid, nlayers, len(tag_to_ix))
loss_function = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.SGD(model.parameters(), lr=0.01)


validation_sentences, validation_pos_tags = load_data("UD_French-Sequoia/fr_sequoia-ud-dev.conllu")
# Préparer le DataLoader pour les données de validation
validation_dataset = POSDataset(validation_sentences, validation_pos_tags, word_to_ix, tag_to_ix)
validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Paramètres pour l'arrêt précoce
patience = 2  # Nombre d'époques à attendre après la dernière amélioration de la loss de validation
best_val_accuracy = 0
epochs_no_improve = 0
# Assuming tag_counts is a Counter object containing counts of each tag
print(tag_counts)
most_common_tag = tag_counts.most_common(1)[0][0]
print(most_common_tag)

#Training
for epoch in range(epochs): 
    model.train()
    model.to(device)  # Déplacer le modèle sur le GPU si disponible
    total_loss = 0
    for sentence_in, targets in data_loader:
        sentence_in, targets = sentence_in.to(device), targets.to(device)  # Déplacer les données sur le périphérique
        optimizer.zero_grad()
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores.view(-1, len(tag_to_ix)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Utiliser la fonction modifiée pour évaluer la validation loss et l'accuracy
    val_loss, val_accuracy = evaluate_model(model, validation_data_loader, loss_function)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # Arrêt précoce si aucune amélioration
    if epochs_no_improve == patience:
        print("Arrêt précoce : La loss de validation ne s'améliore plus")
        break
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")

# Modifier la section de prédiction pour déplacer les entrées sur le périphérique
model.eval()
with torch.no_grad():
    inputs = torch.tensor([word_to_ix[word] for word in sentences[0]], dtype=torch.long).unsqueeze(0).to(device)
    tag_scores = model(inputs)
    predicted_tags = [list(tag_to_ix.keys())[tag] for tag in tag_scores[0].argmax(dim=1).cpu()]
    print(f"Sentence: {' '.join(sentences[0])}")
    print(f"Predicted POS Tags: {predicted_tags}")
    true_tags = [tag for tag in pos_tags[0]]
    print(f"Vraies étiquettes POS: {true_tags}")




# Charger les données de test
test_sentences, test_pos_tags = load_data("UD_French-Sequoia/fr_sequoia-ud-test.conllu")
# Prepare the test dataset
unk_count_test=0
nombre_mot=0
test_dataset_sentences = []
for sentence in test_sentences:
    indexed_sentence = []
    for word in sentence:
        nombre_mot+=1
        if word in word_to_ix:
            indexed_sentence.append(word_to_ix[word])
        else:
            indexed_sentence.append(word_to_ix['<UNK>'])
            unk_count_test += 1  # Increment the counter for each unknown word
    test_dataset_sentences.append(torch.tensor(indexed_sentence, dtype=torch.long))

# Préparer le DataLoader pour les données de test
test_dataset = POSDataset(test_sentences, test_pos_tags, word_to_ix, tag_to_ix)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Calculer l'accuracy
loss,accuracy = evaluate_model(model,test_data_loader,loss_function)
print(f"Test Accuracy : {accuracy:.4f}")