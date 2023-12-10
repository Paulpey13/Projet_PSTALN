#Baseline LSTM (pour changer du GRU vu en TP)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from conllu import parse_incr
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

#Lecture des donnnées
def load_data(conllu_file):
    sentences, pos_tags = [], []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            sentences.append([token['form'].lower() for token in sentence])
            pos_tags.append([token['upos'] for token in sentence])
    return sentences, pos_tags

#Importer les pos
class POSDataset(Dataset):
    def __init__(self, sentences, tags, word_to_ix, tag_to_ix):
        self.sentences = [torch.tensor([word_to_ix[word] for word in sentence], dtype=torch.long) for sentence in sentences]
        self.tags = [torch.tensor([tag_to_ix[tag] for tag in tag_list], dtype=torch.long) for tag_list in tags]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]

#Padding de sequences
def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=-1)
    return sentences_padded, tags_padded

#classe du model (ici lstm mais on peut changer cetait pour test un truc different du TP)
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

#load les données
sentences, pos_tags = load_data("UD_French-Sequoia/fr_sequoia-ud-dev.conllu")

#init du vocab
word_counts = Counter(word for sentence in sentences for word in sentence)
word_to_ix = {word: i+1 for i, word in enumerate(word_counts)}  # +1 pour le padding
word_to_ix['<PAD>'] = 0

tag_counts = Counter(tag for tags in pos_tags for tag in tags)
tag_to_ix = {tag: i for i, tag in enumerate(tag_counts)}

#parametres
embedding_dim = 64
hidden_dim = 128
epochs=50
batch_size=2

# données et data loader
dataset = POSDataset(sentences, pos_tags, word_to_ix, tag_to_ix)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

#init du model
model = POSModel(len(word_to_ix), embedding_dim, hidden_dim, len(tag_to_ix))
loss_function = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

#training
for epoch in range(epochs):  #nb d'epochs
    total_loss = 0
    for sentence_in, targets in data_loader:
        model.zero_grad()
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores.view(-1, len(tag_to_ix)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")


#exemple de pred
with torch.no_grad():
    inputs = torch.tensor([word_to_ix[word] for word in sentences[0]], dtype=torch.long).unsqueeze(0)
    tag_scores = model(inputs)
    predicted_tags = [list(tag_to_ix.keys())[tag] for tag in tag_scores[0].argmax(dim=1)]
    print(f"Sentence: {' '.join(sentences[0])}")
    print(f"Predicted POS Tags: {predicted_tags}")
    true_tags = [tag for tag in pos_tags[0]]
    print(f"Vraies étiquettes POS: {true_tags}")


#Evaluations du modele : ici accuracy mais voir ce qui est le plus important
def calculate_accuracy(true_tags, pred_tags):
    correct = sum(t1 == t2 for t1, t2 in zip(true_tags, pred_tags))
    return correct / len(true_tags)

#evaluation du modele sur l'ensemble des données ()
model.eval()
all_true_tags = []
all_predicted_tags = []
with torch.no_grad():
    for sentences, tags in data_loader:
        tag_scores = model(sentences)
        predicted = torch.argmax(tag_scores, dim=2)
        all_true_tags.extend(tags.flatten().tolist())
        all_predicted_tags.extend(predicted.flatten().tolist())
filtered_true_tags = [tag for tag in all_true_tags if tag != -1]
filtered_predicted_tags = [all_predicted_tags[i] for i, tag in enumerate(all_true_tags) if tag != -1]
#Calculer l'accuracy
accuracy = calculate_accuracy(filtered_true_tags, filtered_predicted_tags)
print(f"Accuracy: {accuracy:.4f}")
