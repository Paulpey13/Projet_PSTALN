import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from conllu import parse_incr
from collections import Counter

# Initialisation du tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Fonction pour charger les données
def load_data(conllu_file):
    sentences, pos_tags = [], []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            sentences.append([token['form'].lower() for token in sentence])
            pos_tags.append([token['upos'] for token in sentence])
    return sentences, pos_tags

# Dataset
class POSDataset(Dataset):
    def __init__(self, sentences, pos_tags, tag_to_ix, max_len=128):
        self.sentences = sentences
        self.pos_tags = pos_tags
        self.tag_to_ix = tag_to_ix
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        pos_tag = self.pos_tags[idx]

        # Encodage BERT
        encoding = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)

        # Encodage des POS tags
        tag_ids = [self.tag_to_ix.get(tag, 0) for tag in pos_tag]
        tag_ids = tag_ids[:self.max_len]  # Ensure tag_ids is not longer than max_len
        tag_ids += [0] * (self.max_len - len(tag_ids))  # Padding
        tag_ids = torch.tensor(tag_ids, dtype=torch.long)

        return input_ids, attention_mask, tag_ids


# Custom collate_fn pour DataLoader
def collate_fn(batch):
    input_ids, attention_masks, tag_ids = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    tag_ids = torch.stack(tag_ids)
    return input_ids, attention_masks, tag_ids

# Modèle
class BertPOSTagger(nn.Module):
    def __init__(self, tagset_size):
        super(BertPOSTagger, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, tagset_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        tag_scores = self.hidden2tag(sequence_output)
        return tag_scores

# Chargement des données et création des dictionnaires
sentences, pos_tags = load_data("UD_French-Sequoia/fr_sequoia-ud-dev.conllu")[:100]
tag_counts = Counter(tag for tags in pos_tags for tag in tags)
tag_to_ix = {tag: i for i, tag in enumerate(tag_counts)}

# Création du dataset et du DataLoader
dataset = POSDataset(sentences, pos_tags, tag_to_ix)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Initialisation du modèle et des paramètres d'entraînement
model = BertPOSTagger(len(tag_to_ix))
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # Boucle d'entraînement
# for epoch in range(3):  # Nombre d'époques
#     model.train()
#     total_loss = 0
#     for input_ids, attention_mask, targets in data_loader:
#         optimizer.zero_grad()
#         tag_scores = model(input_ids, attention_mask)
#         loss = loss_function(tag_scores.view(-1, len(tag_to_ix)), targets.view(-1))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}")

# Fonction pour calculer l'accuracy
def calculate_accuracy(preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the maximum score
    non_pad_elements = (y != tag_to_ix['<PAD>']).nonzero()  # assuming PAD is the padding tag
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

# Boucle d'entraînement avec calcul de l'accuracy
for epoch in range(3):  # Nombre d'époques
    model.train()
    total_loss, total_accuracy = 0, 0

    for input_ids, attention_mask, targets in data_loader:
        optimizer.zero_grad()
        tag_scores = model(input_ids, attention_mask)

        # Reshape pour CrossEntropyLoss
        tag_scores = tag_scores.view(-1, len(tag_to_ix))
        targets = targets.view(-1)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calcul de l'accuracy
        with torch.no_grad():
            accuracy = calculate_accuracy(tag_scores, targets)
            total_accuracy += accuracy.item()

    print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}, Accuracy: {total_accuracy / len(data_loader)}")
