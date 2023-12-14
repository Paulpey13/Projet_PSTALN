import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from conllu import parse_incr
from collections import Counter
import utils
import models



# Chargement des données et création des dictionnaires
sentences, pos_tags = utils.load_data("UD_French-Sequoia/fr_sequoia-ud-train.conllu",pos=True)
#réduire data pour phase de test
limit=100
sentences=sentences[0:limit]
pos_tags=pos_tags[0:limit]


tag_counts = Counter(tag for tags in pos_tags for tag in tags)
tag_to_ix = {tag: i+1 for i, tag in enumerate(tag_counts)}  # Décalage pour le padding
tag_to_ix['PAD'] = 0  # Ajout du tag de padding

#Création du dataset et du DataLoader
dataset = models.BertPOSDataset(sentences, pos_tags, tag_to_ix)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=models.bert_collate_fn)


#Modele et parametre d'entrainement
model = models.BertPOSTagger(len(tag_to_ix))
loss_function = nn.CrossEntropyLoss(ignore_index=tag_to_ix['PAD'])  # Ignorer les tags de padding dans le calcul de la loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Boucle d'entraînement avec calcul de l'accuracy
epochs=3
utils.train(model, data_loader, loss_function, optimizer, tag_to_ix,epochs)


test_sentences, test_pos_tags = utils.load_data("UD_French-Sequoia/fr_sequoia-ud-test.conllu",pos=True)
test_sentences=test_sentences[0:limit]
test_pos_tags=test_pos_tags[0:limit]

# Créer le DataLoader pour l'ensemble de test
test_dataset = models.BertPOSDataset(test_sentences, test_pos_tags, tag_to_ix)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=models.bert_collate_fn)

# Évaluer le modèle sur l'ensemble de test
utils.test_performance(model, test_loader, loss_function, tag_to_ix)

#Best accuracy pour le moment : 
#Test Loss: 0.36735320182031084, Test Accuracy: 0.886901938601544