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

def load_data(conllu_file, char_to_ix, max_word_len):
    sentences, pos_tags = [], []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            char_sentences = []
            for token in sentence:
                chars = [char_to_ix.get(c, char_to_ix['<UNK>']) for c in token['form'].lower()]
                chars = chars[:max_word_len] + [char_to_ix['<PAD>']] * (max_word_len - len(chars))
                char_sentences.append(chars)
            sentences.append(char_sentences)
            pos_tags.append([token['upos'] for token in sentence])
    return sentences, pos_tags


#Init le POS


class POSDataset(Dataset):
    def __init__(self, sentences, pos_tags, tag_to_ix, max_word_len):
        # sentences: List of lists, where each inner list is a sequence of character indices for a sentence
        # pos_tags: List of lists, where each inner list contains POS tags for the corresponding sentence
        # tag_to_ix: Dictionary mapping POS tags to unique indices
        # max_word_len: The maximum length of words to standardize word lengths in sentences

        # Convert sentences to PyTorch tensors
        self.sentences = [torch.tensor([self.pad_word(word, max_word_len, char_to_ix['<PAD>']) for word in sentence], dtype=torch.long) for sentence in sentences]
        
        # Convert POS tags to indices and then to PyTorch tensors
        self.tags = [torch.tensor([tag_to_ix[tag] for tag in tag_list], dtype=torch.long) for tag_list in pos_tags]

    def pad_word(self, word, max_word_len, pad_index):
        # Function to pad or truncate each word to the same length
        return word[:max_word_len] + [pad_index] * (max_word_len - len(word))

    def __len__(self):
        # Returns the total number of sentences in the dataset
        return len(self.sentences)

    def __getitem__(self, idx):
        # Retrieves the sentence and its corresponding POS tags at the given index
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


class CharCNNEmbedding(nn.Module):
    def __init__(self, num_chars, embedding_dim, num_filters, kernel_size):
        super(CharCNNEmbedding, self).__init__()
        self.char_embedding = nn.Embedding(num_chars, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, num_filters, kernel_size, padding=kernel_size // 2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.embedding_dim = num_filters

    def forward(self, x):
        # x: [batch_size, seq_len, word_len]
        batch_size, seq_len, word_len = x.size(0), x.size(1), x.size(2)
        x = self.char_embedding(x)  # [batch_size, seq_len, word_len, embedding_dim]
        x = x.view(batch_size * seq_len, word_len, -1).transpose(1, 2)  # [batch_size * seq_len, embedding_dim, word_len]
        x = self.conv(x)  # Apply Conv1d
        x = self.pool(x).squeeze(-1)  # [batch_size * seq_len, num_filters]
        x = x.view(batch_size, seq_len, -1)  # Reshape back to [batch_size, seq_len, num_filters]
        return x

class POSTransformerModel(nn.Module):
    def __init__(self, num_chars, char_embedding_dim, num_filters, kernel_size, nhead, nhid, nlayers, tagset_size):
        super(POSTransformerModel, self).__init__()
        self.char_cnn_embedding = CharCNNEmbedding(num_chars, char_embedding_dim, num_filters, kernel_size)
        self.pos_encoder = PositionalEncoding(num_filters)
        transformer_layers = nn.TransformerEncoderLayer(num_filters, nhead, nhid, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layers, nlayers)
        self.hidden2tag = nn.Linear(num_filters, tagset_size)

    def forward(self, sentence):
        embeds = self.char_cnn_embedding(sentence)  # Replace word embedding
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



def load_data_1(conllu_file):
    sentences, pos_tags = [], []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            sentences.append([token['form'].lower() for token in sentence])
            pos_tags.append([token['upos'] for token in sentence])
    return sentences, pos_tags


# Define load_data function
def load_data(conllu_file, char_to_ix, max_word_len):
    sentences, pos_tags = [], []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            char_sentences = []
            for token in sentence:
                chars = [char_to_ix.get(c, char_to_ix['<UNK>']) for c in token['form'].lower()]
                chars = chars[:max_word_len] + [char_to_ix['<PAD>']] * (max_word_len - len(chars))
                char_sentences.append(chars)
            sentences.append(char_sentences)
            pos_tags.append([token['upos'] for token in sentence])
    return sentences, pos_tags



batch_size=64
epochs=20
# Load data using the load_data_1 function
sentences, pos_tags = load_data_1("UD_French-Sequoia/fr_sequoia-ud-train.conllu")

# Create character and tag mappings
char_counts = Counter(char for sentence in sentences for word in sentence for char in word)
char_to_ix = {char: i for i, char in enumerate(char_counts, start=2)}
char_to_ix['<PAD>'], char_to_ix['<UNK>'] = 0, 1  # Padding and unknown character

tag_counts = Counter(tag for tags in pos_tags for tag in tags)
tag_to_ix = {tag: i for i, tag in enumerate(tag_counts)}

max_word_len = max(len(word) for sentence in sentences for word in sentence)

# Now load the data in the desired format using the load_data function
train_sentences, train_pos_tags = load_data("UD_French-Sequoia/fr_sequoia-ud-train.conllu", char_to_ix, max_word_len)
validation_sentences, validation_pos_tags = load_data("UD_French-Sequoia/fr_sequoia-ud-dev.conllu", char_to_ix, max_word_len)

# Rest of your code for Dataset, DataLoader, Model initialization, etc.

# Dataset and DataLoader
dataset = POSDataset(train_sentences, train_pos_tags, tag_to_ix, max_word_len)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

validation_dataset = POSDataset(validation_sentences, validation_pos_tags, tag_to_ix, max_word_len)
validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Model initialization
num_chars = len(char_to_ix)
char_embedding_dim = 256
num_filters = 256
kernel_size = 3
nhead = 4
nhid = 512
nlayers = 2
tagset_size = len(tag_to_ix)

model = POSTransformerModel(num_chars, char_embedding_dim, num_filters, kernel_size, nhead, nhid, nlayers, tagset_size)

# Loss and Optimizer
loss_function = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.SGD(model.parameters(), lr=0.01)


patience = 2  # Nombre d'époques à attendre après la dernière amélioration de la loss de validation
best_val_accuracy = 0
epochs_no_improve = 0

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

model.eval()
with torch.no_grad():
    # Convert the first sentence in the dataset to character indices
    char_indices = [[char_to_ix.get(char, char_to_ix['<UNK>']) for char in word] for word in sentences[0]]
    char_indices = [word[:max_word_len] + [char_to_ix['<PAD>']] * (max_word_len - len(word)) for word in char_indices]

    # Convert to tensor and add batch dimension
    inputs = torch.tensor(char_indices, dtype=torch.long).unsqueeze(0).to(device)

    # Get tag scores from the model
    tag_scores = model(inputs)
    predicted_tags = [list(tag_to_ix.keys())[tag] for tag in tag_scores[0].argmax(dim=1).cpu()]
    
    print(f"Sentence: {' '.join(sentences[0])}")
    print(f"Predicted POS Tags: {predicted_tags}")
    true_tags = [tag for tag in pos_tags[0]]
    print(f"Vraies étiquettes POS: {true_tags}")



test_sentences, test_pos_tags = load_data("UD_French-Sequoia/fr_sequoia-ud-test.conllu", char_to_ix, max_word_len)


test_dataset = POSDataset(test_sentences, test_pos_tags, tag_to_ix, max_word_len)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# Calculer l'accuracy
loss,accuracy = evaluate_model(model,test_data_loader,loss_function)
print(f"Test Accuracy : {accuracy:.4f}")