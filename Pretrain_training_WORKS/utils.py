from conllu import parse_incr
import torch
import time

# Détection du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger les données
def load_data(conllu_file, pos=False, lem=False, morph=False):
    sentences = []
    pos_tags = [] if pos else None
    lemmas = [] if lem else None
    morph_traits = [] if morph else None

    with open(conllu_file, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            sentences.append([token['form'].lower() for token in sentence])
            if pos:
                pos_tags.append([token['upos'] for token in sentence])
            if lem:
                lemmas.append([token['lemma'] for token in sentence])
            if morph:
                morph_traits.append(['|'.join([f"{k}={v}" for k, v in token['feats'].items()]) if token['feats'] else '_' for token in sentence])

    return tuple(data for data in [sentences, pos_tags, lemmas, morph_traits] if data is not None)

# Evaluer les performances du model
def calculate_accuracy(preds, y, pad_idx):
    max_preds = preds.argmax(dim=1, keepdim=True)
    non_pad_elements = (y != pad_idx).nonzero(as_tuple=True)
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)

def evaluate(model, data_loader, loss_function, tag_to_ix):
    model.eval()
    total_loss, total_accuracy = 0, 0

    with torch.no_grad():
        for input_ids, attention_mask, targets in data_loader:
            input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
            tag_scores = model(input_ids, attention_mask)
            loss = loss_function(tag_scores.view(-1, len(tag_to_ix)), targets.view(-1))
            total_loss += loss.item()
            accuracy = calculate_accuracy(tag_scores.view(-1, len(tag_to_ix)), targets.view(-1), tag_to_ix['PAD'])
            total_accuracy += accuracy.item()

    return total_loss / len(data_loader), total_accuracy / len(data_loader)

# Tester le model sur les données de test
def test_performance(model, test_loader, loss_function, tag_to_ix):
    test_loss, test_accuracy = evaluate(model, test_loader, loss_function, tag_to_ix)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    return test_loss, test_accuracy

# Entrainement du model Bert avec validation et arrêt anticipé
def train2(model, train_loader, valid_loader, loss_function, optimizer, tag_to_ix, max_epochs, early_stopping_rounds):
    start_time = time.time()
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    epoch_used = 0

    for epoch in range(max_epochs):
        model.train()
        total_train_loss, total_train_accuracy = 0, 0

        for input_ids, attention_mask, targets in train_loader:
            input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
            optimizer.zero_grad()
            tag_scores = model(input_ids, attention_mask)
            loss = loss_function(tag_scores.view(-1, len(tag_to_ix)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            accuracy = calculate_accuracy(tag_scores.view(-1, len(tag_to_ix)), targets.view(-1), tag_to_ix['PAD'])
            total_train_accuracy += accuracy.item()

        train_loss = total_train_loss / len(train_loader)
        train_accuracy = total_train_accuracy / len(train_loader)

        valid_loss, valid_accuracy = evaluate(model, valid_loader, loss_function, tag_to_ix)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Train Accuracy: {train_accuracy}, Valid Accuracy: {valid_accuracy}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            epoch_used = epoch + 1
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_rounds:
                print("Early stopping triggered")
                break

        print(f"Time elapsed: {(time.time() - start_time)/60} min")

    return train_loss, train_accuracy, valid_loss, valid_accuracy, epoch_used
