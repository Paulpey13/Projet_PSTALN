from conllu import parse_incr
import torch
import time

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

# Évaluer les performances du modèle
def calculate_accuracy(preds, y, pad_idx):
    max_preds = preds.argmax(dim=1, keepdim=True)
    non_pad_elements = (y != pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])

    # Assurez-vous que le tensor pour le calcul de la précision est sur le même appareil que 'y'
    correct_count = correct.sum()
    non_pad_count = torch.tensor([y[non_pad_elements].shape[0]], dtype=torch.float, device=y.device)

    return correct_count / non_pad_count


def evaluate(model, data_loader, loss_function, tag_to_ix, device):
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

# Tester le modèle sur les données de test
def test_performance(model, test_loader, loss_function, tag_to_ix, device):
    test_loss, test_accuracy = evaluate(model, test_loader, loss_function, tag_to_ix, device)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    return test_loss, test_accuracy

# Entraînement du modèle
def train(model, data_loader, loss_function, optimizer, tag_to_ix, epochs, device):
    start_time = time.time()
    for epoch in range(epochs):  # Nombre d'époques
        model.train()
        total_loss, total_accuracy = 0, 0

        for input_ids, attention_mask, targets in data_loader:
            input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
            optimizer.zero_grad()
            tag_scores = model(input_ids, attention_mask)
            loss = loss_function(tag_scores.view(-1, len(tag_to_ix)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calcul de l'accuracy
            with torch.no_grad():
                accuracy = calculate_accuracy(tag_scores.view(-1, len(tag_to_ix)), targets.view(-1), tag_to_ix['PAD'])
                total_accuracy += accuracy.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}, Accuracy: {total_accuracy / len(data_loader)}")
        print(f"Time elapsed: {(time.time() - start_time)/60} min")
