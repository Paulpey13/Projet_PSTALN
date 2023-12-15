from conllu import parse_incr
import torch
import time

#Charger les données
#Mettre true ce qu'on veut charger, ne pas spécifier sinon
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

    #Retourner seulement les liste non nulle
    return tuple(data for data in [sentences, pos_tags, lemmas, morph_traits] if data is not None)





#Evaluer les performances du model :
def calculate_accuracy(preds, y, pad_idx):
    max_preds = preds.argmax(dim=1, keepdim=True)
    non_pad_elements = (y != pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

def evaluate(model, data_loader, loss_function, tag_to_ix):
    model.eval()
    total_loss, total_accuracy = 0, 0

    with torch.no_grad():
        for input_ids, attention_mask, targets in data_loader:
            tag_scores = model(input_ids, attention_mask)
            loss = loss_function(tag_scores.view(-1, len(tag_to_ix)), targets.view(-1))
            total_loss += loss.item()
            accuracy = calculate_accuracy(tag_scores.view(-1, len(tag_to_ix)), targets.view(-1), tag_to_ix['PAD'])
            total_accuracy += accuracy.item()

    return total_loss / len(data_loader), total_accuracy / len(data_loader)

#Tester le model sur les données de test :

def test_performance(model, test_loader, loss_function, tag_to_ix):
    test_loss, test_accuracy = evaluate(model, test_loader, loss_function, tag_to_ix)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    return test_loss, test_accuracy


### Entrainement du model Bert (voir si c adapté a tous les models)


def train(model, data_loader, loss_function, optimizer, tag_to_ix,epochs):
    start_time = time.time()
    for epoch in range(epochs):  # Nombre d'époques
        model.train()
        total_loss, total_accuracy = 0, 0

        for input_ids, attention_mask, targets in data_loader:
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