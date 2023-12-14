from conllu import parse_incr



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
