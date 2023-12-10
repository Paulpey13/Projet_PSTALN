from conllu import parse
from conllu import parse_incr

def extractVocab(conlluFileName):
    dicoVocab = {}
    data_file = open(conlluFileName, "r", encoding="utf-8")
    vocabSize = 0

    for sentence in parse_incr(data_file):
        for token in sentence :
            form = token['form']
            if form not in dicoVocab:
                dicoVocab[form] = vocabSize
                vocabSize += 1
    data_file.close()
    return dicoVocab

print(extractVocab("UD_French-Sequoia/fr_sequoia-ud-dev.conllu"))