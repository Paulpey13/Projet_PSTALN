#Classe pour encoder les target (donnée par le prof TP2)
class UDTagSet():
    def __init__(self):

        self.dico = {
        "ADJ"   : 0, 
        "ADP"   : 1,
        "ADV"   : 2,
        "AUX"   : 3,
        "CCONJ" : 4,
        "DET"   : 5,
        "NOUN"  : 6,
        "NUM"   : 7,
        "PRON"  : 8,
        "PROPN" : 9,
        "PUNCT" : 10,
        "SCONJ" : 11,
        "SYM"   : 12,
        "VERB"  : 13,
        "INTJ"  : 14,
        "X"     : 15,
        "_"     : 16
        }

        self.array = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "NOUN", "NUM", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "INTJ", "X", "_"]

    def tag2code(self, tag):
        if tag not in self.dico :
            return None
        else :
            return self.dico[tag]

    def code2tag(self, code):
        if code > 16 or code < 0:
            return None
        else :
            return self.array[code]

    def size(self):
        return 17

    def tags(self):
        return self.array;
