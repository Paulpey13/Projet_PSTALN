import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# Spécification du modèle BERT
bert_model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Dataset
class BertPOSDataset(Dataset):
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
        tag_ids = tag_ids[:self.max_len]
        tag_ids += [self.tag_to_ix['PAD']] * (self.max_len - len(tag_ids))
        tag_ids = torch.tensor(tag_ids, dtype=torch.long)

        return input_ids, attention_mask, tag_ids

def bert_collate_fn(batch):
    input_ids, attention_masks, tag_ids = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    tag_ids = torch.stack(tag_ids)
    return input_ids, attention_masks, tag_ids

# Modèle
class BertPOSTagger(nn.Module):
    def __init__(self, tagset_size):
        super(BertPOSTagger, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, tagset_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        tag_scores = self.hidden2tag(sequence_output)
        return tag_scores
