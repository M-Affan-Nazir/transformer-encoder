import csv
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch



def tokenize(input,target):
   
    #Dictionary:
    dictionary = {}
    index = 0
    for text in input:
        # Tokenize by splitting on spaces and removing punctuation (simple approach)
        words = text.translate(str.maketrans('', '', ',.!?')).lower().split()
        for word in words:
            if word not in dictionary:
                dictionary[word] = index
                index += 1


    #Tokenizing Sentences (replacing words in each sentence; with its token index in dictionary):
    tokenized_texts = []
    for text in input:
        words = text.translate(str.maketrans('', '', ',.!?')).lower().split()
        token_sequence = [dictionary.get(word) for word in words]
        tokenized_texts.append(token_sequence)



    #Shuffling:
    zipped_lists = list(zip(tokenized_texts, target))
    random.shuffle(zipped_lists)
    tokenized_texts, target = zip(*zipped_lists)
    tokenized_texts = list(tokenized_texts)
    target = list(sents)

    return [tokenized_texts, sents, len(dictionary)]

tokenized_texts, sents, vocabSize = tokenize()

class TextDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are suitable for your loss function
        }
    
def collate_batch(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = (input_ids_padded != 0).long()
    
    labels = torch.stack(labels)    # Ensure labels are properly formed tensor
    
    return {'input_ids': input_ids_padded, 'attention_mask': attention_masks, 'labels': labels}


train_dataset = TextDataset(input_ids=tokenized_texts, labels=sents)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)