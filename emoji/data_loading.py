import pandas as pd
import time
import torch.utils.data as data

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer

class PretrainDataset(data.Dataset):
    def __init__(self, split: str, max_length: int):
        super().__init__()
        self.split: str = split
        self.max_length: int = max_length
        self.data: pd.DataFrame = self.split_data()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenize()
    
    def split_data(self):
        self.data_frame = pd.DataFrame(load_dataset("LabHC/moji", split="train"))
        train_df, remaining_df = train_test_split(self.data_frame, test_size=0.2, random_state=42)
        validation_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)
        if self.split == "train":
            return train_df
        elif self.split == "valid":
            return validation_df
        else:
            return test_df
    
    def tokenize(self):
        self.dict = self.tokenizer.batch_encode_plus(list(self.data["text"]), padding="max_length", max_length=self.max_length,
                                   pad_to_max_length=True, return_attention_mask=True, truncation=True, return_tensors="pt")
        self.tokens = self.dict["input_ids"]
        self.masks = self.dict["attention_mask"]
        self.labels = list(self.data["label"])
    
    def __getitem__(self, index:int):
        tokens = self.tokens[index]
        mask = self.masks[index]
        label = self.labels[index]
        return tokens, mask, label
    
    def __len__(self):
        return len(self.labels)

class Augdataset(data.Dataset):
    def __init__(self, split: str, max_length: int):
        super().__init__()
        self.split: str = split
        self.max_length: int = max_length
        self.data: pd.DataFrame = self.split_data()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenize()
        
    def split_data(self):
        self.data_frame = pd.read_csv("counter_dataset.csv")
        train_df, remaining_df = train_test_split(self.data_frame, test_size=0.2, random_state=42)
        validation_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)  
        if self.split == "train":
            return train_df
        elif self.split == "valid":
            return validation_df
        else:
            return test_df      
    
    def tokenize(self):
        self.dict = self.tokenizer.batch_encode_plus(list(self.data["text"]), padding="max_length", max_length=self.max_length,
                                    pad_to_max_length=True, return_attention_mask=True, truncation=True, return_tensors="pt")
        self.c_dict = self.tokenizer.batch_encode_plus(list(self.data["counter text"]), padding="max_length", max_length=self.max_length,
                                    pad_to_max_length=True, return_attention_mask=True, truncation=True, return_tensors="pt")
        self.tokens = self.dict["input_ids"]
        self.masks = self.dict["attention_mask"]
        self.c_tokens = self.c_dict["input_ids"]
        self.c_masks = self.c_dict["attention_mask"]
        self.labels = list(self.data["label"])
        self.sas = list(self.data["sa"])
    
    def __getitem__(self, index: int):
        tokens = self.tokens[index]
        mask = self.masks[index]
        c_tokens = self.c_tokens[index]
        c_mask = self.c_masks[index]
        label = self.labels[index]
        sa = self.sas[index]
        return tokens, mask, c_tokens, c_mask, label, sa
    
    def __len__(self):
        return len(self.data)
