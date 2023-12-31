import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

class PretrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig(num_hidden_layers=3)
        self.encoder = BertModel(config)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)
    
    def forward(self, tokens, masks):
        rep = self.encoder(input_ids=tokens, attention_mask=masks)[1]
        output = self.classifier(rep)
        return output

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
    
    def forward(self, tokens, masks, c_tokens, c_masks):
        #rep_1 = self.encoder(input_ids=tokens, attention_mask=masks).pooler_output
        rep_1 = self.encoder(input_ids=tokens, attention_mask=masks)[1]
        rep_2 = self.encoder(input_ids=c_tokens, attention_mask=c_masks)[1]
        rep = (rep_1 + rep_2) / 2
        return rep_1, rep_2

class MultiTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
    
    def forward(self, tokens, mask, c1_tokens, c1_mask, c2_tokens, c2_mask, c3_tokens, c3_mask,
                c4_tokens, c4_mask, c5_tokens, c5_mask):
        rep_0 = self.encoder(input_ids=tokens, attention_mask=mask)[1]
        rep_1 = self.encoder(input_ids=c1_tokens, attention_mask=c1_mask)[1]
        rep_2 = self.encoder(input_ids=c2_tokens, attention_mask=c2_mask)[1]
        rep_3 = self.encoder(input_ids=c3_tokens, attention_mask=c3_mask)[1]
        rep_4 = self.encoder(input_ids=c4_tokens, attention_mask=c4_mask)[1]
        rep_5 = self.encoder(input_ids=c5_tokens, attention_mask=c5_mask)[1]
        rep = (5 * rep_0 + rep_1 + rep_2 + rep_3 + rep_4 + rep_5) / 10
        return rep_0, rep

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
    
    def forward(self, rep):
        return self.classifier(rep)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #config = BertConfig(num_hidden_layers=1, hidden_size=16, num_attention_heads=1, intermediate_size=32, hidden_dropout_prob=0.5)
        config = BertConfig(num_hidden_layers=3)
        self.encoder = BertModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 2)
        )
    
    def forward(self, tokens, masks, c_tokens, c_masks):
        rep = self.encoder(input_ids=tokens, attention_mask=masks).pooler_output
        c_rep = self.encoder(input_ids=c_tokens, attention_mask=c_masks).pooler_output
        output = self.classifier(rep + c_rep)
        return output
    
class SimpleModel(nn.Module):
    def __init__(self, vocab_size: int =50000, embed_dim: int =768, num_classes: int =2):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, text):
        print("type text = {}".format(text))
        embedded = self.embedding(text)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        out = F.relu(self.fc1(pooled))
        return self.fc2(out)