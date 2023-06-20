import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch

class BERTProcessor():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)


    def get_embeddings(self, batch_word):
        d1, d2, d3 = len(batch_word), len(batch_word[0]), len(batch_word[0][0])
        flatten = [word if word not in {'-1', '-2', '-3'} else '' for batch in batch_word for people in batch for word in people]
        inputs = self.tokenizer(flatten, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state[:, 0, :].squeeze()
        embeddings = embeddings.view(d1, d2, d3, -1)
            
        return embeddings



#if __name__ =='__main__':
    