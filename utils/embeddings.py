from transformers import AutoTokenizer, AutoModel
import torch


def get_word_embeddings(word):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    d1, d2 = len(word), len(word[0])
    flatten = [word if word not in {'-1', '-2', '-3'} else '' for people in word for word in people]
    inputs = tokenizer(flatten, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    embeddings = outputs.last_hidden_state[:, 0, :].squeeze()
    embeddings = embeddings.view(d1, d2, -1)

    return embeddings
    