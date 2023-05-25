import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch

class BERTProcessor():
    """ Computes BERT embeddings from the instructions on disk 
    """

    cls_uuid: str = "rxr_instruction"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cpu')

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)


    # def _get_observation_space(self, *args: Any, **kwargs: Any):
    #     return spaces.Box(
    #         low=np.finfo(np.float).min,
    #         high=np.finfo(np.float).max,
    #         shape=(512, 768),
    #         dtype=np.float,
    #     )

    def get_instruction_embeddings(self, input_text):
        token_ids = self.tokenizer.encode(input_text)

        with torch.no_grad():
            token_ids = torch.tensor(token_ids).unsqueeze(0).to(self.device)

            embedding = self.model(token_ids)['last_hidden_state']
            embedding = embedding.squeeze(0).detach().cpu().numpy()

        feats = np.zeros((512, 768), dtype=np.float32)
        s = embedding.shape
        feats[: s[0], : s[1]] = embedding
        return feats



if __name__ =='__main__':
    processor = BERTProcessor()

    input_text = "go forward then move left"

    feats = processor.get_instruction_embeddings(input_text)
    observations['rxr_instruction'] = feats