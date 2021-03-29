import torch
import torch.nn as nn
from transformers import BertModel

class BERTGrader(nn.Module):
    '''
    BERT encoder, multihead attention and regression head
    '''
    def __init__(self, h1_dim=600, h2_dim=20, embedding_size=768):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')

        self.attn1 = torch.nn.Linear(embedding_size, embedding_size)
        self.attn2 = torch.nn.Linear(embedding_size, embedding_size)
        self.attn3 = torch.nn.Linear(embedding_size, embedding_size)
        self.attn4 = torch.nn.Linear(embedding_size, embedding_size)

        self.layer1 = torch.nn.Linear(embedding_size*4, h1_dim)
        self.layer2 = torch.nn.Linear(h1_dim, h2_dim)
        self.layer3 = torch.nn.Linear(h2_dim, 1)

    def forward(self, input_ids, attention_mask):
        '''
        input_ids: Tensor [N x L]
            Token ids

        attention_mask: Tensor [N x L]
            Utterance, token level mask

            where:
                N is batch size
                L is the maximum number of tokens in sequence (typically 512)

        '''
        output = self.encoder(input_ids, attention_mask)
        word_embeddings = output.last_hidden_state
        
        head1 = self._apply_attn(word_embeddings, attention_mask, self.attn1)
        head2 = self._apply_attn(word_embeddings, attention_mask, self.attn2)
        head3 = self._apply_attn(word_embeddings, attention_mask, self.attn3)
        head4 = self._apply_attn(word_embeddings, attention_mask, self.attn4)

        all_heads = torch.cat((head1, head2, head3, head4), dim=1)

        h1 = self.layer1(all_heads).clamp(min=0)
        h2 = self.layer2(h1).clamp(min=0)
        y = self.layer3(h2).squeeze()
        return y

    def _apply_attn(self, embeddings, mask, weights_transformation):
        '''
        Self-attention variant to get sentence embedding
        '''
        transformed_values = weights_transformation(embeddings)
        score = torch.einsum('ijk,ijk->ij', embeddings, transformed_values)
        T = nn.Tanh()
        score_T = T(score) * mask
        # use mask to convert padding scores to -inf (go to zero after softmax)
        mask_complement = 1 - mask
        inf_mask = mask_complement * (-10000)
        scaled_score = score_T + inf_mask
        # Normalize with softmax
        SM = nn.Softmax(dim=1)
        w = SM(scaled_score)
        repeated_w = torch.unsqueeze(w, -1).expand(-1,-1, embeddings.size(-1))
        x_attn = torch.sum(embeddings*repeated_w, dim=1)
        return x_attn
