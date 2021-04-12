import torch
import torch.nn as nn

class Layer_Handler():
    '''
    Allows you to get the outputs at a BERT layer of a trained BertGrader

    AND has a separate method to pass an embedding through
    and remaining layers of the model and further for BERTGrader
    '''

    def __init__(self, trained_model, layer_num=1):
        trained_model.eval()
        self.model = trained_model
        self.layer_num = layer_num

    def get_layern_outputs(self, input_ids, attention_mask, device=torch.device('cpu')):
        '''
        Get output hidden states from nth layer
        '''
        # Need to extend mask for encoder - from HuggingFace implementation
        self.input_shape = input_ids.size()
        extended_attention_mask: torch.Tensor = self.model.encoder.get_extended_attention_mask(attention_mask, self.input_shape, device)

        hidden_states = self.model.encoder.embeddings(input_ids=input_ids)
        for layer_module in self.model.encoder.encoder.layer[:self.layer_num]:
            layer_outputs = layer_module(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]
        return hidden_states

    def pass_through_rest(self, hidden_states, attention_mask, device=torch.device('cpu')):
        '''
        Pass hidden states through remainder of BertGrader model
        after nth layer
        '''
        extended_attention_mask: torch.Tensor = self.model.encoder.get_extended_attention_mask(attention_mask, self.input_shape, device)

        for layer_module in self.model.encoder.encoder.layer[self.layer_num:]:
            layer_outputs = layer_module(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]

        head1 = self.model.apply_attn(hidden_states, attention_mask, self.model.attn1)
        head2 = self.model.apply_attn(hidden_states, attention_mask, self.model.attn2)
        head3 = self.model.apply_attn(hidden_states, attention_mask, self.model.attn3)
        head4 = self.model.apply_attn(hidden_states, attention_mask, self.model.attn4)

        all_heads = torch.cat((head1, head2, head3, head4), dim=1)

        h1 = self.model.layer1(all_heads).clamp(min=0)
        h2 = self.model.layer2(h1).clamp(min=0)
        y = self.model.layer3(h2).squeeze()
        return y
