from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from teach_encoder_decoder_src.modeling.context_aware_attention import ContextAwareAttention

class DuoModalityProcessor(nn.Module):

    def __init__(self,
                 dim_model: int,
                 dropout_rate: int,
                 util_config: dict):
        super(DuoModalityProcessor, self).__init__()

        self.dropout_rate = dropout_rate

        self.action_context_transform = nn.Linear(util_config["SOURCE_ACTION_MAX_LEN"],
                                                  util_config["SOURCE_DIALOG_MAX_LEN"], bias=False)
        
        self.action_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                              dim_context=util_config["ACTION_DIM"],
                                                              dropout_rate=dropout_rate)
        
        self.action_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)
        
    def forward(self,
                text_input: torch.Tensor,
                action_context: Optional[torch.Tensor]=None):

        # Audio as Context for Attention
        action_context = action_context.permute(0, 2, 1)
        action_context = self.action_context_transform(action_context)
        action_context = action_context.permute(0, 2, 1)

        action_out = self.action_context_attention(q=text_input,
                                                    k=text_input,
                                                    v=text_input,
                                                    context=action_context)

        # Global Information Fusion Mechanism
        weight_a = F.sigmoid(self.action_gate(torch.cat((action_out, text_input), dim=-1)))

        output = self.final_layer_norm(text_input +
                                       weight_a * action_out)

        return output