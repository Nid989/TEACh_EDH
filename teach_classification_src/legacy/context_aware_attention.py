from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextAwareAttention(nn.Module):

    def __init__(self,
                 dim_model: int,
                 dim_context: int,
                 dropout_rate: Optional[float]=0.0):
        super(ContextAwareAttention, self).__init__()

        self.dim_model = dim_model
        self.dim_context = dim_context
        self.dropout_rate = dropout_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.attention_layer = nn.MultiheadAttention(embed_dim=self.dim_model,
                                                     num_heads=1,
                                                     dropout=self.dropout_rate,
                                                     bias=True,
                                                     add_zero_attn=False,
                                                     batch_first=True,
                                                     device=self.device)


        self.u_k = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_k = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_k = nn.Linear(self.dim_model, 1, bias=False)

        self.u_v = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_v = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_v = nn.Linear(self.dim_model, 1, bias=False)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                context: Optional[torch.Tensor]=None):

        key_context = self.u_k(context)
        value_context = self.u_v(context)

        lambda_k = F.sigmoid(self.w1_k(k) + self.w2_k(key_context))
        lambda_v = F.sigmoid(self.w1_v(v) + self.w2_v(value_context))

        k_cap = (1 - lambda_k) * k + lambda_k * key_context
        v_cap = (1 - lambda_v) * v + lambda_v * value_context

        attention_output, _ = self.attention_layer(query=q,
                                                   key=k_cap,
                                                   value=v_cap)
        return attention_output