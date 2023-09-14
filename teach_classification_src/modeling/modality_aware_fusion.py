from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from teach_classification_src.modeling.context_aware_attention import ContextAwareAttention

class MAF(nn.Module):

    def __init__(self,
                 dim_model: int,
                 dropout_rate: int,
                 util_config: dict):
        super(MAF, self).__init__()
        self.dropout_rate = dropout_rate

        self.game_plan_context_transform = nn.Linear(util_config["SOURCE_GAME_PLAN_MAX_LEN"],
                                                  util_config["SOURCE_DIALOG_MAX_LEN"], bias=False)
        self.visual_context_transform = nn.Linear(util_config["SOURCE_VISUAL_MAX_LEN"],
                                                  util_config["SOURCE_DIALOG_MAX_LEN"], bias=False)

        self.game_plan_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                              dim_context=dim_model,
                                                              dropout_rate=dropout_rate)
        self.visual_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                              dim_context=util_config["SOURCE_VISUAL_DIM"],
                                                              dropout_rate=dropout_rate)
        self.action_gate = nn.Linear(2*dim_model, dim_model)
        self.visual_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

    def forward(self,
                text_input: torch.Tensor,
                game_plan_context: Optional[torch.Tensor]=None,
                visual_context: Optional[torch.Tensor]=None):

        # Audio as Context for Attention
        game_plan_context = game_plan_context.permute(0, 2, 1)
        game_plan_context = self.game_plan_context_transform(game_plan_context)
        game_plan_context = game_plan_context.permute(0, 2, 1)

        # Video as Context for Attention
        visual_context = visual_context.permute(0, 2, 1)
        visual_context = self.visual_context_transform(visual_context)
        visual_context = visual_context.permute(0, 2, 1)

        action_out = self.game_plan_context_attention(q=text_input,
                                                    k=text_input,
                                                    v=text_input,
                                                    context=game_plan_context)


        video_out = self.visual_context_attention(q=text_input,
                                                  k=text_input,
                                                  v=text_input,
                                                  context=visual_context)

        # Global Information Fusion Mechanism
        weight_a = F.sigmoid(self.action_gate(torch.cat((action_out, text_input), dim=-1)))
        weight_v = F.sigmoid(self.visual_gate(torch.cat((video_out, text_input), dim=-1)))

        output = self.final_layer_norm(text_input +
                                       weight_a * action_out +
                                       weight_v * video_out)

        return output