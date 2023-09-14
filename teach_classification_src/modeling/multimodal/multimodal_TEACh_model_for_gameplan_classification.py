import gc
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartPretrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from teach_classification_src.modeling.multimodal.multimodal_TEACh_model import MultimodalTEAChModel

@dataclass
class MultimodalTokenClassifierOutput(TokenClassifierOutput):
    dialog_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    game_plan_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    visual_hidden_states: Optional[torch.FloatTensor] = None
    dialog_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    game_plan_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    dialog_encoder_attentions: Optional[torch.FloatTensor] = None
    game_plan_encoder_attentions: Optional[torch.FloatTensor] = None

class MultimodalTEAChModelforGamePlanClassification(BartPretrainedModel):
    def __init__(self,
                 config: BartConfig,
                 util_config: dict):
        super().__init__(config)

        self.util_config = util_config
        self.num_labels = config.num_labels

        # `mmt_model`, `multimodalteach_model`
        self.mmt_model = MultimodalTEAChModel(config=config,
                                              util_config=util_config)

        # config.classifier_dropout = `0.0`
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)
        self.output_attention_layer = nn.MultiheadAttention(embed_dim=config.d_model,
                                                     num_heads=4)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        game_plan_input_ids: Optional[torch.Tensor] = None,     # New addition of game_plan_input_ids
        game_plan_attention_mask: Optional[torch.Tensor] = None,     # New addition of game_plan_attention_mask
        visual_input: Optional[torch.Tensor] = None,     # New addition of visual_input
        head_mask: Optional[torch.Tensor] = None,
        # inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            labels[labels == self.util_config["TARGET_GAME_PLAN_PADDING_IDX"]] = -100

        outputs = self.mmt_model(
            input_ids,
            attention_mask=attention_mask,
            game_plan_input_ids=game_plan_input_ids,
            game_plan_attention_mask=game_plan_attention_mask,
            visual_input=visual_input,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # apply attention_layer for output-context
        sequence_output, _ = self.output_attention_layer(query=sequence_output,
                                                         key=sequence_output,
                                                         value=sequence_output)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultimodalTokenClassifierOutput(
            loss=loss,
            logits=logits,
            dialog_encoder_hidden_states=outputs.dialog_encoder_hidden_states,
            game_plan_encoder_hidden_states=outputs.game_plan_encoder_hidden_states,
            dialog_encoder_attentions=outputs.dialog_encoder_attentions,
            game_plan_encoder_attentions=outputs.game_plan_encoder_attentions,
        )