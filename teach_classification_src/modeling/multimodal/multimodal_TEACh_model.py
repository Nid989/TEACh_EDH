import gc
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import (
    Seq2SeqModelOutput
)
from transformers.models.bart.modeling_bart import (
    BartPretrainedModel,
    BartEncoder
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformer_encoder import TransformerEncoder

from teach_classification_src.modeling.modality_aware_fusion import MAF

@dataclass
class MultimodalSeq2SeqModelOutput(Seq2SeqModelOutput):
    dialog_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    game_plan_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    visual_hidden_states: Optional[torch.FloatTensor] = None
    dialog_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    game_plan_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    dialog_encoder_attentions: Optional[torch.FloatTensor] = None
    game_plan_encoder_attentions: Optional[torch.FloatTensor] = None
    
class MultimodalTEAChModel(BartPretrainedModel):
    def __init__(self, config: BartConfig,
                 util_config: dict,
                 freeze_encoders: bool=True):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # ================================ Modifications ================================ #
        # We convert the unique action indices to a learned representation which will be further
        # utilized alongwith visual and text modality features to retrieve final representation.
        self.util_config = util_config

        self.dialog_encoder = BartEncoder(config, self.shared)
        self.game_plan_encoder = BartEncoder(config, self.shared)

        # NOTE: Literally we can omit this!
        self.visual_transformer = TransformerEncoder(d_model=self.util_config["SOURCE_VISUAL_DIM"],
                                                     n_layers=4,
                                                     n_heads=8,
                                                     d_ff=self.util_config["SOURCE_VISUAL_DIM"])
        self.MAF_layer = MAF(util_config=self.util_config,
                             dim_model=self.config.d_model, # embed_dim
                             dropout_rate=0.2)

        if freeze_encoders:
            # Freeze the parameters of dialog_encoder
            for param in self.dialog_encoder.parameters():
                param.requires_grad = False

            # Freeze the parameters of game_plan_encoder
            for param in self.game_plan_encoder.parameters():
                param.requires_grad = False
        # =============================================================================== #

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        game_plan_input_ids=None,      # New addition of game_plan_input_ids
        game_plan_attention_mask=None,      # New addition of game_plan_attention_mask
        visual_input=None,      # New addition of visual_input
        head_mask=None,
        encoder_outputs=None,
        # inputs_embeds=None,      # Omiting to avoid confusions
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ================================ Modifications ================================ #

        # generate encoder_outputs for `dialog_history`
        dialog_encoder_outputs = self.dialog_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # generate encoder_outputs for `game_plan`
        game_plan_encoder_outputs = self.game_plan_encoder(
            input_ids=game_plan_input_ids,
            attention_mask=game_plan_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        visual_input = self.visual_transformer(visual_input, mask=None)

        hidden_states = self.MAF_layer(text_input=dialog_encoder_outputs[0],
                                       game_plan_context=game_plan_encoder_outputs[0],
                                       visual_context=visual_input)

        # =============================================================================== #

        if not return_dict:
            return (dialog_encoder_outputs + game_plan_encoder_outputs + hidden_states)

        return MultimodalSeq2SeqModelOutput(
            last_hidden_state=hidden_states,
            dialog_encoder_last_hidden_state=dialog_encoder_outputs.last_hidden_state,
            game_plan_encoder_last_hidden_state=game_plan_encoder_outputs.last_hidden_state,
            visual_hidden_states=visual_input,
            dialog_encoder_hidden_states=dialog_encoder_outputs.hidden_states,
            game_plan_encoder_hidden_states=game_plan_encoder_outputs.hidden_states,
            dialog_encoder_attentions=dialog_encoder_outputs.attentions,
            game_plan_encoder_attentions=game_plan_encoder_outputs.attentions,
        )