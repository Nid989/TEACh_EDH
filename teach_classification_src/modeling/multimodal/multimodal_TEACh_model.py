from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import (
    BartPretrainedModel,
    BartEncoder
) 
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformer_encoder import TransformerEncoder

@dataclass
class MultimodalSeq2SeqModelOutput(Seq2SeqModelOutput):
    pooler_output: Optional[torch.FloatTensor] = None
    dialog_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    game_plan_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    visual_hidden_states: Optional[torch.FloatTensor] = None
    dialog_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    game_plan_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    dialog_encoder_attentions: Optional[torch.FloatTensor] = None
    game_plan_encoder_attentions: Optional[torch.FloatTensor] = None

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class Multimodal_Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

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

        self.visual_transform = nn.Linear(util_config["SOURCE_VISUAL_DIM"], config.d_model)
        self.visual_dropout = nn.Dropout(util_config["VISUAL_DROPOUT"])
        # maybe add a layernorm after the projection of visual feats to d_model

        self.pos_enc_layer = PositionalEncoding(d_model=config.d_model,
                                               dropout=0.2)

        self.multimodal_encoder = TransformerEncoder(d_model=config.d_model,
                                                     n_layers=6,
                                                     n_heads=8,
                                                     d_ff=config.d_model)

        self.multimodal_pooler = Multimodal_Pooler(config=config)

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

        # project the visual to same dimensionality as config.d_model
        visual_input = self.visual_dropout(self.visual_transform(visual_input))

        # concatenate and apply positional/temporal encodings
        hidden_states = torch.cat((dialog_encoder_outputs.last_hidden_state,
                                   visual_input,
                                   game_plan_encoder_outputs.last_hidden_state), dim=1)
        hidden_states = self.pos_enc_layer(hidden_states)

        # apply contextualization via a TransformerEncoder call
        hidden_states = self.multimodal_encoder(hidden_states, mask=None)

        pooled_output = self.multimodal_pooler(hidden_states)
        # =============================================================================== #

        if not return_dict:
            return (dialog_encoder_outputs + game_plan_encoder_outputs + hidden_states)

        return MultimodalSeq2SeqModelOutput(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            dialog_encoder_last_hidden_state=dialog_encoder_outputs.last_hidden_state,
            game_plan_encoder_last_hidden_state=game_plan_encoder_outputs.last_hidden_state,
            visual_hidden_states=visual_input,
            dialog_encoder_hidden_states=dialog_encoder_outputs.hidden_states,
            game_plan_encoder_hidden_states=game_plan_encoder_outputs.hidden_states,
            dialog_encoder_attentions=dialog_encoder_outputs.attentions,
            game_plan_encoder_attentions=game_plan_encoder_outputs.attentions,
        )