from typing import Optional
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

class BartActionDecoder(nn.Module):
    def __init__(self, config: dict, embed_action_tokens: Optional[nn.Embedding]=None):
        super(BartActionDecoder, self).__init__()

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_size = config["ACTION_DIM"]
        self.num_decoder_layers = config["NUM_DECODER_LAYERS"]
        self.dropout = nn.Dropout(config["DROPOUT"])
        self.padding_idx = config["ACTION_PADDING_IDX"]

        # FIXME batch_first seems to crash the system
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self.hidden_size, nhead=8, batch_first=True),
            self.num_decoder_layers
        )

        self.embed_action_tokens = nn.Embedding(config["ACTION_COUNT"], self.hidden_size, self.padding_idx)
        if embed_action_tokens is not None:
            self.embed_action_tokens.weight = embed_action_tokens.weight

    def forward(
        self,
        decoder_action_input=None,
        encoder_hidden_states=None,
    ):
        # embed actions
        action_embeds = self.embed_action_tokens(decoder_action_input.to(torch.int64))
        action_embeds = self.dropout(action_embeds)
        # padded indices do not contribute to the gradient
        padding_mask = (decoder_action_input == 0)

        hidden_states = action_embeds
        encoder_hidden_states = encoder_hidden_states
        # NOTE: ideally the TransformerDecoder works w/ input dimensionality of
        # sequence_length x batch_size x hidden_size but somehow the converse
        # is working instead.
        # hidden_states = fused_representation.permute(1, 0, 2)
        # encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)

        # decoder block
        hidden_states = self.transformer_decoder(
            tgt=hidden_states,
            memory=encoder_hidden_states,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None, # fix the permutation issue
            memory_key_padding_mask=None
        )

        return BaseModelOutput(
            last_hidden_state=hidden_states
        )