import torch.nn as nn
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartPretrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqModelOutput
)

from encoder_decoder_src.modeling.unimodal.bart_action_encoder import BartActionEncoder
from encoder_decoder_src.bart_action_decoder import BartActionDecoder

class ActionOnlyTEAChModel(BartPretrainedModel):
    def __init__(self, config: BartConfig, util_config: dict):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # ================================ Modifications ================================ #
        # We convert the unique action indices to a learned representation which will be further
        # utilized alongwith visual and text modality features to retrieve final representation.
        self.shared_action_embed = nn.Embedding(util_config["ACTION_COUNT"], util_config["ACTION_DIM"], util_config["ACTION_PADDING_IDX"])
        self.encoder = BartActionEncoder(config, util_config, self.shared, self.shared_action_embed)
        self.decoder = BartActionDecoder(util_config, self.shared_action_embed)
        # =============================================================================== #

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        action_input=None,      # New addition of action_input
        # decoder_input_ids=None,
        # decoder_attention_mask=None,
        decoder_action_input=None,      # New addition of decoder_action_input
        head_mask=None,
        # decoder_head_mask=None,
        # cross_attn_head_mask=None,
        encoder_outputs=None,
        # past_key_values=None,
        inputs_embeds=None,
        # decoder_inputs_embeds=None,
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

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                action_input=action_input,      # New addition of action_input
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # ================================ Modifications ================================ #
        decoder_outputs = self.decoder(
            decoder_action_input=decoder_action_input,
            encoder_hidden_states=encoder_outputs[0]
        )
        # =============================================================================== #

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )