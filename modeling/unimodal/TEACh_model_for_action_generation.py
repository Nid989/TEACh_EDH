import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    BartPretrainedModel,
    shift_tokens_right
)
from transformers.modeling_outputs import Seq2SeqLMOutput

from modeling.unimodal.action_only_TEACh_model import ActionOnlyTEAChModel

class TEAChModelForActionGeneration(BartPretrainedModel):

    def __init__(self, config: BartConfig, util_config: dict):
        super().__init__(config)

        self.config = config
        self.util_config = util_config
        self.model = ActionOnlyTEAChModel(config=config, util_config=util_config)
        self.lm_head = nn.Linear(util_config["ACTION_DIM"], util_config["ACTION_COUNT"], bias=False)

        # Initialize weights and apply final processing
        self.post_init()

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
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        Returns:
        """

        if labels is not None:
            if decoder_action_input is None:
                decoder_action_input = shift_tokens_right(
                    labels,
                    self.util_config["ACTION_PADDING_IDX"],
                    self.util_config["ACTION_START_IDX"]
                )

        if labels is not None:
            # replace indices with pad_tokens in `labels` and set to -100
            # since masked (-100) tokens are ignored while computing loss
            mask = (labels.clone() == self.util_config["ACTION_PADDING_IDX"])
            labels[mask] = -100
            # object_labels[mask] = -100
            labels.to(torch.long)
            # object_labels.to(torch.long)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            action_input=action_input,      # New addition of action_input
            # decoder_input_ids=decoder_input_ids,
            # decoder_attention_mask=decoder_attention_mask,
            decoder_action_input=decoder_action_input,      # New addition of decoder_action_input
            head_mask=head_mask,
            # decoder_head_mask=decoder_head_mask,
            # cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            # past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            # decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.util_config["ACTION_COUNT"]), labels.view(-1).to(torch.int64))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )