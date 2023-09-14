import torch
import torch.nn.functional as F

from encoder_decoder_src.modeling.multimodal.multimodal_TEACh_model_for_action_generation import MultimodalTEAChModelForActionGeneration
from encoder_decoder_src.modeling.unimodal.TEACh_model_for_action_generation import TEAChModelForActionGeneration

class Generation:

    @torch.no_grad()
    def generate_no_beam_search(
        self,
        model,
        encodings,
        action_input,
        decoder_action_input,
        cur_len,
        sos_token_id,
        eos_token_id,
        max_decoding_length,
        min_decoding_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        visual_input=None,
        decoder_visual_input=None,
    ):
        """
        Inspired from the Huggingface implementation available here under the generation_utils.py file.
        """
        while cur_len < max_decoding_length:

            # prepare inputs & shift them to `device`
            model_inputs = self.prepare_inputs_for_generation(encodings=encodings,
                                                              action_input=action_input,
                                                              visual_input=visual_input,
                                                              decoder_visual_input=decoder_visual_input,
                                                              decoder_action_input=decoder_action_input,
                                                              device="cuda",
                                                              sos_token_id=sos_token_id)
            
            outputs = self.generate_output(model, **model_inputs)
            next_token_logits = outputs.logits[:, -1, :]

            # post-processing generated logits for the final token
            if eos_token_id is not None and cur_len < min_decoding_length:
                next_token_logits[:, eos_token_id] = -float("inf")

            if do_sample:
                scores = next_token_logits
                # Temprature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                next_token_logscores = self.top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # add token and increase length by one
            decoder_action_input = model_inputs["decoder_action_input"].clone().detach()
            decoder_action_input = torch.cat([decoder_action_input, next_token.unsqueeze(dim=0)], dim=-1)

            # ================================ TODO ================================ #
            # Add code corresponding to object determination and simulator here.
            # ====================================================================== #

            cur_len = cur_len + 1

            # discontinue the loop, if reached eos_token_id
            if next_token == eos_token_id: break

        return decoder_action_input

    def top_k_top_p_filtering(
        self,
        token_logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        """
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        logits = token_logits.clone()
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def prepare_inputs_for_generation(
        self,
        encodings,
        action_input,
        visual_input=None,
        decoder_visual_input=None,
        decoder_action_input=None,
        device=None,
        sos_token_id=None,
    ):
        assert isinstance(encodings["input_ids"], torch.Tensor), "`input_ids` should be of dtype torch.Tensor."
        assert isinstance(encodings["attention_mask"], torch.Tensor), "`attention_mask` should be of dtype torch.Tensor."
        assert isinstance(action_input, torch.Tensor), "`action_input` should be of dtype torch.Tensor."

        if visual_input is not None:
            assert isinstance(visual_input, torch.Tensor), "`visual_input` should be of dtype torch.Tensor."
            assert action_input.shape[1] == visual_input.shape[1], "length of `action_input`, & `visual_input` should be strictly equal."

        if decoder_visual_input is not None:
            assert isinstance(decoder_visual_input, torch.Tensor), "`decoder_visual_input` should be of dtype torch.Tensor."
        if decoder_action_input is not None:
            assert isinstance(decoder_action_input, torch.Tensor), "`decoder_action_input` should be of dtype torch.Tensor."
        if decoder_visual_input is not None and decoder_action_input is not None:
            assert decoder_visual_input.shape[1] == decoder_action_input.shape[1], "length of `decoder_visual_input`, & `decoder_action_input` should be strictly equal."

        if visual_input is not None and decoder_visual_input is None:
            decoder_visual_input = visual_input[:, -1, :].unsqueeze(dim=1)
        if decoder_action_input is None:
            assert sos_token_id is not None, "provide the `sos_token_id` value, for further processing"
            decoder_action_input = torch.tensor([sos_token_id], dtype=torch.long).unsqueeze(dim=0)

        if device is None:
            device = "cpu"
        assert isinstance(device, torch.device) or (isinstance(device, str) and device in ["cuda", "cpu"]), "`device` is not defined properly, should be of dtype `torch.device` or str(cuda, cpu)"

        return {
            "input_ids": encodings["input_ids"].to(device),
            "attention_mask": encodings["attention_mask"].to(device),
            "action_input": action_input.to(device),
            "visual_input": visual_input.to(device) if visual_input is not None else None,
            "decoder_visual_input": decoder_visual_input.to(device) if decoder_visual_input is not None else None,
            "decoder_action_input": decoder_action_input.to(device).clone().detach(),
        }

    def generate_output(self, model, **args):

        """
            model: PyTorch based model capable of encoding the
                processing **args
            **args:
                `input_ids`: torch.tensor
                `attention_mask`: torch.tensor
                `action_input`: torch.tensor
                `visual_input`: torch.tensor # None if action_only
                `decoder_visual_input`: torch.tensor # None if action_only
                `decoder_action_input`: torch.tensor
        """

        required_args = ['input_ids', 'attention_mask', 'action_input', 'visual_input', 'decoder_visual_input', 'decoder_action_input']

        for arg_name in required_args:
            assert arg_name in args, f"Missing required argument: {arg_name}"

        # args = dict((t, args[t].to(model.device)) for t in args)

        model.eval()
        with torch.no_grad():
            if args["visual_input"] is None and args["decoder_visual_input"] is None:
                outputs = model(
                    input_ids=args["input_ids"],
                    attention_mask=args["attention_mask"],
                    action_input=args["action_input"],
                    decoder_action_input=args["decoder_action_input"],
                )
            else:
                outputs = model(
                    input_ids=args["input_ids"],
                    attention_mask=args["attention_mask"],
                    action_input=args["action_input"],
                    visual_input=args["visual_input"],
                    decoder_visual_input=args["decoder_visual_input"],
                    decoder_action_input=args["decoder_action_input"],
                )
        return outputs