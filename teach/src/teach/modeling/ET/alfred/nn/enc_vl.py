import torch
from alfred.nn.encodings import PosEncoding, PosLearnedEncoding, TokenLearnedEncoding
from alfred.utils import model_util
from torch import nn
        
    
class EncoderVL(nn.Module):
    def __init__(self, args):
        """
        transformer encoder for language, frames and action inputs
        """
        super(EncoderVL, self).__init__()

        # transofmer layers
        encoder_layer = nn.TransformerEncoderLayer(
            args.demb,
            args.encoder_heads,
            args.demb,
            args.dropout["transformer"]["encoder"],
        )
        self.enc_transformer = nn.TransformerEncoder(encoder_layer, args.encoder_layers)

        # how many last actions to attend to
        self.num_input_actions = args.num_input_actions

        # encodings
        self.enc_pos = PosEncoding(args.demb) if args.enc["pos"] else None
        self.enc_pos_learn = PosLearnedEncoding(args.demb) if args.enc["pos_learn"] else None
        self.enc_token = TokenLearnedEncoding(args.demb) if args.enc["token"] else None
        self.enc_layernorm = nn.LayerNorm(args.demb)
        self.enc_dropout = nn.Dropout(args.dropout["emb"], inplace=True)
        
        if args.use_cross_attn:
            # cross-attention 
            self.mm_cross_attn = MultimodalCrossAttn(args)
        
    def forward(
        self,
        emb_lang,
        emb_frames,
        emb_actions,
        lengths_lang,
        lengths_frames,
        lengths_actions,
        length_frames_max,
        attn_masks=True,
    ):
        """
        pass embedded inputs through embeddings and encode them using a transformer
        """
        
        # emb_lang is processed on each GPU separately so they size can vary
        # ========================= highlight ========================= #
        # computation of length_lang_max is incorrect thus last_hidden idx 
        # was being ignored.
        # length_lang_max = lengths_lang.max().item()
        length_lang_max = lengths_lang.max().item() + 1
        # ============================================================= #
        emb_lang = emb_lang[:, :length_lang_max]
        # create a mask for padded elements
        length_mask_pad = length_lang_max + length_frames_max * (2 if lengths_actions.max() > 0 else 1)
        mask_pad = torch.zeros((len(emb_lang), length_mask_pad), device=emb_lang.device).bool()
        for i, (len_l, len_f, len_a) in enumerate(zip(lengths_lang, lengths_frames, lengths_actions)):
            # mask padded words
            mask_pad[i, len_l:length_lang_max] = True
            # mask padded frames
            mask_pad[i, length_lang_max + len_f : length_lang_max + length_frames_max] = True
            # mask padded actions
            mask_pad[i, length_lang_max + length_frames_max + len_a :] = True

        if args.use_cross_attn:    
            # apply cross-attention over pairs of `emb_lang`, `emb_frames`, and `emb_actions`    
            emb_lang, emb_frames, emb_actions = self.mm_cross_attn(emb_lang, emb_frames, emb_actions,
                                                                   length_lang_max, length_frames_max, self.num_input_actions)
        
        # encode the inputs
        emb_all = self.encode_inputs(emb_lang, emb_frames, emb_actions, lengths_lang, lengths_frames, mask_pad)
        
        # create a mask for attention (prediction at t should not see frames at >= t+1)
        if attn_masks:
            mask_attn = model_util.generate_attention_mask(
                length_lang_max,
                length_frames_max,
                emb_all.device,
                self.num_input_actions,
            )
        else:
            # allow every token to attend to all others
            mask_attn = torch.zeros((mask_pad.shape[1], mask_pad.shape[1]), device=mask_pad.device).float()

        # encode the inputs
        output = self.enc_transformer(emb_all.transpose(0, 1), mask_attn, mask_pad).transpose(0, 1)
        return output, mask_pad

    def encode_inputs(self, emb_lang, emb_frames, emb_actions, lengths_lang, lengths_frames, mask_pad):
        """
        add encodings (positional, token and so on)
        """
        if self.enc_pos is not None:
            emb_lang, emb_frames, emb_actions = self.enc_pos(
                emb_lang, emb_frames, emb_actions, lengths_lang, lengths_frames
            )
        if self.enc_pos_learn is not None:
            emb_lang, emb_frames, emb_actions = self.enc_pos_learn(
                emb_lang, emb_frames, emb_actions, lengths_lang, lengths_frames
            )
        if self.enc_token is not None:
            emb_lang, emb_frames, emb_actions = self.enc_token(emb_lang, emb_frames, emb_actions)
        emb_cat = torch.cat((emb_lang, emb_frames, emb_actions), dim=1)
        emb_cat = self.enc_layernorm(emb_cat)
        emb_cat = self.enc_dropout(emb_cat)
        return emb_cat
    
    
class MultimodalCrossAttn(nn.Module):
    def __init__(self, args):
        super(MultimodalCrossAttn, self).__init__()
        """
        multi-modal cross-attention b/w language, visual and actions 
        representations.
        """
        print(args.demb, args.cross_attn_num_head)
        self.args = args
        self.device = args.device
        # self-attention layers
        self.mha_v_t = nn.MultiheadAttention(embed_dim=args.demb, 
                                             num_heads=args.cross_attn_num_head, 
                                             batch_first=True)
        self.mha_a_t = nn.MultiheadAttention(embed_dim=args.demb, 
                                             num_heads=args.cross_attn_num_head, 
                                             batch_first=True)
        self.mha_t_v = nn.MultiheadAttention(embed_dim=args.demb, 
                                             num_heads=args.cross_attn_num_head, 
                                             batch_first=True)
        self.mha_a_v = nn.MultiheadAttention(embed_dim=args.demb, 
                                             num_heads=args.cross_attn_num_head, 
                                             batch_first=True)
        self.mha_t_a = nn.MultiheadAttention(embed_dim=args.demb, 
                                             num_heads=args.cross_attn_num_head, 
                                             batch_first=True)
        self.mha_v_a = nn.MultiheadAttention(embed_dim=args.demb, 
                                             num_heads=args.cross_attn_num_head, 
                                             batch_first=True)
        
        self.final_lang_layer_norm = nn.LayerNorm(args.demb)
        self.final_visual_layer_norm = nn.LayerNorm(args.demb)
        self.final_action_layer_norm = nn.LayerNorm(args.demb)
        
    def forward(self,
                emb_lang,
                emb_frames,
                emb_actions,
                len_lang,
                len_frames,
                num_input_actions):
        
        # formulate attention-mask for modality-pairs
        lang_to_frames_attn = torch.zeros((len_lang, len_frames), device=self.device).float()
        lang_to_actions_attn = torch.zeros((len_lang, len_frames), device=self.device).float()
        frames_to_lang_attn = torch.zeros((len_frames, len_lang), device=self.device).float()
        # applying mask of `-inf` might result in propogation of `nan` gradient so replacing it w/ large negative value `-1e9`
        frames_to_actions_attn = torch.ones((len_frames, len_frames), device=self.device).float() * float("-1e9")
        # 2.3 then unmask `num_input_actions` previous actions for each frame (excluding index t)
        for a_idx in range(num_input_actions):
            for f_idx in range(len_frames):
                if f_idx - 1 - a_idx < 0:
                    # the index is out of bound
                    continue
                frames_to_actions_attn[f_idx, f_idx - 1 - a_idx] = 0.0

        actions_to_lang_attn = frames_to_lang_attn.clone()
        actions_to_frames_attn = frames_to_actions_attn.clone()
        
        # apply attention over pairs of modalities
        # V, A => T
        x_v2t, _ = self.mha_v_t(emb_lang, emb_frames, emb_frames)
        x_a2t, _ = self.mha_a_t(emb_lang, emb_frames, emb_frames)
        
        # T, A => V
        x_t2v, _ = self.mha_t_v(emb_frames, emb_lang, emb_lang)
        x_a2v, _ = self.mha_t_v(emb_frames, emb_actions, emb_actions)

        # T, V => A
        x_t2a, _ = self.mha_t_a(emb_actions, emb_lang, emb_lang)
        x_v2a, _ = self.mha_t_a(emb_actions, emb_frames, emb_frames)
        
        emb_lang_out = self.final_lang_layer_norm(x_v2t + x_a2t)
        emb_frames_out = self.final_visual_layer_norm(x_t2v + x_a2v)
        emb_actions_out = self.final_action_layer_norm(x_t2a + x_v2a)
        
        return (emb_lang_out, emb_frames_out, emb_actions_out)
