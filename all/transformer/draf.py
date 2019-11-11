d_word_vec = 512
d_h=512
d_model = 512
d_inner = 2048
n_layers = 6
n_head = 8
d_k = 64
d_v = 64
dropout = 0.1
n_position, d_hid=n_position, d_word_vec
padding_idx=0
tgt_emb_prj_weight_sharing = True
emb_src_tgt_weight_sharing = True

n_position = len_max_seq + 1 #53


n_src_vocab, n_tgt_vocab, len_max_seq=opt.src_vocab_size,opt.tgt_vocab_size,opt.max_token_seq_len
# 2911 ,3149,  52

for batch in tqdm(
        training_data, mininterval=2,
        desc='  - (Training)   ', leave=False):
    batch=batch
    break

src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
seq_k, seq_q=src_seq,src_seq
seq=src_seq
get_non_pad_mask(src_seq)

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
    ...
    ...
    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []
        # src_seq.size()= torch.Size([64, 29])
        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        #torch.Size([64, 29, 29])
        non_pad_mask = get_non_pad_mask(src_seq)  #torch.Size([64, 29, 1])

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,
