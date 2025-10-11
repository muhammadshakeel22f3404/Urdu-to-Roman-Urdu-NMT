import torch
import torch.nn as nn

class EncoderBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, src, src_lens):
        # src: [B, T]
        emb = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        enc_out_packed, (h, c) = self.lstm(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out_packed, batch_first=True)
        # h, c: [num_layers*2, B, hidden_size]
        return enc_out, (h, c)

class LuongAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.W = nn.Linear(dec_dim, enc_dim, bias=False)

    def forward(self, dec_h, enc_out, mask):
        # dec_h: [B, dec_dim], enc_out: [B, T, enc_dim], mask: [B, T] (True where padding)
        wdh = self.W(dec_h).unsqueeze(2)              # [B, enc_dim, 1]
        scores = torch.bmm(enc_out, wdh).squeeze(2)   # [B, T]
        scores.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)          # [B, T]
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)  # [B, enc_dim]
        return ctx, attn

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers=4, dropout=0.3, pad_idx=0, enc_hidden_size=512, use_attention=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.use_attention = use_attention
        self.enc_dim = enc_hidden_size * 2
        self.lstm = nn.LSTM(
            input_size=emb_dim + (self.enc_dim if use_attention else 0),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        if use_attention:
            self.attn = LuongAttention(enc_dim=self.enc_dim, dec_dim=hidden_size)
        self.proj = nn.Linear(hidden_size + (self.enc_dim if use_attention else 0), vocab_size)

    def forward_step(self, y_prev, hidden, enc_out, enc_mask):
        # y_prev: [B]
        emb = self.dropout(self.embedding(y_prev).unsqueeze(1))  # [B,1,emb]
        if self.use_attention:
            dec_h = hidden[0][-1]  # top layer hidden [B, H]
            ctx, attn = self.attn(dec_h, enc_out, enc_mask)      # [B, enc_dim]
            inp = torch.cat([emb, ctx.unsqueeze(1)], dim=-1)     # [B,1,emb+enc_dim]
        else:
            inp = emb
            attn = None
        out, hidden = self.lstm(inp, hidden)  # out: [B,1,H]
        if self.use_attention:
            logits = self.proj(torch.cat([out.squeeze(1), ctx], dim=-1))  # [B,V]
        else:
            logits = self.proj(out.squeeze(1))  # [B,V]
        return logits, hidden, attn

    def forward(self, tgt, hidden, enc_out, enc_mask, teacher_forcing_ratio=0.5):
        # tgt: [B, T] with BOS...EOS
        B, T = tgt.size()
        device = tgt.device
        outputs = []
        y_t = tgt[:, 0]  # BOS
        for t in range(1, T):
            logits, hidden, attn = self.forward_step(y_t, hidden, enc_out, enc_mask)
            outputs.append(logits.unsqueeze(1))
            use_tf = torch.rand(1).item() < teacher_forcing_ratio
            y_t = tgt[:, t] if use_tf else torch.argmax(logits, dim=-1)
        outputs = torch.cat(outputs, dim=1)  # [B, T-1, V]
        return outputs

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim, enc_hidden, enc_layers, dec_hidden, dec_layers, dropout, pad_idx=0, use_attention=True):
        super().__init__()
        self.encoder = EncoderBiLSTM(src_vocab_size, emb_dim, enc_hidden, num_layers=enc_layers, dropout=dropout, pad_idx=pad_idx)
        self.bridge_h = nn.Linear(enc_hidden*2, dec_hidden)
        self.bridge_c = nn.Linear(enc_hidden*2, dec_hidden)
        self.decoder = DecoderLSTM(
            tgt_vocab_size, emb_dim, dec_hidden,
            num_layers=dec_layers, dropout=dropout, pad_idx=pad_idx,
            enc_hidden_size=enc_hidden, use_attention=use_attention
        )

    def init_decoder_state(self, h, c):
        """
        Convert encoder (BiLSTM) states to decoder initial states and ensure
        the number of layers matches the decoder's num_layers by truncating
        or repeating the top layer as needed.
        h,c from encoder: [enc_layers*2, B, H_enc]
        After concatenating directions: [enc_layers, B, 2*H_enc]
        After projection: [enc_layers, B, H_dec]
        Must return: [dec_layers, B, H_dec]
        """
        def _cat_directions(x):
            # [layers*2, B, H] -> [layers, B, 2H]
            return torch.cat([x[0::2], x[1::2]], dim=-1)

        h_ = _cat_directions(h)
        c_ = _cat_directions(c)
        h_ = torch.tanh(self.bridge_h(h_))  # [enc_layers, B, dec_H]
        c_ = torch.tanh(self.bridge_c(c_))  # [enc_layers, B, dec_H]

        dec_layers = self.decoder.lstm.num_layers
        enc_layers = h_.size(0)

        if enc_layers == dec_layers:
            return (h_, c_)
        elif enc_layers > dec_layers:
            # Take the last dec_layers (closer to top of encoder)
            return (h_[-dec_layers:], c_[-dec_layers:])
        else:
            # Repeat the last layer to match dec_layers
            need = dec_layers - enc_layers
            h_rep = torch.cat([h_, h_[-1:].repeat(need, 1, 1)], dim=0)
            c_rep = torch.cat([c_, c_[-1:].repeat(need, 1, 1)], dim=0)
            return (h_rep, c_rep)

    def forward(self, src, src_lens, tgt, src_pad_idx=0, teacher_forcing_ratio=0.5):
        enc_out, (h, c) = self.encoder(src, src_lens)
        dec_init = self.init_decoder_state(h, c)
        enc_mask = (src == src_pad_idx)  # True for padding
        out = self.decoder(tgt, dec_init, enc_out, enc_mask, teacher_forcing_ratio=teacher_forcing_ratio)
        return out