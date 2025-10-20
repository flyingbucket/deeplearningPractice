# mini_mt_transformer.py
# A minimal Encoder-Decoder Transformer for toy MT: "i love you" -> "我 爱 你"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = torch.device("cpu")

SRC_TOKENS = ["<pad>", "<bos>", "<eos>", "this", "is", "a", "simple", "message"]
TGT_TOKENS = ["<pad>", "<bos>", "<eos>", "这", "是", "一", "条", "信息"]

src_stoi = {w: i for i, w in enumerate(SRC_TOKENS)}
tgt_stoi = {w: i for i, w in enumerate(TGT_TOKENS)}
src_itos = {i: w for w, i in src_stoi.items()}
tgt_itos = {i: w for w, i in tgt_stoi.items()}

PAD_S, BOS_S, EOS_S = src_stoi["<pad>"], src_stoi["<bos>"], src_stoi["<eos>"]
PAD_T, BOS_T, EOS_T = tgt_stoi["<pad>"], tgt_stoi["<bos>"], tgt_stoi["<eos>"]


def encode_src(s: str):
    # "i love you" -> [BOS i love you EOS]
    toks = s.strip().split()
    return torch.tensor(
        [BOS_S] + [src_stoi[t] for t in toks] + [EOS_S], dtype=torch.long
    )


def encode_tgt(s: str):
    toks = s.strip().split()
    return torch.tensor(
        [BOS_T] + [tgt_stoi[t] for t in toks] + [EOS_T], dtype=torch.long
    )


def decode_tgt(ids):
    # drop BOS until EOS
    out = []
    for i in ids:
        if i == BOS_T:
            continue
        if i == EOS_T:
            break
        out.append(tgt_itos[i])
    return " ".join(out)


src_sentence = "this is a simple message"
tgt_sentence = "这 是 一 条 信息"

src = encode_src(src_sentence)  # shape [S]
tgt = encode_tgt(tgt_sentence)  # shape [T]
# Teacher forcing targets：预测 t=1..T-1 的下一个 token
tgt_inp = tgt[:-1]  # [BOS 我 爱 你]
tgt_out = tgt[1:]  # [我 爱 你 EOS]

# Add batch dim (N=1)
src = src.unsqueeze(1)  # [S, N]
tgt_inp = tgt_inp.unsqueeze(1)  # [T_in, N]
tgt_out = tgt_out.unsqueeze(1)  # [T_out, N]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.cos(pos * div)
        pe[:, 1::2] = torch.sin(pos * div)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):  # x: [L, N, E]
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


d_model = 64
nhead = 4
ffn_dim = 128
num_layers = 2


class MiniMT(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_emb = nn.Embedding(len(SRC_TOKENS), d_model)
        self.tgt_emb = nn.Embedding(len(TGT_TOKENS), d_model)
        self.posenc = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            batch_first=False,  # PyTorch default expects [L, N, E]
        )
        self.generator = nn.Linear(d_model, len(TGT_TOKENS))

    def make_src_key_padding_mask(self, src_ids):
        # toy: no padding here, return None or all False
        return None

    def make_tgt_key_padding_mask(self, tgt_ids):
        # toy: no padding here, return None or all False
        return None

    def make_subsequent_mask(self, size: int):
        # causal mask for decoder: [T,T], True = mask
        return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)

    def forward(self, src_ids, tgt_ids_in):
        """
        src_ids: [S, N]
        tgt_ids_in: [T, N] (teacher forcing inputs)
        """
        S, N = src_ids.size()
        T, _ = tgt_ids_in.size()

        src = self.posenc(self.src_emb(src_ids) * math.sqrt(d_model))  # [S,N,E]
        tgt = self.posenc(self.tgt_emb(tgt_ids_in) * math.sqrt(d_model))  # [T,N,E]

        tgt_mask = self.make_subsequent_mask(T).to(src_ids.device)  # [T,T]
        # key padding masks (None for this toy)
        src_pad_mask = self.make_src_key_padding_mask(src_ids)  # [N,S] if used
        tgt_pad_mask = self.make_tgt_key_padding_mask(tgt_ids_in)  # [N,T] if used

        memory = self.transformer.encoder(
            src, src_key_padding_mask=src_pad_mask
        )  # [S,N,E]
        out = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )  # [T,N,E]
        logits = self.generator(out)  # [T,N,V_tgt]
        return logits


model = MiniMT().to(device)
optim = torch.optim.Adam(model.parameters(), lr=5e-3)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_T)

model.train()
steps = 600
for step in range(1, steps + 1):
    optim.zero_grad()
    logits = model(src, tgt_inp)  # [T,N,V]
    loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
    loss.backward()
    optim.step()
    if step % 100 == 0:
        print(f"step {step:04d} | loss = {loss.item():.4f}")


# -------------------------
# 5) Greedy decode
# -------------------------
@torch.no_grad()
def greedy_decode(model: MiniMT, src_ids, max_len=10):
    model.eval()
    # Encode source
    memory = model.transformer.encoder(
        model.posenc(model.src_emb(src_ids) * math.sqrt(d_model))
    )  # [S,N,E]

    # start from <bos>
    ys = torch.tensor([[BOS_T]], dtype=torch.long, device=src_ids.device)  # [1,1]
    for _ in range(max_len):
        tgt = model.posenc(model.tgt_emb(ys) * math.sqrt(d_model))  # [t,N,E]
        tgt_mask = model.make_subsequent_mask(ys.size(0)).to(src_ids.device)
        out = model.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)  # [t,N,E]
        logits = model.generator(out[-1])  # [N,V]
        next_id = torch.argmax(logits, dim=-1)  # [N]
        ys = torch.cat([ys, next_id.unsqueeze(0)], dim=0)  # append
        if next_id.item() == EOS_T:
            break
    return ys.squeeze(1)  # [t']


# Decode result
pred_ids = greedy_decode(model, src)
print("SRC :", src_sentence)
print("TGT :", tgt_sentence)
print("PRED:", decode_tgt(pred_ids.tolist()))
