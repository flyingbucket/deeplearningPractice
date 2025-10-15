import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class RNN(nn.Module):
    def __init__(
        self, vocab_size, hidden_size: int, seq_len: int, n_hidden=3, *args, **kwargs
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.Wxh = nn.Linear(self.vocab_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)
        Whh_list = [nn.Linear(hidden_size, hidden_size) for _ in range(n_hidden)]
        self.WhhSeq = nn.Sequential(*Whh_list)
        self.Why = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, inputs_idxs, h, seed_idx=None, teacher_forcing=True):
        """
        inputs_idxs: iterable of ints or 1D LongTensor, length T (输入序列索引)
        h: tensor shape (1, hidden_size)
        seed_idx: optional int to use as initial x if inputs not provided
        teacher_forcing: if True, at step t use inputs_idxs[t] as x; else use model output
        Returns:
            logits: Tensor shape (T, self.vocab_size)  -- 未softmax的logits
            h: final hidden state (1, hidden_size)
        """
        device = h.device
        # ensure inputs are a tensor on device
        if not torch.is_tensor(inputs_idxs):
            inputs = torch.tensor(list(inputs_idxs), dtype=torch.long, device=device)
        else:
            inputs = inputs_idxs.to(device)

        T = inputs.shape[0]
        logits = []

        # initial input x: use seed_idx if provided, else use inputs[0]
        if seed_idx is None:
            cur_idx = int(inputs[0].item())
        else:
            cur_idx = int(seed_idx)

        for t in range(T):
            # build one-hot row vector (1, self.vocab_size)
            x = torch.zeros(1, self.vocab_size, device=device)
            x[0, cur_idx] = 1.0

            # RNN update: use both Wxh and Whh
            h = torch.tanh(self.Wxh(x) + self.Whh(h))  # (1, hidden_size)
            h = torch.tanh(self.WhhSeq(h))
            y = self.Why(h)  # (1, self.vocab_size) logits
            logits.append(y.squeeze(0))  # collect shape (self.vocab_size,)

            # decide next input index
            if teacher_forcing:
                # next input is ground-truth inputs[t] (shifted in typical setups)
                # here we assume inputs[t] corresponds to current step's "next" token.
                # Advance cur_idx from inputs sequence if available.
                if t + 1 < T:
                    cur_idx = int(inputs[t + 1].item())
                else:
                    # last step: keep current or sample; here keep current
                    cur_idx = int(inputs[t].item())
            else:
                # use model's prediction (sample or argmax)
                p = F.softmax(y, dim=1)
                sampled = torch.multinomial(p, num_samples=1)  # (1,1)
                cur_idx = int(sampled.item())

        logits = torch.stack(logits, dim=0)  # (T, self.vocab_size)
        return logits, h

    def loss(self, inputs_idxs, targets_idxs, h, seed_idx=None):
        """
        inputs_idxs: length T input indices (teacher forcing)
        targets_idxs: LongTensor shape (T,) 目标索引
        h: initial hidden state
        """
        device = h.device
        logits, _ = self.forward(
            inputs_idxs, h, seed_idx=seed_idx, teacher_forcing=True
        )
        # logits: (T, self.vocab_size)
        # targets_idxs: ensure tensor on device and dtype long
        if not torch.is_tensor(targets_idxs):
            targets = torch.tensor(list(targets_idxs), dtype=torch.long, device=device)
        else:
            targets = targets_idxs.to(device).long()

        criterion = nn.CrossEntropyLoss()
        # CrossEntropyLoss expects (N, C) logits and (N,) targets
        loss = criterion(logits, targets)
        return loss

    def sample(self, seed_idx, h, length, use_gumbel=False, tau=1.0):
        """
        Generate a sequence of `length` tokens starting from seed_idx and h.
        Returns list of ints and final h.
        """
        device = h.device
        cur_idx = int(seed_idx)
        idxs = []
        for _ in range(length):
            x = torch.zeros(1, self.vocab_size, device=device)
            x[0, cur_idx] = 1.0
            h = torch.tanh(self.Wxh(x) + self.Whh(h))
            y = self.Why(h)  # (1, self.vocab_size)
            if use_gumbel:
                # gumbel_softmax returns a (1, self.vocab_size) vector (one-hot if hard=True)
                x_soft = F.gumbel_softmax(
                    y, tau=tau, hard=True
                )  # still differentiable approx
                # get index
                cur_idx = int(x_soft.argmax(dim=1).item())
            else:
                p = F.softmax(y, dim=1)
                cur_idx = int(torch.multinomial(p, num_samples=1).item())

            idxs.append(cur_idx)
        return idxs, h


class RNNBuiltIn(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,  # RNN 默认 (seq, batch, feat)
        )
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def _one_hot(self, idxs: torch.Tensor) -> torch.Tensor:
        """
        idxs: (B, T)
        return: onehot (B, T, V)
        """
        B, T = idxs.shape
        onehot = torch.zeros(B, T, self.vocab_size, device=idxs.device)
        onehot.scatter_(2, idxs.unsqueeze(-1), 1.0)
        return onehot

    def step(self, x_t, h):
        # x_t: (1, 1, input_size)  或  (1,) 索引配合 Embedding
        y, h = self.rnn(x_t, h)  # y: (1,1,H)
        logits = self.decoder(y)
        return logits, h

    def forward(self, inputs_idxs, h0=None, teacher_forcing=True):
        """
        inputs_idxs: (B, T) 的 LongTensor（输入序列）
        h0: (num_layers, batch_size, hidden_size)，默认全零
        返回:
          logits: (T, vocab_size)
          hT: (num_layers, 1, hidden_size)
        """
        if not torch.is_tensor(inputs_idxs):
            inputs_idxs = torch.tensor(
                list(inputs_idxs),
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        else:
            inputs_idxs = inputs_idxs.to(next(self.parameters()).device).long()

        # one-hot 序列作为 RNN 输入，形状 (B,T, vocab_size)
        x = self._one_hot(inputs_idxs)

        # 直接一次性喂给 RNN（不需要手写时间步循环）
        logits, hT = self.step(x, h0)

        return logits, hT

    def loss(self, inputs_idxs, targets_idxs, h0=None):
        logits, _ = self.forward(inputs_idxs, h0, teacher_forcing=True)
        logits = logits.reshape(-1, logits.size(-1))
        if not torch.is_tensor(targets_idxs):
            targets = torch.tensor(
                list(targets_idxs), dtype=torch.long, device=logits.device
            )
        else:
            targets = targets_idxs.to(logits.device).long()
        targets = targets.reshape(-1)
        # CrossEntropyLoss: (N,C) vs (N,)
        return F.cross_entropy(logits, targets)

    # @torch.no_grad()
    # def sample(self, seed_idx, h0=None, length=50):
    #     device = next(self.parameters()).device
    #     if h0 is None:
    #         h = torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
    #     else:
    #         h = h0.to(device)
    #     cur = torch.tensor([seed_idx], device=device, dtype=torch.long)
    #     out_idxs = []
    #     for _ in range(length):
    #         x = torch.zeros(1, 1, self.vocab_size, device=device)
    #         x[0, 0, cur.item()] = 1.0
    #         logits, h = self.step(x, h)
    #         probs = F.softmax(logits, dim=1)
    #         cur = torch.multinomial(probs, 1).squeeze(1)  # (1,)
    #         out_idxs.append(int(cur.item()))
    #     return out_idxs, h

    @torch.no_grad()
    def sample(self, seed_idx, h0=None, length=50, temperature: float = 1.0):
        """
        seed_idx: int 或 LongTensor(B,)
        返回:
        - 若 seed_idx 是 int：list[int]（长度=length）, h
        - 若 seed_idx 是 (B,)：LongTensor(B, length), h
        """
        device = next(self.parameters()).device
        V = self.vocab_size
        bf = getattr(self.rnn, "batch_first", False)

        # 规范化 batch 维
        if isinstance(seed_idx, int):
            cur = torch.tensor([seed_idx], device=device, dtype=torch.long)  # (B=1,)
            single = True
        else:
            cur = seed_idx.to(device).long().view(-1)  # (B,)
            single = False
        B = cur.size(0)

        # 初始化隐藏态
        if h0 is None:
            h = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
        else:
            h = h0.to(device)

        # 预分配输出
        out_idxs = torch.empty(B, length, dtype=torch.long, device=device)

        for t in range(length):
            # 构造一步输入（one-hot），配合 batch_first 自适应
            if bf:
                x = torch.zeros(B, 1, V, device=device)
                x.scatter_(2, cur.view(B, 1, 1), 1.0)  # (B,1,V)
            else:
                x = torch.zeros(1, B, V, device=device)
                x.scatter_(2, cur.view(1, B, 1), 1.0)  # (1,B,V)

            # RNN 一步 + 线性层
            y, h = self.rnn(x, h)  # (B,1,H) 或 (1,B,H)
            logits = self.decoder(y)  # (B,1,V) 或 (1,B,V)

            # 统一规整到 (B, V)：压掉时间维，并展平批次
            if bf:
                logits = logits.squeeze(1)  # (B,V)
            else:
                logits = logits.squeeze(0)  # (B,V)

            # 温度 & 抽样（保证 2D）
            probs = F.softmax(logits / max(1e-6, float(temperature)), dim=-1)  # (B,V)
            cur = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)

            out_idxs[:, t] = cur

        # 返回单条还是批量
        if single:
            return out_idxs[0].tolist(), h
        else:
            return out_idxs, h


def data2tensors(data, seq_len, device):
    chars = list(set(data))

    data_size, vocab_size = len(data), len(chars)
    print("data has %d characters, %d unique." % (data_size, vocab_size))

    char2idx = {ch: idx for idx, ch in enumerate(chars)}
    idx2char = {idx: ch for idx, ch in enumerate(chars)}
    data_idxs = [char2idx[ch] for ch in data]

    X = []
    Y = []
    for i in range(0, len(data), seq_len):
        inputs = data_idxs[i : i + seq_len]
        targets = data_idxs[i + 1 : i + seq_len + 1]  # 预测下一个字符
        if len(inputs) == len(targets) == seq_len:
            X.append(inputs)
            Y.append(targets)
    X = torch.tensor(X, dtype=torch.long)  # (N, T)
    Y = torch.tensor(Y, dtype=torch.long)  # (N, T)
    return X, Y, char2idx, idx2char


def train_char_rnn(
    model,
    train_loader,
    device,
    num_epochs=20,
    lr=5e-3,
    clip_grad=1.0,
    optimizer=None,
    scheduler=None,
    sample_every=5,
    sample_len=120,
    idx2char=None,
    seed_idx=0,
):
    model.to(device)
    model.train()
    if not optimizer:
        optimizer = (
            optim.SGD(model.parameters(), lr=lr)
            if scheduler is None
            else scheduler.optimizer
        )

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        total_tokens = 0

        for Xb, Yb in train_loader:  # Xb,Yb: (B, T)
            Xb = Xb.to(device)
            Yb = Yb.to(device)

            optimizer.zero_grad()
            loss = model.loss(Xb, Yb, h0=None)
            loss.backward()
            clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            total_loss += loss.item()
            total_tokens += Xb.shape[0] * Xb.shape[1]

        # 每个 epoch 结束的统计
        # avg_loss = total_loss / max(1, total_tokens)
        # ppl = (
        #     torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 20 else float("inf")
        # )

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch:02d} | total_loss: {total_loss:.4f}")

        # 简单采样看看效果
        if (idx2char is not None) and (epoch % sample_every == 0):
            model.eval()
            with torch.no_grad():
                out_idxs, _ = model.sample(seed_idx=seed_idx, length=sample_len)
                txt = "".join(idx2char[i] for i in out_idxs)
            model.train()
            print(f"[sample@epoch {epoch}] {txt[:150].replace('\\n', ' ')} ...")


if __name__ == "__main__":
    # 超参数
    hidden_size = 512
    seq_len = 256  # 每次训练用多少个字符的序列
    lr = 5e-3
    batch_size = 1024
    num_epochs = 200  # 训练轮数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    data = open("gcc.txt").read()
    X, Y, char2idx, idx2char = data2tensors(data, seq_len, device)
    vocab_size = len(char2idx)
    # model
    model = RNNBuiltIn(vocab_size, hidden_size)
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.95)

    # dataloader
    train_loader = DataLoader(
        TensorDataset(X, Y), batch_size=batch_size, shuffle=True, num_workers=0
    )
    # 训练
    train_char_rnn(
        model=model,
        train_loader=train_loader,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
        clip_grad=1.0,
        optimizer=optimizer,
        scheduler=scheduler,
        sample_every=20,
        sample_len=200,
        idx2char=idx2char,
        seed_idx=0,  # 可改成 char2idx['\n'] 或某起始字符
    )
    # # 初始化模型
    # model = RNN(vocab_size, hidden_size, seq_len).to(device)
    #
    # # 初始隐藏状态
    # h = torch.zeros(1, hidden_size, device=device)
    #
    # # 将数据转成索引序列
    # data_idxs = [char2idx[ch] for ch in data]
    #
    #
    # # dataloader
    # train_loader = DataLoader(TensorDataset())
    # # 训练循环
    # pbar = range(num_epochs)
    # for epoch in pbar:
    #     total_loss = 0
    #     h = h.detach()  # detach 防止梯度在整个序列上无限累积
    #
    #     # 内层进度条：显示当前 step 的 loss 和 lr
    #     pbar_in = tqdm(
    #         range(0, len(data_idxs) - seq_len, seq_len),
    #         desc=f"Epoch {epoch + 1}",
    #         leave=False,
    #     )
    #     for step, i in enumerate(pbar_in):
    #         inputs = data_idxs[i : i + seq_len]
    #         targets = data_idxs[i + 1 : i + seq_len + 1]  # 预测下一个字符
    #
    #         optimizer.zero_grad()
    #         loss = model.loss(inputs, targets, h)
    #         loss.backward()
    #         optimizer.step()
    #
    #         total_loss += loss.item()
    #
    #         # 更新内层进度条后缀
    #         pbar_in.set_postfix(
    #             {
    #                 "loss": f"{loss.item():.4f}",
    #                 "lr": f"{scheduler.get_last_lr()[0]:.5f}",
    #             },
    #             refresh=True,
    #         )
    #         if (step + 1) % 100 == 0:
    #             scheduler.step()
    #     # 计算平均 loss
    #     avg_loss = total_loss / ((len(data_idxs) - 1) // seq_len)
    #
    #     # 每隔一个 epoch 输出信息
    #     if (epoch + 1) % 1 == 0:
    #         tqdm.write(
    #             f"Epoch {epoch + 1} completed, avg_loss: {avg_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.5f}"
    #         )
    #     if (epoch + 1) % 5 == 0:
    #         # 推理示例
    #         seed_idx = data_idxs[0]  # 用文本开头作为种子
    #         sampled_idxs, _ = model.sample(seed_idx, h, length=100)
    #         sampled_text = "".join(idx2char[idx] for idx in sampled_idxs)
    #         print(f"Sampled text:\n{sampled_text}\n{'-' * 50}")
    #
    #         # 保存模型
    #         torch.save(model.state_dict(), "vanilla_rnn.pth")
