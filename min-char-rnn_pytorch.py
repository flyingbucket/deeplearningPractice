import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = open("input.txt").read()
chars = list(set(data))

data_size, vocab_size = len(data), len(chars)
print("data has %d characters, %d unique." % (data_size, vocab_size))

char2idx = {ch: idx for idx, ch in enumerate(data)}
idx2char = {idx: ch for idx, ch in enumerate(data)}


class RNN(nn.Module):
    def __init__(
        self, vocab_size, hidden_size: int, seq_len: int, *args, **kwargs
    ) -> None:
        super().__init__()
        self.Wxh = nn.Linear(vocab_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)
        self.Why = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs_idxs, h, seed_idx=None, teacher_forcing=True):
        """
        inputs_idxs: iterable of ints or 1D LongTensor, length T (输入序列索引)
        h: tensor shape (1, hidden_size)
        seed_idx: optional int to use as initial x if inputs not provided
        teacher_forcing: if True, at step t use inputs_idxs[t] as x; else use model output
        Returns:
            logits: Tensor shape (T, vocab_size)  -- 未softmax的logits
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
            # build one-hot row vector (1, vocab_size)
            x = torch.zeros(1, vocab_size, device=device)
            x[0, cur_idx] = 1.0

            # RNN update: use both Wxh and Whh
            h = torch.tanh(self.Wxh(x) + self.Whh(h))  # (1, hidden_size)

            y = self.Why(h)  # (1, vocab_size) logits
            logits.append(y.squeeze(0))  # collect shape (vocab_size,)

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

        logits = torch.stack(logits, dim=0)  # (T, vocab_size)
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
        # logits: (T, vocab_size)
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
            x = torch.zeros(1, vocab_size, device=device)
            x[0, cur_idx] = 1.0
            h = torch.tanh(self.Wxh(x) + self.Whh(h))
            y = self.Why(h)  # (1, vocab_size)
            if use_gumbel:
                # gumbel_softmax returns a (1, vocab_size) vector (one-hot if hard=True)
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


if __name__ == "__main__":
    # 超参数
    hidden_size = 128
    seq_len = 25  # 每次训练用多少个字符的序列
    lr = 1e-1
    num_epochs = 500  # 训练轮数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = RNN(vocab_size, hidden_size, seq_len).to(device)

    # 初始隐藏状态
    h = torch.zeros(1, hidden_size, device=device)

    # 将数据转成索引序列
    data_idxs = [char2idx[ch] for ch in data]

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        h = h.detach()  # detach 防止梯度在整个序列上无限累积
        for i in range(0, len(data_idxs) - seq_len, seq_len):
            inputs = data_idxs[i : i + seq_len]
            targets = data_idxs[i + 1 : i + seq_len + 1]  # 预测下一个字符

            optimizer.zero_grad()
            loss = model.loss(inputs, targets, h)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, loss: {total_loss:.4f}")

    # 推理示例
    seed_idx = data_idxs[0]  # 用文本开头作为种子
    sampled_idxs, _ = model.sample(seed_idx, h, length=100)
    sampled_text = "".join(idx2char[idx] for idx in sampled_idxs)
    print(f"Sampled text:\n{sampled_text}\n{'-' * 50}")

    # 保存模型
    torch.save(model.state_dict(), "vanilla_rnn.pth")
