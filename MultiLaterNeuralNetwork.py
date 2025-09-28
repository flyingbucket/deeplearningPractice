import torch
import torch.nn as nn
import numpy as numpy


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, numLayers=2) -> None:
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(numLayers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def classfierLoss(pred, label, model, reg=1e-3) -> float:
    correct_logpred = -torch.log(pred[range(pred), label])
    data_loss = correct_logpred.mean()
    reg_loss = torch.tensor(0.0, device=pred.device)
    for param in model.parameters():
        reg_loss += torch.sum(param**2)
    loss = data_loss + reg * reg_loss / pred.shape[0]
    return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    # 训练集
    X_train = torch.randn(100, 5)  # 100个样本，5个特征
    Y_train = torch.randint(0, 3, (100,))  # 3个类别

    # 测试集
    X_test = torch.randn(50, 5)  # 50个测试样本，5个特征
    Y_test = torch.randint(0, 3, (50,))  # 3个类别

    model = MLP(input_dim=5, hidden_dim=10, output_dim=1, numLayers=4)
    for _ in range(100):
        pred = torch.softmax(model(X_train), dim=1)
        loss = classfierLoss(pred, Y_train, model)
        loss.backwoard()
