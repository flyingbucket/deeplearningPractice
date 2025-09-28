import torch
from tqdm import tqdm


class SoftmaxClassifier:
    """
    这是一个基于softmax损失函数的线性分类器
    """

    def __init__(self, Xtr, Ytr):
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.feature_dim = Xtr.shape[1]
        self.num_classes = len(torch.unique(Ytr))
        self.W = torch.randn(self.feature_dim, self.num_classes, requires_grad=True)
        self.b = torch.zeros(1, self.num_classes, requires_grad=True)

    @staticmethod
    def softmax(X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition

    def loss(self, X, Y, reg):
        num_train = X.shape[0]
        scores = X.mm(self.W) + self.b
        probs = self.softmax(scores)
        correct_logprobs = -torch.log(probs[range(num_train), Y])
        data_loss = correct_logprobs.mean()
        reg_loss = 0.5 * reg * (self.W**2).sum()
        loss = data_loss + reg_loss
        return loss

    def train(self, learning_rate=0.01, reg=0.01, num_iters=1000, verbose=False):
        """
        训练模型
        """
        losses = []

        for i in tqdm(range(num_iters)):
            # 计算损失
            current_loss = self.loss(self.Xtr, self.Ytr, reg)
            losses.append(current_loss.item())

            # 计算梯度
            current_loss.backward()

            # 更新参数
            with torch.no_grad():
                self.W -= learning_rate * self.W.grad
                self.b -= learning_rate * self.b.grad

                # 清零梯度
                self.W.grad.zero_()
                self.b.grad.zero_()

            if verbose and i % 100 == 0:
                tqdm.write(f"Iteration {i}, Loss: {current_loss.item():.4f}")

        return losses

    def predict(self, X):
        """
        预测函数
        """
        # 计算分数
        scores = X.mm(self.W) + self.b
        # 计算概率
        probs = self.softmax(scores)
        # 返回预测类别（概率最大的类别）
        _, predicted_classes = torch.max(probs, 1)

        return predicted_classes

    def predict_proba(self, X):
        """
        预测概率
        """
        scores = X.mm(self.W) + self.b
        probs = self.softmax(scores)
        return probs

    def accuracy(self, X, Y):
        """
        计算准确率
        """
        predictions = self.predict(X)
        accuracy = (predictions == Y).float().mean()
        return accuracy.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    # 训练集
    X_train = torch.randn(100, 5)  # 100个样本，5个特征
    Y_train = torch.randint(0, 3, (100,))  # 3个类别

    # 测试集
    X_test = torch.randn(50, 5)  # 50个测试样本，5个特征
    Y_test = torch.randint(0, 3, (50,))  # 3个类别

    # 创建模型并训练
    model = SoftmaxRegression(X_train, Y_train)
    losses = model.train(learning_rate=0.1, reg=0.01, num_iters=1000, verbose=True)

    # 训练集准确率
    predictions_train = model.predict(X_train)
    accuracy_train = model.accuracy(X_train, Y_train)
    print(f"训练集准确率: {accuracy_train:.4f}")

    # 测试集准确率
    predictions_test = model.predict(X_test)
    accuracy_test = model.accuracy(X_test, Y_test)
    print(f"测试集准确率: {accuracy_test:.4f}")

    # probabilities_train = model.predict_proba(X_train)
    # print(f"前5个训练样本的概率分布:\n{probabilities_train[:5]}")

    # probabilities_test = model.predict_proba(X_test)
    # print(f"前5个测试样本的概率分布:\n{probabilities_test[:5]}")
