import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# 定义 LeNet-5 网络结构
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),  # 这里拉平是因为下面要进全连接层，所以要变成一维
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

# 打印各层输出形状，用于调试
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# 数据加载函数
def load_data_fashion_mnist(batch_size, root='./data'):
    """
    下载 Fashion-MNIST 数据集，并返回训练和测试的 DataLoader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True,
                                                    transform=transform, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False,
                                                   transform=transform, download=True)
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter

# 评估函数（使用 GPU 计算模型在数据集上的精度）
def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用 GPU 计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(net.parameters()).device
    # 正确预测的数量，总预测的数量
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            _, predicted = outputs.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total

# 训练函数（适配于 GPU 的训练过程）
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用 GPU 训练模型"""
    # 权重初始化
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # 用于记录训练过程中的 metrics
    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(num_epochs):
        net.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()  # 清零梯度
            y_hat = net(X)  # 前向传播
            l = loss_fn(y_hat, y)  # 计算损失
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数
            # 累加 loss 和 acc
            total_loss += l.item() * y.size(0)
            _, predicted = y_hat.max(1)
            total_acc += (predicted == y).sum().item()
            n += y.size(0)
        # 记录每个 epoch 的平均 loss 和训练准确率
        train_losses.append(total_loss / n)
        train_accs.append(total_acc / n)
        # 在测试集上评估准确率
        test_acc = evaluate_accuracy_gpu(net, test_iter, device)
        test_accs.append(test_acc)
        print(f'epoch {epoch + 1}, loss {train_losses[-1]:.4f}, '
              f'train acc {train_accs[-1]:.3f}, test acc {test_acc:.3f}')

    # 绘制训练过程曲线
    epochs = list(range(1, num_epochs + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label='train loss')
    plt.plot(epochs, train_accs, label='train acc')
    plt.plot(epochs, test_accs, label='test acc')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

# 主函数：训练和评估 LeNet-5
if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    lr, num_epochs = 0.9, 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
