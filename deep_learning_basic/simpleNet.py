import torch.nn as nn
import torch

#最小可运行的二分类神经网络训练
class simpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(10,20) #全连接层不是输入维度=输出维度吗？它不是输入维度 = 输出维度，而是一个从输入维度 10 投影到输出维度 20 的线性变换。所以把全连接层理解为线性变换更合适，它只是神经元全连接而已。
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(20,1)
        self.sigmoid=nn.Sigmoid() #最后加一个激活函数，把输出变成 (batch_size, 1)而不是仅仅一个值，匹配labels

    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.sigmoid(x)

        return x
    

# 实例化模型和优化器
model = simpleNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# 模拟一批数据：输入是 (batch_size=4, features=10)，标签是 0 或 1
inputs = torch.randn(4, 10)
labels = torch.randint(0, 2, (4, 1)).float() #生成形状为 (4, 1) 的整数张量，值是 0 或 1（取值范围 [0, 2)）
#最后的.float()是为了匹配 BCELoss 要求（float 类型输入），int转成float

# 前向传播
outputs = model(inputs)

# 计算损失
loss = criterion(outputs, labels) #计算损失就是把模型的输出和数据的标签进行比对计算

# 反向传播 + 参数更新
optimizer.zero_grad() #梯度清零
loss.backward()
optimizer.step() #梯度更新参数

print("Loss:", loss.item())


    
        