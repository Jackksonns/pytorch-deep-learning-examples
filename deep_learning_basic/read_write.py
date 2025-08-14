import torch
from torch import nn
from torch.nn import functional as F

#加载和保存张量的方法——torch.save()和torch.load()
x = torch.arange(4)
torch.save(x, 'x-file')   #也就是把什么东西存在哪里

x2 = torch.load('x-file')
x2

y = torch.zeros(4)
torch.save([x, y],'x-files') #可以存列表
x2, y2 = torch.load('x-files')
(x2, y2)

mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict') #也可以存字典
mydict2 = torch.load('mydict')
mydict2

#加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20)) #随机生成参数，然后放入网络中
#注意：torch.randn是生成符合标准正态分布（均值为 0，标准差为 1）的随机数。
#而这里的2指的是我随机生成的样本数量，也就是batch_size，填多少整数都可以。但是后面的参数20就要满足对应网络的输入的维度要求了。
Y = net(X)

#将模型参数储存在mlp.params文件中
torch.save(net.state_dict(), 'mlp.params')

#原始模型的备份
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()

Y_clone = clone(X)
Y_clone == Y



