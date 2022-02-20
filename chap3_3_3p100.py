from torch import nn
import torch
import numpy as np
from torch.utils import data

def synthetic_data(w, b, num_example):
    x = torch.normal(0, 1, (num_example, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    # y = y.type(torch.FloatTensor)
    return x, y.reshape((-1, 1))

net = nn.Sequential(nn.Linear(3, 1))
# 定义⼀个模型变量net，它是⼀个Sequential类的实例
# 全连接层在Linear类中定义。
# 将两个参数传递到nn.Linear中, 第⼀个输⼊特征形状2，第⼆个指定输出特征形状，为单个标量,为1。

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# 使⽤net之前，初始化模型参数中的权重和偏置。深度学习框架通常有预定
# 指定每个权重参数应该从均值为0、标准差为0.01的正态分布，偏置参数将初始化为零

loss = nn.MSELoss()
# 定义损失函数,计算均⽅误差使⽤的是MSELoss类

trainer = torch.optim.SGD(net.parameters(), lr = 0.03)
# 定义优化算法为小批量随机梯度下降算法
# 实例化⼀个SGD实例时，要指定优化的参数（可通过net.parameters()从模型中获得）以及优化算法所需的超参数字典。设置lr值为0.03

def load_array(data_array, batch_size, is_train = True):
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)
# 构造一个数据迭代器
# 通过数据迭代器指定batch_size
# 布尔值is_train表⽰是否希望数据迭代器对象在每个迭代周期内打乱数据

true_w = torch.tensor([2, -3, 4], dtype = torch.float)
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
batch_size = 10
data_iter = load_array((features, labels), batch_size)
print("————————————————————features——————————————————")
print(features)
print("——————————————————————labels———————————————————")
print(labels)

# 训练
# 在每个迭代周期⾥，我们将完整遍历⼀次数据集（train_data），中获取⼀个小批量的输⼊和相应的标签
# 进行以下步骤：
"""
    • 通过调⽤net(X)⽣成预测并计算损失l（前向传播）。
    • 通过进⾏反向传播来计算梯度。
    • 通过调⽤优化器来更新模型参数。
    计算每个迭代周期后的损失，并打印它来监控训练过程
"""
num_epoch = 3
for epoch in range(num_epoch):
    for X, y in data_iter:
        print("——————————————————————————X————————————————————")
        print(X)
        print("——————————————————————————y—————————————————————")
        print(y)
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f"epoch{epoch + 1}, loss = {l :f}")

