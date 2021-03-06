import numpy as np
import json
#506条数据，13个x，预测一个y，506*14=7084
def load_data():
    path = './housing.data'
    data = np.fromfile(path, sep=' ')
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
        'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]

    feature_num = len(feature_names)    #14
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    print(data.shape)
    #数据拆分成训练集和测试集    
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    #14个最大值，14个最小值，14个平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), training_data.sum(axis=0) / training_data.shape[0]
    for i in range(feature_num):       #训练集和测试集都归一化（训练集的最大值、最小值、平均值归一化训练集和测试集）
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    # 归一化后，训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

    
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
    #前向传播  z=wx+b（这里没有激活函数）
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    #损失函数（MSE均值平方差）
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]    #404个预测结果，求平均值
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    #梯度下降
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = 2*(z-y)*x            #old：gradient_w = (z-y)*x ，求导少了一个2倍
        gradient_w = np.mean(gradient_w, axis=0)    
        gradient_w = gradient_w[:, np.newaxis]     #等价于reshape(num,1)
        gradient_b = 2*(z - y)            #也添加了一个2倍
        gradient_b = np.mean(gradient_b)        
        return gradient_w, gradient_b
    #更新
    def update(self, gradient_w, gradient_b, eta = 0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
    #训练   
    def train(self, x, y, iterations=100, eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            if (i+1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses
    #测试
    def test(self,x,y):
        z=np.dot(x, self.w) + self.b
        print(z,y)
if __name__=='__main__':
    training_data, test_data = load_data()
    x = training_data[:, :-1]
    y = training_data[:, -1:]
    # 创建网络
    net = Network(13)
    num_iterations=10000
    # 启动训练
    losses = net.train(x,y, iterations=num_iterations, eta=0.01)
    net.test(x,y)
    
    