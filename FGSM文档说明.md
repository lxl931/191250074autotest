#FGSM
##引入相关包
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn functional as F
等
##输入
设置不同的扰动大小
epsilions = [0,.05,.1,.15,.2,.25,.3]
预训练模型
pretrained_model="./data/lenet_mnist_model.pth"
是否使用uda
use_cude = True
##定义被攻击的模型
定义LENET模型
class Net(nn.Module):
声明MNIST测试数据集和数据加载
test_loader = torch.utils.data.DataLoader
##定义FGSM攻击函数
def fgsm_attack(image,epsilon,data_grad):

##测试函数
def test(model,device,test_loader,epsilon):
##可视化结果
plt.figure(figsize=(5.5))
...
##可视化对抗样本
在每个epsilon上绘制几个对抗样本的例子
plt.figure(figsize=(8,10))
...
##训练模型
定义LENET模型
class Net(nn.Mocule):
训练模型
def train(args,model,device,train_loader,optimizer,epoch):
测试
def test(model,device,test_loader):
main函数
def main():
if __name__ == '__main__':
main()
