import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image

img_size = 32
EPOCH = 200
BATCH_SIZE = 64
learning_rate = 0.0001
seed = 1
test_every = 10
z_dim = 20
# input_dim = 28
input_channel = 1
resume = ''
cuda = True if torch.cuda.is_available() else False


def dataloader():
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    mnist_train = datasets.MNIST("./Data/mnist", train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST("./Data/mnist", train=False, transform=transform, download=True)
    train_data = DataLoader(dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=True)
    test_data = DataLoader(dataset=mnist_test, batch_size=BATCH_SIZE, shuffle=True)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    return test_data, train_data, classes


def loss_function(x_hat, x, mu, log_var):
    # 1. the reconstruction loss.
    # We regard the MNIST as binary classification
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    # 2. KL-divergence
    # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
    # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
    KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)
    # 3. total loss
    loss = BCE + KLD
    return loss, BCE, KLD


def save_checkpoint(state, is_best, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')  # join函数创建子文件夹，也就是把第二个参数对应的文件保存在'outdir'里
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file)  # 把state保存在checkpoint_file文件夹中
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)


def test(model, optimizer, test_data, epoch, best_test_loss):
    test_avg_loss = 0.0
    with torch.no_grad():
        for test_batch_index, (test_x, _) in enumerate(test_data):
            test_x = test_x.cuda()
            test_x_hat, test_mu, test_log_var = model(test_x)
            test_loss, test_BCE, test_KLD = loss_function(test_x_hat, test_x, test_mu, test_log_var)
            test_avg_loss += test_loss

        test_avg_loss /= len(test_data.dataset)
        z = torch.randn(BATCH_SIZE, z_dim).cuda()
        random_res = model.decode(z).view(-1, 1, 28, 28)
        save_image(random_res, '%s/random_sampled-%d.png' % ('./VAE_Result', epoch + 1))
        is_best = test_avg_loss < best_test_loss
        best_test_loss = min(test_avg_loss, best_test_loss)
        save_checkpoint({
            'epoch': epoch,  # 迭代次数
            'best_test_loss': best_test_loss,  # 目前最佳的损失函数值
            'state_dict': model.state_dict(),  # 当前训练过的模型的参数
            'optimizer': optimizer.state_dict(),
        }, is_best, './checkPoint')
        return best_test_loss


def main():
    # Step 1: 载入数据
    test_data, train_data, classes = dataloader()
    # print(test_data, train_data)
    # 查看每一个batch图片的规模
    # x, label = iter(train_data).__next__()  # 取出第一批(batch)训练所用的数据集
    # print(' img : ', x.shape)  # img :  torch.Size([batch_size, 1, 28, 28])， 每次迭代获取batch_size张图片，每张图大小为(1,28,28)
    # Step 2: 准备工作 : 搭建计算流程
    model = VAE(z=z_dim).cuda()  # 生成AE模型，并转移到GPU上去
    # print('The structure of our model is shown below: \n')
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 生成优化器，需要优化的是model的参数，学习率为0.001
    # Step 3: optionally resume(恢复) from a checkpoint
    start_epoch = 0
    best_test_loss = np.finfo('f').max
    if resume:
        if os.path.isfile(resume):
            # 载入已经训练过的模型参数与结果
            print('=> loading checkpoint %s' % resume)
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % resume)
        else:
            print('=> no checkpoint found at %s' % resume)
    if not os.path.exists('./VAEResult'):
        os.makedirs('./VAEResult')
    # Step 4: 开始迭代
    loss_epoch = []
    for epoch in range(start_epoch, EPOCH):
        # 训练模型
        # 每一代都要遍历所有的批次
        loss_batch = []
        for batch_index, (x, _) in enumerate(train_data):
            print(x.shape)
            # x : [b, 1, 28, 28], remember to deploy the input on GPU
            x = x.cuda()
            # 前向传播
            x_hat, mu, log_var = model(x)  # 模型的输出，在这里会自动调用model中的forward函数
            loss, BCE, KLD = loss_function(x_hat, x, mu, log_var)  # 计算损失值，即目标函数
            loss_batch.append(loss.item())  # loss是Tensor类型
            # 后向传播
            optimizer.zero_grad()  # 梯度清零，否则上一步的梯度仍会存在
            loss.backward()  # 后向传播计算梯度，这些梯度会保存在model.parameters里面
            optimizer.step()  # 更新梯度，这一步与上一步主要是根据model.parameters联系起来了
            # print statistics every 100 batch
            if (batch_index + 1) % 100 == 0:
                print('Epoch [{}/{}], Batch [{}/{}] : Total-loss = {:.4f}, BCE-Loss = {:.4f}, KLD-loss = {:.4f}'
                      .format(epoch + 1, EPOCH, batch_index + 1, len(train_data.dataset) // BATCH_SIZE,
                              loss.item() / BATCH_SIZE, BCE.item() / BATCH_SIZE,
                              KLD.item() / BATCH_SIZE))
            if batch_index == 0:
                # visualize reconstructed result at the beginning of each epoch
                x_concat = torch.cat([x.view(-1, 1, 28, 28), x_hat.view(-1, 1, 28, 28)], dim=3)
                save_image(x_concat, '%s/reconstructed-%d.png' % ('./VAE_Result', epoch + 1))
        # 把这一个epoch的每一个样本的平均损失存起来
        loss_epoch.append(np.sum(loss_batch) / len(train_data.dataset))  # len(mnist_train.dataset)为样本个数
        # 测试模型
        if (epoch + 1) % test_every == 0:
            best_test_loss = test(model, optimizer, test_data, epoch, best_test_loss)
    return loss_epoch


class VAE(nn.Module):
    def __init__(self, input=784, h=400, z=20):
        super(VAE, self).__init__()
        self.input_dim = input
        self.h_dim = h
        self.z_dim = z
        self.fc1 = nn.Linear(self.input_dim, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, self.z_dim)
        self.fc3 = nn.Linear(self.h_dim, self.z_dim)

        self.fc4 = nn.Linear(self.z_dim, self.h_dim)
        self.fc5 = nn.Linear(self.h_dim, self.input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  # 这里的“*”是点乘的意思

    def decode(self, z):
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))  # 图片数值取值为[0,1]，不宜用ReLU
        return x_hat

    def forward(self, x):
        batch_size = x.shape[0]  # 每一批含有的样本的个数
        # print(batch_size, self.input_dim)
        # flatten  [b, batch_size, 1, 28, 28] => [b, batch_size, 784]
        # tensor.view()方法可以调整tensor的形状，但必须保证调整前后元素总数一致。view不会修改自身的数据，
        # 返回的新tensor与原tensor共享内存，即更改一个，另一个也随之改变。
        x = x.view(batch_size, self.input_dim)  # 一行代表一个样本
        # encoder
        mu, log_var = self.encode(x)
        # reparameterization trick
        sampled_z = self.reparameterization(mu, log_var)
        # decoder
        x_hat = self.decode(sampled_z)
        # reshape
        x_hat = x_hat.view(batch_size, 1, 28, 28)
        return x_hat, mu, log_var


if __name__ == '__main__':
    # test_data, train_data, classes = dataloader()
    # print(train_data.dataset, test_data.dataset)
    loss_epoch = main()
    # 绘制迭代结果
    plt.plot(loss_epoch)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
