### Section 1 - First, let's import everything we will be needing.

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import ImageFile
from model import MBConv

ImageFile.LOAD_TRUNCATED_IMAGES = True
BASE_LR = 0.002
EPOCH_DECAY = 10  # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.95  # factor by which the learning rate is reduced.

# DATASET INFO
NUM_CLASSES = 2  # set the number of classes in your dataset
# DATA_DIR = 'T2andDce_two/' # to run with the sample dataset, just set to 'hymenoptera_data'

# DATALOADER PROPERTIES
BATCH_SIZE = 8  # Set as high as possible. If you keep it too high, you'll get an out of memory error.

### GPU SETTINGS
CUDA_DEVICE = 0  # Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
GPU_MODE = 0  # set to 1 if want to run on gpu.

# SETTINGS FOR DISPLAYING ON TENSORBOARD
USE_TENSORBOARD = 0  # if you want to use tensorboard set this to 1.
TENSORBOARD_SERVER = "YOUR TENSORBOARD SERVER ADDRESS HERE"  # If you set.
EXP_NAME = "fine_tuning_experiment"  # if using tensorboard, enter name of experiment you want it to be displayed as.
data = 0
if data == 0:
    DATA_DIR = './'

#else:
    #DATA_DIR = 'BK breast/'

## If you want to keep a track of your network on tensorboard, set USE_TENSORBOARD TO 1 in config file.

if USE_TENSORBOARD:
    from pycrayon import CrayonClient

    cc = CrayonClient(hostname=TENSORBOARD_SERVER)
    try:
        cc.remove_experiment(EXP_NAME)
    except:
        pass
    foo = cc.create_experiment(EXP_NAME)

## If you want to use the GPU, set GPU_MODE TO 1 in config file

use_gpu = GPU_MODE
if use_gpu:
    torch.cuda.set_device(CUDA_DEVICE)

count = 0

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.Grayscale(),
        # transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    ]),
    'val': transforms.Compose([
        # transforms.Grayscale(),
        # transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    ]),
}

data_dir = DATA_DIR
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=0)
                for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes




def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=100):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                mode = 'train'
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()
                mode = 'val'

            running_loss = 0.0
            running_corrects = 0

            counter = 0
            # Iterate over data.
            for data in dset_loaders[phase]:
                inputs, labels = data
                # print(inputs.size())
                # wrap them in Variable
                if use_gpu:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()),
                        Variable(labels.long().cuda())
                    except:
                        print(inputs, labels)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                # print('loss done')
                # Just so that you can keep track that something's happening and don't feel like the program isn't running.
                # if counter%10==0:
                #     print("Reached iteration ",counter)
                counter += 1

                # backward + optimize only if in training phase
                if phase == 'train':
                    # print('loss backward')
                    loss.backward()
                    # print('done loss backward')
                    optimizer.step()
                    # print('done optim')
                # print evaluation statistics
                try:
                    # running_loss += loss.data[0]
                    running_loss += loss.item()
                    #print(labels.data)
                    #print(preds)
                    running_corrects += torch.sum(preds == labels.data)
                    # print('running correct =',running_corrects)
                except:
                    print('unexpected error, could not calculate loss or do a sum.')
            print('trying epoch loss')
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects.item() / float(dset_sizes[phase])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if USE_TENSORBOARD:
                    foo.add_scalar_value('epoch_loss', epoch_loss, step=epoch)
                    foo.add_scalar_value('epoch_acc', epoch_acc, step=epoch)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    a = epoch
                    best_model = copy.deepcopy(model)
                    print('new best accuracy = ', best_acc)
                    torch.save(model_ft, "L" + str(epoch + 1) + "_acc_" + str(best_acc) + ".pth")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('returning and looping back')
    # print('epoch is '+a)
    return best_model


# This function changes the learning rate over the training model.
def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (DECAY_WEIGHT ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class ChannelAttentionModule(nn.Module):  # in_channels==out_channels
    def __init__(self, channel, ratio=8):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out


class CoTNetLayer(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            # 通过K*K的卷积提取上下文信息，视作输入X的静态上下文表达
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False),  # 1*1的卷积进行Value的编码
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(  # 通过连续两个1*1的卷积计算注意力矩阵
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),  # 输入concat后的特征矩阵 Channel = 2*C
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1, stride=1)  # out: H * W * (K*K*C)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # shape：bs,c,h,w  提取静态上下文信息得到key
        v = self.value_embed(x).view(bs, c, -1)  # shape：bs,c,h*w  得到value编码

        y = torch.cat([k1, x], dim=1)  # shape：bs,2c,h,w  Key与Query在channel维度上进行拼接进行拼接
        att = self.attention_embed(y)  # shape：bs,c*k*k,h,w  计算注意力矩阵
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # shape：bs,c,h*w  求平均降低维度
        k2 = nn.functional.softmax(att, dim=-1) * v  # 对每一个H*w进行softmax后
        k2 = k2.view(bs, c, h, w)

        return k1 + k2  # 注意力融合


class CM_block(nn.Module):
    def __init__(self, channel, gap_size=(1, 1)):  # 输入和输出通道数一致
        super(CM_block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # average = torch.mean(x, dim=1, keepdim=True)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2).unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


# se_inception_cm block的上半部分
class block_up(nn.Module):
    def __init__(self, input_channels, filter_num, stride=1):
        super(block_up, self).__init__()
        self.cot = CoTNetLayer(dim=filter_num[1])
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=filter_num[0],
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=filter_num[0]),
            # nn.ReLU(),
        )
        self.conv = nn.Conv2d(in_channels=filter_num[0],
                              out_channels=filter_num[1],
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.block_2 = nn.Sequential(
            nn.BatchNorm2d(num_features=filter_num[1]),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filter_num[1],
                      out_channels=filter_num[2],
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=filter_num[2]),
        )
        self.relu = nn.ReLU(inplace=True)
        self.se = SE_Block(ch_in=filter_num[2])
        # self.downsample=nn.Sequential(
        #    nn.Conv2d(input_channels,filter_num[2]*2,kernel_size=1,stride=stride,bias=False),
        #    nn.BatchNorm2d(filter_num[2]*2),
        #    )
        self.relu = nn.ReLU()

    def forward(self, x):
        input = x
        x = self.block_1(x)
        x = self.conv(x)
        x = self.block_2(x)
        output = self.relu(x)
        return output


# se_incepption_cm block的下半部分
class block_down(nn.Module):
    def __init__(self, input_channels, filter_num, stride=1):
        super(block_down, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=filter_num[0],
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=filter_num[0]),
            # nn.ReLU(),
            nn.Conv2d(in_channels=filter_num[0],
                      out_channels=filter_num[1],
                      kernel_size=(1, 3),
                      stride=1,
                      padding=(0, 1),
                      bias=False),
            nn.BatchNorm2d(num_features=filter_num[1]),
            # nn.ReLU(),
            nn.Conv2d(in_channels=filter_num[1],
                      out_channels=filter_num[2],
                      kernel_size=(3, 1),
                      stride=1,
                      padding=(1, 0),
                      bias=False),
            nn.BatchNorm2d(num_features=filter_num[2]),
            # nn.ReLU(),
            nn.Conv2d(in_channels=filter_num[2],
                      out_channels=filter_num[3],
                      kernel_size=(1, 3),
                      stride=1,
                      padding=(0, 1),
                      bias=False),
            nn.BatchNorm2d(num_features=filter_num[3]),
            # nn.ReLU(),
            nn.Conv2d(in_channels=filter_num[3],
                      out_channels=filter_num[4],
                      kernel_size=(3, 1),
                      stride=1,
                      padding=(1, 0),
                      bias=False),
            nn.BatchNorm2d(num_features=filter_num[4])
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.block(x)
        out = self.relu(x)
        return out


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):  # ch_in 为输入通道数
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Conv_Ba_Re(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, strides, paddings, relu=False):
        super(Conv_Ba_Re, self).__init__()
        self.conv = nn.Conv2d(in_channels=ch_in,
                              out_channels=ch_out,
                              kernel_size=kernel,
                              stride=strides,
                              padding=paddings,
                              bias=False)
        self.batch = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.flag = relu

    def forward(self, input):
        x = self.conv(input)
        x = self.batch(x)
        if (self.flag == True):
            x = self.relu(x)

        return x


# define model
class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.pre1 = Conv_Ba_Re(ch_in=3, ch_out=16, kernel=3, strides=1, paddings=1, relu=True)
        # def __init__(self, in_, out_, expand,kernel_size, stride, skip,se_ratio, dc_ratio=0.2):
        self.mb1 = MBConv(in_=16, out_=16, expand=8, kernel_size=3, stride=1, skip=1, se_ratio=16, dc_ratio=0.3)

        self.blockup_1 = block_up(input_channels=16, filter_num=[8, 8, 16])
        self.blockdown_1 = block_down(input_channels=16, filter_num=[4, 8, 8, 8, 16])
        self.maxpool = torch.nn.MaxPool2d((2, 2))
        self.bn = nn.BatchNorm2d(16)
        self.bn_after_maxpool_1 = nn.BatchNorm2d(32)
        self.bn_after_maxpool_2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(64, NUM_CLASSES)
        self.CM_16 = CM_block(16)
        self.mb2 = MBConv(in_=32, out_=64, expand=4, kernel_size=3, stride=2, skip=1, se_ratio=16, dc_ratio=0.3)
        self.ca = ChannelAttentionModule(channel=16)
        self.ca1 = ChannelAttentionModule(channel=64)
        self.sa = SpatialAttentionModule()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.pre1(input)

        m = self.mb1(x)
        m = self.bn(m)
        y = self.blockup_1(x)
        z = self.blockdown_1(x)
        y = self.ca(y) * y
        y = self.sa(y) * y
        z = self.ca(z) * z
        z = self.sa(z) * z
        x=y+z
        x = torch.cat([x, m], 1)
        #x = x+m

        x = self.maxpool(x)
        x = self.bn_after_maxpool_1(x)
        x = self.relu(x)

        x = self.mb2(x)
        # x=self.ca1(x)*x
        # x=self.sa(x)*x

        x = self.maxpool(x)
        x = self.bn_after_maxpool_2(x)
        x = self.relu(x)
        x = self.gap(x)
        x = self.drop(x)
        x = torch.squeeze(x)
        output = self.fc(x)
        output = nn.Softmax()(output)
        return output


### SECTION 4 : DEFINING MODEL ARCHITECTURE.

# We use Resnet18 here. If you have more computational power, feel free to swap it with Resnet50, Resnet100 or Resnet152.
# Since we are doing fine-tuning, or transfer learning we will use the pretrained net weights. In the last line, the number of classes has been specified.
# Set the number of classes in the config file by setting the right value for NUM_CLASSES.
if __name__ == '__main__':
    model_ft = BasicNet()
    # model_ft.classifier._modules['6'] = nn.Linear(4096, 2)
    # model_ft=torch.load("programming_21_acc_0.8693069306930693.pth")
    criterion = nn.CrossEntropyLoss()

    if use_gpu:
        criterion.cuda()
        model_ft.cuda()

    optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=0.002)

    # Run the functions and save the best model in the function model_ft.
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=200)

# Save model
# torch.save(model_ft, 'programming.pth')