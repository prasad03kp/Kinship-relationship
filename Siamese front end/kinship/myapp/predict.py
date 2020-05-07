import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import itertools
import os
import pandas as pd

train_on_gpu = torch.cuda.is_available()

class FamilyDataset(Dataset):
    """Family Dataset."""

    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            df (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.relations = df
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.relations)
    
    def __getpair__(self,idx):
        pair = self.root_dir+self.relations.iloc[idx,0] + '/' + self.relations.iloc[idx,1],\
        self.root_dir+self.relations.iloc[idx,2] + '/' + self.relations.iloc[idx,3]
#         print(pair)
        return pair
    
    def __getlabel__(self,idx):
        return self.relations.iloc[idx,4]
    
    def __getitem__(self, idx):
        pair =  self.__getpair__(idx)
        label = self.__getlabel__(idx)
        
        first = random.choice(os.listdir(pair[0]))
        second = random.choice(os.listdir(pair[1]))
        
        img0 = Image.open(pair[0] + '/' + first)
        img1 = Image.open(pair[1] + '/'  + second)
#         img0 = img0.convert("L")
#         img1 = img1.convert("L")
        
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            
        return idx,img0,img1,label


class Vgg_face_dag(nn.Module):

    def __init__(self):
        super(Vgg_face_dag, self).__init__()
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward_once(self, x0):
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31_preflatten = self.pool5(x30)
        x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x32 = self.fc6(x31)
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.relu7(x35)
        x37 = self.dropout7(x36)
        x38 = self.fc8(x37)
        return x38
    
    def forward(self,input1,input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        #euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        difference = output1 - output2
        return difference

def vgg_face_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg_face_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2622, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2))
  
    def forward(self, x):
        x = self.fc1(x)
        return x


def process():
    df=pd.read_csv(os.getcwd()+'\\predict.csv')

    new = df["p1"].str.split("/", n = 1, expand = True)

    # making separate first name column from new data frame 
    df["Family1"]= new[0]
    # making separate last name column from new data frame 
    df["Person1"]= new[1]

    # Dropping old Name columns
    df.drop(columns =["p1"], inplace = True)

    new = df["p2"].str.split("/", n = 1, expand = True)

    # making separate first name column from new data frame 
    df["Family2"]= new[0]
    # making separate last name column from new data frame 
    df["Person2"]= new[1]

    # Dropping old Name columns
    df.drop(columns =["p2"], inplace = True)

    df['Related']=1
    df.head()

    pred_transform = transforms.Compose([transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    pred_dataset= FamilyDataset(df=df,root_dir=os.getcwd()+'\\kinship\\media\\',transform=pred_transform)

    pred_dataloader = DataLoader(pred_dataset,
                            shuffle=False,
                            num_workers=0,
                            batch_size=1)
    dataiter = iter(pred_dataloader)

    vggnet = vgg_face_dag(weights_path=os.getcwd()+"\\pytorch-pretrained-models-for-face-detection\\VGG Face")
    vggnet = vggnet.cuda()
    vggnet.eval()
    for param in vggnet.parameters(): 
        param.requires_grad = False

    net = Model().cuda()
    net.load_state_dict(torch.load(os.getcwd()+'\\model.pt'))
    net.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.008, momentum=0.9)

    pred_loader = torch.utils.data.DataLoader(pred_dataset,shuffle=True,num_workers=0, batch_size = 1)

    for i, data in enumerate(pred_loader,0):
            row, img0, img1, label = data
            row, img0, img1  = row.cuda(), img0.cuda(), img1.cuda() 
            
            optimizer.zero_grad()
            output1= vggnet(img0,img1)
            output = net(output1)
            _, pred= torch.max(output,1)
    print(_)
    print('Result : ',pred[0]==1)

    if pred[0]==1:
        return "Belongs To The Same Family/"+str((float(_.data.tolist()[0])/2)*100)
    else:
        return "Not Belongs To The Same Family/"+str((float(_.data.tolist()[0])/2)*100)
        