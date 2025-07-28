from torch import nn
import torch
import monai

class SFCNModel(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv3d(n_channels,32,kernel_size=(3,3,3),padding='same')
        self.norm1 = nn.LazyBatchNorm3d(track_running_stats=False)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.relu1 = nn.PReLU()
        self.block1 = nn.Sequential(self.conv1, self.norm1, self.maxpool1, self.relu1)
        #self.block1 = nn.Sequential(self.conv1, self.maxpool1, self.relu1)

        # Block 2
        self.conv2 = nn.Conv3d(32,64,kernel_size=(3,3,3),padding='same')
        self.norm2 = nn.LazyBatchNorm3d(track_running_stats=False)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.relu2 = nn.PReLU()
        self.block2 = nn.Sequential(self.conv2, self.norm2, self.maxpool2, self.relu2)
        #self.block2 = nn.Sequential(self.conv2, self.maxpool2, self.relu2) 
        
        # Block 3
        self.conv3 = nn.Conv3d(64,128,kernel_size=(3,3,3),padding='same')
        self.norm3 = nn.LazyBatchNorm3d(track_running_stats=False)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.relu3 = nn.PReLU()
        self.block3 = nn.Sequential(self.conv3, self.norm3, self.maxpool3, self.relu3)
        #self.block3 = nn.Sequential(self.conv3, self.maxpool3, self.relu3)


        # Block 4
        self.conv4 = nn.Conv3d(128,256,kernel_size=(3,3,3),padding='same')
        self.norm4 = nn.LazyBatchNorm3d(track_running_stats=False)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.relu4 = nn.PReLU()
        self.block4 = nn.Sequential(self.conv4, self.norm4, self.maxpool4, self.relu4)
        #self.block4 = nn.Sequential(self.conv4, self.maxpool4, self.relu4)


        # Block 5
        self.conv5 = nn.Conv3d(256,256,kernel_size=(3,3,3),padding='same')
        self.norm5 = nn.LazyBatchNorm3d(track_running_stats=False)
        self.maxpool5 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.relu5 = nn.PReLU()
        self.block5 = nn.Sequential(self.conv5, self.norm5, self.maxpool5, self.relu5)
        #self.block5 = nn.Sequential(self.conv5, self.maxpool5, self.relu5)

        # Block 6
        self.conv6 = nn.Conv3d(256,64, kernel_size=(1,1,1), padding='same')
        self.norm6 = nn.LazyBatchNorm3d(track_running_stats=False)
        self.relu6 = nn.PReLU()
        self.block6 = nn.Sequential(self.conv6, self.norm6, self.relu6)
        #self.block6 = nn.Sequential(self.conv6, self.relu6)

        # Block 7
        self.avgpool1 = nn.AvgPool3d(kernel_size=(1,1,1))
        self.dropout1 = nn.Dropout(.5)
        self.flat1 = nn.Flatten()
        self.linear1 = nn.LazyLinear(2)
        self.block7 = nn.Sequential(self.avgpool1, self.flat1, self.dropout1, self.linear1)
        #self.block7 = nn.Sequential(self.avgpool1, self.flat1, self.linear1)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        return torch.squeeze(x,-1)#argh, that should be handled differently and outside of the network
    

class SFCNModelMONAI(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1=nn.Sequential(monai.networks.blocks.Convolution(3, 1, 32, strides=1, kernel_size=3,padding='same'),nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)))
        self.block2=nn.Sequential(monai.networks.blocks.Convolution(3, 32, 64, strides=1, kernel_size=3,padding='same'),nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)))
        self.block3=nn.Sequential(monai.networks.blocks.Convolution(3, 64, 128, strides=1, kernel_size=3,padding='same'),nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)))       
        self.block4=nn.Sequential(monai.networks.blocks.Convolution(3, 128, 256, strides=1, kernel_size=3,padding='same'),nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)))
        self.block5=nn.Sequential(monai.networks.blocks.Convolution(3, 256, 256, strides=1, kernel_size=3,padding='same'),nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)))
        self.block6=monai.networks.blocks.Convolution(3, 256, 64, strides=1, kernel_size=1)   
        
        # Block 7
        self.avgpool1 = nn.AvgPool3d(kernel_size=(2,2,2))
        self.dropout1 = nn.Dropout(.2)
        self.flat1 = nn.Flatten()
        self.linear1 = nn.LazyLinear(1)
        self.block7 = nn.Sequential(self.avgpool1, self.flat1, self.dropout1, self.linear1)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        return torch.squeeze(x,-1)#argh, that should be handled differently and outside of the network
    

class SFCNModelMONAIClassification(nn.Module):
    def __init__(self):
        super().__init__()

        initial_depth=32

        self.block1=nn.Sequential(monai.networks.blocks.Convolution(3, 1, initial_depth, strides=1, kernel_size=3,act="RELU"),nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)))
        self.block2=nn.Sequential(monai.networks.blocks.Convolution(3, initial_depth, initial_depth*2, strides=1, kernel_size=3,act="RELU"),nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)))
        self.block3=nn.Sequential(monai.networks.blocks.Convolution(3, initial_depth*2, initial_depth*3, strides=1, kernel_size=3,act="RELU"),nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)))       
        self.block4=nn.Sequential(monai.networks.blocks.Convolution(3, initial_depth*3, initial_depth*4, strides=1, kernel_size=3,act="RELU"),nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)))
        self.block5=nn.Sequential(monai.networks.blocks.Convolution(3, initial_depth*4, initial_depth*4, strides=1, kernel_size=3,act="RELU"),nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)))
        self.block6=monai.networks.blocks.Convolution(3, initial_depth*4, initial_depth*2, strides=1, kernel_size=1,act="RELU")   
        
        # Block 7
        self.avgpool1 = nn.AdaptiveAvgPool3d(1)#kernel_size=(2,2,2))
        self.dropout1 = nn.Dropout(.5)
        self.flat1 = nn.Flatten()
        self.linear1 = nn.LazyLinear(1)
        self.block7 = nn.Sequential(self.avgpool1, self.flat1, self.dropout1, self.linear1)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        return torch.squeeze(x,-1)#argh, that should be handled differently and outside of the network