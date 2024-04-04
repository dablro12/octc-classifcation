import torch 
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F 
import timm 

class FCN(nn.Module):
    # input = channels * width * height  
    def __init__(self):
        super(FCN, self).__init__()
        
        # Block 단위로 만듬 2D Conv + BatchNorm + LeakyReLU로 정의
        def block(in_channels, out_channels, kernel_size = 3, stride =1 , padding = 1, bias = True):
            layers = []
            ## Conv 2D
            layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)]
            ## Batch Norm
            layers += [nn.BatchNorm2d(num_features = out_channels)]
            ## LeakyReLU
            layers += [nn.LeakyReLU()]
            
            cbr = nn.Sequential(*layers)
            return cbr     
        
        # 1->4->16->64 --> 1 
        self.enc1 = block(in_channels= 1, out_channels=4)
        self.enc2 = block(in_channels= 4, out_channels= 16)
        self.enc3 = block(in_channels= 16, out_channels= 64)
        self.fc = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 1, stride = 1, padding = 0, bias = True)
        
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        out = self.fc(enc3)
        return out
class UNet_2d(nn.Module):
    # input = channels * width * height  
    def __init__(self):
        super(UNet_2d, self).__init__() 
        
        # 네트워크에서 반복적으로 사용하는 Convolution + BatchNormalize + Relu 를 하나의 block으로 정의
        def CBR2d(in_channels, out_channels, kernel_size = 3, stride =1, padding = 1, bias = True):
            layers = []
            ## conv2d
            layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_szie = kernel_size, stride = stride, padding = padding, bias = bias)]
            ## batchnorm2d
            layers += [nn.BatchNorm2d(num_features = out_channels)]
            ## ReLU
            layers += [nn.ReLU()]
            
            cbr = nn.Sequential(*layers)
            
            return cbr
        
        ## Encoder 
        self.enc1_1 = CBR2d(in_channels= 1, out_channels= 64)
        self.enc1_2 = CBR2d(in_channels= 64, out_channels= 64)
        
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        
        self.enc2_1 = CBR2d(in_channels= 64, out_channels= 128)
        self.enc2_2 = CBR2d(in_channels= 128, out_channels= 128)
        
        self.pool2 = nn.MaxPool2d(kernel_size = 2)
        
        self.enc3_1 = CBR2d(in_channels= 128, out_channels= 256)
        self.enc3_2 = CBR2d(in_channels= 256, out_channels= 256)
        
        self.pool3 = nn.MaxPool2d(kernel_size = 2)
        
        self.enc4_1 = CBR2d(in_channels= 256, out_channels= 512)
        self.enc4_2 = CBR2d(in_channels= 512, out_channels= 512)
        
        self.pool4 = nn.MaxPool2d(kernel_size = 2)
        
        self.enc5_1 = CBR2d(in_channels = 512, out_channels = 1024)
        ## Decoder 
        self.dec5_1 = CBR2d(in_channels= 1024, out_channels = 512)
        
        self.unpool4 = nn.MaxUnpool2d(kernel_size = 2)
        
        self.dec4_2 = CBR2d(in_channels= 2 * 512, out_channels= 512)
        self.dec4_1 = CBR2d(in_channels= 512, out_channels= 256)
        
        self.unpool3 = nn.MaxUnpool2d(kernel_size = 2)
        
        self.dec3_2 = CBR2d(in_channels= 2 * 256, out_channels= 256)
        self.dec3_1 = CBR2d(in_channels= 256, out_channels= 128)
        
        self.unpool2 = nn.MaxUnpool2d(kernel_size = 2)
        
        self.dec2_2 = CBR2d(in_channels= 2 * 128, out_channels= 128)
        self.dec2_1 = CBR2d(in_channels= 128, out_channels= 64)
        
        self.unpool1 = nn.MaxUnpool2d(kernel_size = 2)
        
        self.dec1_2 = CBR2d(in_channels= 2 * 64, out_channels= 64)
        self.dec1_1 = CBR2d(in_channels= 64, out_channels= 64)
        
        self.fc = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size =1, stride = 1, padding = 0, bias = True)
        
    def forward(self, x):
        # Channel : 1 --> 64
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)
        
        # Channel : 64 --> 128
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)
        
        # Channel : 128 --> 256
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        
        # Channel : 256 --> 512
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)
        
        # Channel : 512 --> 1024 --> 512
        enc5_1 = self.enc5_1(pool4)
        dec5_1 = self.dec5_1(enc5_1)
        
        # Channel : 1024 --> 512 
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim = 1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        
        # Channel : 512 --> 256
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim = 1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        # Channel : 256 --> 128
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim = 1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        # Channel : 128 --> 64
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim = 1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        # Channel -> FCL 로 전환 
        out = self.fc(dec1_1) 
        
        return out 
class UNet_3d(nn.Module):
    # input = channels * width * height  
    def __init__(self):
        super(UNet_3d, self).__init__() 
        # 네트워크에서 반복적으로 사용하는 Convolution + BatchNormalize + Relu 를 하나의 block으로 정의
        def CBR3d(in_channels, out_channels, kernel_size = 3, stride =1, padding = 1, bias = True):
            layers = []
            ## conv3d
            layers += [nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_szie = kernel_size, stride = stride, padding = padding, bias = bias)]
            ## batchnorm2d
            layers += [nn.BatchNorm3d(num_features = out_channels)]
            layers += [nn.ReLU()]
            
            cbr = nn.Sequential(*layers)
            
            return cbr
        
        ## Encoder 
        self.enc1_1 = CBR3d(in_channels= 1, out_channels= 32)
        self.enc1_2 = CBR3d(in_channels= 32, out_channels= 32)
        
        self.pool1 = nn.MaxPool3d(kernel_size = 2)
        
        self.enc2_1 = CBR3d(in_channels= 32, out_channels= 64)
        self.enc2_2 = CBR3d(in_channels= 64, out_channels= 64)
        
        self.pool2 = nn.MaxPool3d(kernel_size = 2)
        
        self.enc3_1 = CBR3d(in_channels= 64, out_channels= 128)
        self.enc3_2 = CBR3d(in_channels= 128, out_channels= 128)
        
        self.pool3 = nn.MaxPool3d(kernel_size = 2)
        
        self.enc4_1 = CBR3d(in_channels= 128, out_channels= 256)
        self.enc4_2 = CBR3d(in_channels= 256, out_channels= 256)
        
        self.pool4 = nn.MaxPool3d(kernel_size = 2)
        
        self.enc5_1 = CBR3d(in_channels = 256, out_channels = 512)
        ## Decoder 
        self.dec5_1 = CBR3d(in_channels= 512, out_channels = 256)
        
        self.unpool4 = nn.MaxUnpool3d(kernel_size = 2)
        
        self.dec4_2 = CBR3d(in_channels= 2 * 256, out_channels= 256)
        self.dec4_1 = CBR3d(in_channels= 256, out_channels= 128)
        
        self.unpool3 = nn.MaxUnpool3d(kernel_size = 2)
        
        self.dec3_2 = CBR3d(in_channels= 2 * 128, out_channels= 128)
        self.dec3_1 = CBR3d(in_channels= 128, out_channels= 64)
        
        self.unpool2 = nn.MaxUnpool3d(kernel_size = 2)
        
        self.dec2_2 = CBR3d(in_channels= 2 * 64, out_channels= 64)
        self.dec2_1 = CBR3d(in_channels= 64, out_channels= 32)
        
        self.unpool1 = nn.MaxUnpool3d(kernel_size = 2)
        
        self.dec1_2 = CBR3d(in_channels= 2 * 32, out_channels= 32)
        self.dec1_1 = CBR3d(in_channels= 32, out_channels= 32)
        
        self.fc = nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size =1, stride = 1, padding = 0, bias = True)
        
    def forward(self, x):
        # Channel : 1 --> 32
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)
        
        # Channel : 32 --> 64
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)
        
        # Channel : 64 --> 128
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        
        # Channel : 128 --> 256
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)
        
        # Channel : 256 --> 512 --> 256
        enc5_1 = self.enc5_1(pool4)
        dec5_1 = self.dec5_1(enc5_1)
        
        # Channel : 512 --> 256 
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim = 1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        
        # Channel : 256 --> 128
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim = 1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        # Channel : 128 --> 64
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim = 1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        # Channel : 64 --> 32
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim = 1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        # Channel -> FCL 로 전환 
        out = self.fc(dec1_1) 
        
        return out
class Conv_AutoEncoder_3D(nn.Module):
    # input = channels * width * height  
    def __init__(self):
        super(Conv_AutoEncoder_3D, self).__init__()
        ## encoder 
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels= 1, out_channels = 16, kernel_size = 5, bias = True),
            nn.BatchNorm3d(num_features = 16),
            nn.ReLU(),
            nn.Conv3d(in_channels= 16, out_channels = 32, kernel_size = 5, bias = True),
            nn.BatchNorm3d(num_features = 32), #BatchnNorm3d 에서 32로 가야됨 
            nn.ReLU(),
            nn.Conv3d(in_channels= 32, out_channels = 64, kernel_size = 5, bias = True),
            nn.BatchNorm3d(num_features = 64),
            nn.ReLU(),
        )
        ## decoder 
        self.decoder = nn.Sequential(
            nn.Conv3d(in_channels= 64, out_channels = 32, kernel_size = 5, bias = True),
            nn.BatchNorm3d(num_features = 32),
            nn.ReLU(),
            nn.Conv3d(in_channels= 32, out_channels = 16, kernel_size = 5, bias = True),
            nn.BatchNorm3d(num_features = 16), #BatchnNorm3d 에서 32로 가야됨 
            nn.ReLU(),
            nn.Conv3d(in_channels= 16, out_channels = 1, kernel_size = 5, bias = True),
            nn.BatchNorm3d(num_features = 1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1 , hidden_dim2):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),            
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        out = x.view(x.size(0), -1) #flatten 시키기
        out = self.encoder(out)
        out = self.decoder(out)
        out = out.view(x.size()) #input이미지와 사이즈 동일하게 다시 만들기
        return out 
    
    ## hidden state 값 = latent vector 확인 
    def hidden_state(self, x): 
        return self.encoder(x)

####################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################

class pretrained_vgg16_multi(nn.Module):
    def __init__(self):
        super(pretrained_vgg16_multi, self).__init__()
        self.base_model = models.vgg16(weights = models.VGG16_Weights)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.classifier[-1] = nn.Linear(4096, 3)
        
    def forward(self, x):
        return F.softmax(self.base_model(x), dim = 1)  #Sigmoid 처리X
    
    

class pretrained_mobilenet_multi(nn.Module):
    def __init__(self):
        super(pretrained_mobilenet_multi, self).__init__()
        self.base_model = models.mobilenet_v2(weights = models.MobileNet_V2_Weights)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.classifier[-1] = nn.Linear(1280, 3)
        
    def forward(self, x):
        return F.softmax(self.base_model(x), dim = 1)
    

class pretrained_resnet18_multi(nn.Module):
    def __init__(self):
        super(pretrained_resnet18_multi, self).__init__()
        self.base_model = models.resnet18(weights = models.ResNet18_Weights)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.fc = nn.Linear(512, 3)
        
    def forward(self, x):
        return F.softmax(self.base_model(x), dim = 1)


class pretrained_efficient_multi(nn.Module):
    def __init__(self):
        super(pretrained_efficient_multi, self).__init__()
        self.base_model = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.classifier[-1] = nn.Linear(1280, 3)
        
    def forward(self, x):
        return F.softmax(self.base_model(x), dim = 1)


class pretrained_convnext_multi(nn.Module):
    def __init__(self):
        super(pretrained_convnext_multi, self).__init__()
        self.base_model = models.convnext_large(weights = models.ConvNeXt_Large_Weights)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        # self.base_model.classifier[-1] = nn.Linear(1024, 3)
        self.base_model.classifier[-1] = nn.Linear(1536, 3)
        
    def forward(self, x):
        return F.softmax(self.base_model(x), dim = 1)
    
class pretrained_swin_multi(nn.Module):
    def __init__(self):
        super(pretrained_swin_multi, self).__init__()
        self.base_model = timm.create_model('swin_small_patch4_window7_224.ms_in22k', pretrained=True,  num_classes=3)

        
    def forward(self, x):
        return F.softmax(self.base_model(x), dim = 1)

####################################################################################################################################################################################################################################################################################################################

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups 
        
    def forward(self, x):
        batch_size , num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups 
        
        # reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, height, width)
        return x 

class ECALayer(nn.Module):
    def __init__(self, out_channels, kernel_size):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(in_channels = out_channels, out_channels= out_channels, kernel_size=kernel_size, padding = 1, stride = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.unsqueeze(-1)
        y = x * y.expand_as(x)
        return y


class BlockUnit(nn.Module):
    def __init__(self, out_channels, stride):
        super(BlockUnit, self).__init__()
        
        self.left = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, groups= out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size= 1, groups= out_channels),
            ECALayer(out_channels, kernel_size = 3)
        )
        self.right = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size= 7, padding = 2, groups= out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 2, groups= out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size= 7, padding = 3, groups= out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.channel_shuffle = ChannelShuffle(groups = 2)
        
    def forward(self, x):
        out_left = self.left(x)
        out_right = self.right(x)
        out = torch.concat((out_left, out_right), dim = 1)
        out = self.channel_shuffle(out)
        return out
    
    
class OcysNet(nn.Module):
    def __init__(self):
        super(OcysNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 24, kernel_size= 4, stride = 4, padding = 2)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.block1 = nn.ModuleList([
            BlockUnit(out_channels= 24, stride = 2),
            BlockUnit(out_channels= 48, stride = 2),
            BlockUnit(out_channels= 96, stride = 2),
        ])

        self.block2 = nn.ModuleList([
            BlockUnit(out_channels= 24, stride = 2),
            BlockUnit(out_channels= 48, stride = 2),
            BlockUnit(out_channels= 96, stride = 2),
        ])
        
        self.block3 = nn.ModuleList([
            BlockUnit(out_channels= 24, stride = 2),
            BlockUnit(out_channels= 48, stride = 2),
            BlockUnit(out_channels= 96, stride = 2),
        ])
        self.block4 = nn.ModuleList([
            BlockUnit(out_channels= 24, stride = 2),
            BlockUnit(out_channels= 48, stride = 2),
            BlockUnit(out_channels= 96, stride = 2),
        ])
        self.block5 = nn.ModuleList([
            BlockUnit(out_channels= 24, stride = 2),
            BlockUnit(out_channels= 48, stride = 2),
            BlockUnit(out_channels= 96, stride = 2),
        ])

        self.conv5 = nn.Conv2d(in_channels = 192, out_channels = 1024, kernel_size = 1, stride = 1)
        self.gbp = nn.AvgPool2d(kernel_size= 7, padding = 2)
        self.lm = None
        self.fc1 = nn.Linear(1024, 1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)

        self.block_outputs = [] # 각 block 출력 저장 리스트
        
        self.out_li = [out.clone() for _ in range(5)]  # 이전 maxpool layer의 출력 복사

        # 각 block 1~5를 병렬적으로 거침
        for idx, block in enumerate([self.block1, self.block2, self.block3, self.block4, self.block5]):
            for block_unit in block:
                self.out_li[idx] = block_unit(self.out_li[idx])
                
            self.block_outputs.append(self.out_li[idx])
        
        
        self.stacked_outputs = torch.stack(self.block_outputs, dim = 1)
        self.weights = torch.softmax(self.stacked_outputs, dim = 1)
        out = torch.sum(self.weights * self.stacked_outputs, dim = 1)
        out = self.conv5(out)
        out = self.gbp(out)
        
        bs, channels, height, width = out.shape 
        if self.lm is None:
            out = out.view(bs, channels, -1)
            normal_shape = out.shape[-1]
            self.lm = nn.LayerNorm(normalized_shape= normal_shape, device = out.device)
        out = self.lm(out)
        out = out.view(bs, channels, height, width)
        out = out.squeeze(-1).squeeze(-1)
        out = self.fc1(out)
        return out.view(-1)


class pretrained_unet_encoder_binary(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(pretrained_unet_encoder_binary, self).__init__()
        self.base_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=in_channels, out_channels=out_channels, init_features=init_features, pretrained=True)

        self.enc1 = self.base_model.encoder1
        self.pool1 = self.base_model.pool1
        self.enc2 = self.base_model.encoder2
        self.pool2 = self.base_model.pool2
        self.enc3 = self.base_model.encoder3
        self.pool3 = self.base_model.pool3
        self.enc4 = self.base_model.encoder4
        self.pool4 = self.base_model.pool4
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(256, 128),  # Adjust the input and output dimensions as needed
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(128, 1)  # Output layer with 1 class for binary classification
        )

    def forward(self, x):
        out = self.enc1(x)
        out = self.pool1(out)
        out = self.enc2(out)
        out = self.pool2(out)
        out = self.enc3(out)
        out = self.pool3(out)
        out = self.enc4(out)
        out = self.pool4(out)
        # Pass the output through the classifier
        out = self.classifier(out)

        return out.view(-1)
    
####################################################################################################################################################################################################################################################################################################################
    

class pretrained_unet_encoder_multi(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(pretrained_unet_encoder_multi, self).__init__()
        self.base_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=in_channels, out_channels=out_channels, init_features=init_features, pretrained=True)

        self.enc1 = self.base_model.encoder1
        self.pool1 = self.base_model.pool1
        self.enc2 = self.base_model.encoder2
        self.pool2 = self.base_model.pool2
        self.enc3 = self.base_model.encoder3
        self.pool3 = self.base_model.pool3
        self.enc4 = self.base_model.encoder4
        self.pool4 = self.base_model.pool4
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(256, 128),  # Adjust the input and output dimensions as needed
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(128, 3)  # Output layer with 1 class for binary classification
        )

    def forward(self, x):
        out = self.enc1(x)
        out = self.pool1(out)
        out = self.enc2(out)
        out = self.pool2(out)
        out = self.enc3(out)
        out = self.pool3(out)
        out = self.enc4(out)
        out = self.pool4(out)
        # Pass the output through the classifier
        out = self.classifier(out)

        return F.softmax(self.base_model(out), dim = 1)
    
    
####################################################################################################################################################################################################################################################################################################################
    