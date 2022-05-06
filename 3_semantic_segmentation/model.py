#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.activate(x)

        return outputs


class FeatureMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = conv2DBatchNorm(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.block2 = conv2DBatchNorm(64,64,3,1,1,1,False)
        self.block3 = conv2DBatchNorm(64,128,3,1,1,1,False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)

        return x


#%%
class ResModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)

        return x


class bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super().__init__()
        self.block1 = conv2DBatchNorm(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.block2 = conv2DBatchNorm(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.block3 = ResModule(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.res = ResModule(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)

        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.block3(self.block2(self.block1(x)))
        residual = self.res(x)
        
        return self.activate(conv + residual)


class bottleneckIdentify(nn.Module):
    def __init__(self, in_channels, mid_channels, dilation):
        super().__init__()
        self.block1 = conv2DBatchNorm(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.block2 = conv2DBatchNorm(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.block3 = ResModule(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.block3(self.block2(self.block1(x)))
        
        return self.activate(conv+x)


class Resblock(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super().__init__()

        self.add_module(
            "block1",
            bottleneck(in_channels, mid_channels, out_channels, stride, dilation)
        )

        for i in range(n_blocks-1):
            self.add_module(
                "block" + str(i+2),
                bottleneckIdentify(out_channels, mid_channels, dilation)
            )


#%%
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, height, width):
        super().__init__()
        self.height = height
        self.width = width

        out_channels = int(in_channels/len(pool_sizes))

        self.module = []
        for i in range(len(pool_sizes)):
            self.module.append(nn.AdaptiveAvgPool2d(pool_sizes[i]))
            self.module.append(conv2DBatchNorm(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False))

    def forward(self, x):
        output = x
        for i in range(len(self.module)//2):
            out = self.module[i*2+1](self.module[i*2](x))
            out = F.interpolate(out, size=(self.height, self.width), mode="bilinear", align_corners=True)
            output = torch.cat([output, out], dim=1)

        return output


class DecodeFeature(nn.Module):
    def __init__(self, height, width, n_classes):
        super().__init__()

        self.height = height
        self.width = width

        self.block1 = conv2DBatchNorm(in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.block1(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width), mode="bilinear", align_corners=True)

        return output

class Auxiliary(nn.Module):
    def __init__(self, in_channels, height, width, n_classes):
        super().__init__()

        self.height = height
        self.width = width

        self.block1 = conv2DBatchNorm(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.block1(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width), mode="bilinear", align_corners=True)

        return output


#%%
class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        block_config = [3, 4, 6, 3]
        img_size=475
        img_size_8 = 60

        self.feature_conv = FeatureMap()
        self.res_block1 = Resblock(n_blocks=block_config[0], in_channels=128, mid_channels=64, out_channels=256, stride=1, dilation=1)
        self.res_block2 = Resblock(block_config[1], 256, 128, 512, 2, 1)
        self.res_block3 = Resblock(block_config[2], 512, 256, 1024, 1, 2)
        self.res_block4 = Resblock(block_config[3], 1024, 512, 2048, 1, 1)

        self.pool = PyramidPooling(in_channels=2048, pool_sizes=[6, 3, 2, 1], height=img_size_8, width=img_size_8)

        self.decode = DecodeFeature(height=img_size, width=img_size, n_classes=n_classes)
        self.aux = Auxiliary(in_channels=1024, height=img_size, width=img_size, n_classes=n_classes)

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        aux = self.aux(x)

        x = self.res_block4(x)
        x = self.pool(x)
        output = self.decode(x)

        return output, aux



#%%
if __name__=="__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = PSPNet(n_classes=21).to(device)
    batch_size = 2
    dummy = torch.rand(batch_size, 3, 475, 475)
    dummy = dummy.to(device)
    outputs = net(dummy)
    print(outputs)


# %%
