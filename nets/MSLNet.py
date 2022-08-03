import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.Modules.DCNV2 import DeformableConv2d
from nets.Backbones.MobileNetV2 import mobilenetv2_xbn
import warnings
warnings.filterwarnings("ignore")

class MobileNetV2_XBN(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2_XBN, self).__init__()
        from functools import partial
        
        model           = mobilenetv2_xbn(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        l1 = self.features[:4](x) 
        l2 = self.features[4:7](l1)
        l4 = self.features[7:](l2) 
        return l1, l2, l4 



class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()

		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)

		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class MSLNet(nn.Module):
    def __init__(self, num_classes, backbone="mobilenetv2_xbn", pretrained=True, downsample_factor=8, Use_DCN=True):
        super(MSLNet, self).__init__()
        if backbone=="mobilenetv2_xbn":
            self.backbone = MobileNetV2_XBN(downsample_factor=downsample_factor, pretrained=pretrained)
        else:
            raise ValueError('Unsupported backbone - `{}`!!!'.format(backbone))

        l1_channels = 24
        l2_channels = 32
        l4_channels = 320

        self.aspp = ASPP(dim_in=l4_channels, dim_out=256, rate=16//downsample_factor)
        
        self.cat_convUp1= nn.Sequential(
            nn.Conv2d(64+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.cat_convUp2= nn.Sequential(
            nn.Conv2d(256+48, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.shortcut_convl1 = nn.Sequential(
            nn.Conv2d(l1_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )	        
        self.shortcut_convl2 = nn.Sequential(
            nn.Conv2d(l2_channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )	
        self.conv33 = nn.Sequential(
            DeformableConv2d(256, 256, 3, stride=1, padding=1) if Use_DCN else nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # 获取初始三层特征
        # l1: 24 128 128
        # l2: 32 64 64
        # l4: 320 64 64
        l1, l2, l4 = self.backbone(x) 
        # ASPP
        up1 = self.aspp(l4) # 320 64 64 -> 256,64,64
        # 调整通道数
        l2 = self.shortcut_convl2(l2) # 64,64,64
        # 特征融合
        up2 = self.cat_convUp1(torch.cat((up1, l2), dim=1)) # 256, 64, 64
        # 调整通道数
        l1  = self.shortcut_convl1(l1) #->24 128 128 -> 48 128 128
        # 修改尺寸
        up2 = F.interpolate(up2, size=(l1.size(2), l1.size(3)), mode='bilinear', align_corners=True) #256, 128, 128
        # 特征融合
        x = self.cat_convUp2(torch.cat((up2, l1), dim=1)) # 256, 128, 128
        # 可变形卷积
        x = self.conv33(x) # 256 128 128
        # 通道调整为类别数2
        x = self.cls_conv(x) # 2 128 128
        # 调整到跟输入图片一样的大小
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) # 2,512,512
        return x
