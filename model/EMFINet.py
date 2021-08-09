import torch
import torch.nn as nn
from torchvision import models

from .resnet_model import *

class EMFINet(nn.Module):
    def __init__(self):
        super(EMFINet,self).__init__()

        resnet = models.resnet34(pretrained=True)

        #-------------Multiscale Feature Extraction--------------#
        self.poola = nn.MaxPool2d(2,2,ceil_mode=True)
        self.poolb = nn.MaxPool2d(4,4,ceil_mode=True)

        #stage 1
        self.preconv = nn.Conv2d(3,64,3,padding=1)
        self.prebn = nn.BatchNorm2d(64)
        self.prerelu = nn.ReLU(inplace=True)

        self.encoder1 = resnet.layer1
        #stage 2
        self.encoder2 = resnet.layer2
        #stage 3
        self.encoder3 = resnet.layer3
        #stage 4
        self.encoder4 = resnet.layer4

        self.poolc = nn.MaxPool2d(2,2,ceil_mode=True)
        #stage 5
        self.encoder5_1 = BasicBlock(512,512)
        self.encoder5_2 = BasicBlock(512,512)
        self.encoder5_3 = BasicBlock(512,512)

        #-------------Cascaded feature fusion module--------------#
        #part 1
        self.cat_conv11 = nn.Conv2d(512,512,3,padding=1)
        self.cat_bn11 = nn.BatchNorm2d(512)
        self.cat_relu11 = nn.ReLU(inplace=True)

        self.cat_conv12 = nn.Conv2d(512,512,3,padding=1)
        self.cat_bn12 = nn.BatchNorm2d(512)
        self.cat_relu12 = nn.ReLU(inplace=True)

        self.cat_conv13 = nn.Conv2d(512,512,3,padding=1)
        self.cat_bn13 = nn.BatchNorm2d(512)
        self.cat_relu13 = nn.ReLU(inplace=True)
        #part 2
        self.cat_conv21 = nn.Conv2d(512,512,3,padding=1)
        self.cat_bn21 = nn.BatchNorm2d(512)
        self.cat_relu21 = nn.ReLU(inplace=True)

        self.cat_conv22 = nn.Conv2d(512,512,3,padding=1)
        self.cat_bn22 = nn.BatchNorm2d(512)
        self.cat_relu22 = nn.ReLU(inplace=True)

        self.cat_conv23 = nn.Conv2d(512,512,3,padding=1)
        self.cat_bn23 = nn.BatchNorm2d(512)
        self.cat_relu23 = nn.ReLU(inplace=True)
        #part 3
        self.cat_conv31 = nn.Conv2d(512,512,3,padding=1)
        self.cat_bn31 = nn.BatchNorm2d(512)
        self.cat_relu31 = nn.ReLU(inplace=True)

        self.cat_conv32 = nn.Conv2d(512,512,3,padding=1)
        self.cat_bn32 = nn.BatchNorm2d(512)
        self.cat_relu32 = nn.ReLU(inplace=True)

        self.cat_conv33 = nn.Conv2d(512,512,3,padding=1)
        self.cat_bn33 = nn.BatchNorm2d(512)
        self.cat_relu33 = nn.ReLU(inplace=True)

        #-------------Edge--------------#
        self.up_conv = nn.Conv2d(512,128,3,dilation=8, padding=8)
        self.up_bn = nn.BatchNorm2d(128)

        self.edge_conv = nn.Conv2d(128,128,3,padding=1)
        self.edge_bn = nn.BatchNorm2d(128)
        self.edge_relu = nn.ReLU(inplace=True)

        #-------------Bridge--------------#
        self.bdg_conv1 = nn.Conv2d(512,128,3,dilation=2, padding=2)
        self.bdg_bn1 = nn.BatchNorm2d(128)
        self.bdg_conv2 = nn.Conv2d(512,128,3,dilation=4, padding=4)
        self.bdg_bn2 = nn.BatchNorm2d(128)
        self.bdg_conv3 = nn.Conv2d(512,128,3,dilation=8, padding=8)
        self.bdg_bn3 = nn.BatchNorm2d(128)
        self.bdg_conv4 = nn.Conv2d(512,128,3,dilation=16, padding=16)
        self.bdg_bn4 = nn.BatchNorm2d(128)

        #-------------deep feature aggregation module--------------#
        #stage 5
        self.pool5 = nn.MaxPool2d(8,8,ceil_mode=True)

        self.mconv5 = nn.Conv2d(1024 + 128,1024,3,padding=1)
        self.mbn5 = nn.BatchNorm2d(1024)
        self.mrelu5 = nn.ReLU(inplace=True)

        self.convd5_1 = nn.Conv2d(1024,512,3,padding=1)
        self.bnd5_1 = nn.BatchNorm2d(512)
        self.relud5_1 = nn.ReLU(inplace=True)

        self.convd5_2 = nn.Conv2d(512,512,3,padding=1)
        self.bnd5_2 = nn.BatchNorm2d(512)
        self.relud5_2 = nn.ReLU(inplace=True)

        self.convd5_3 = nn.Conv2d(512,512,3,padding=1)
        self.bnd5_3 = nn.BatchNorm2d(512)
        self.relud5_3 = nn.ReLU(inplace=True)
        #stage 4
        self.pool4 = nn.MaxPool2d(4,4,ceil_mode=True)

        self.mconv4 = nn.Conv2d(512 + 128,512,3,padding=1)
        self.mbn4 = nn.BatchNorm2d(512)
        self.mrelu4 = nn.ReLU(inplace=True)

        self.convd4_1 = nn.Conv2d(1024,512,3,padding=1)
        self.bnd4_1 = nn.BatchNorm2d(512)
        self.relud4_1 = nn.ReLU(inplace=True)

        self.convd4_2 = nn.Conv2d(512,512,3,padding=1)
        self.bnd4_2 = nn.BatchNorm2d(512)
        self.relud4_2 = nn.ReLU(inplace=True)

        self.convd4_3 = nn.Conv2d(512,256,3,padding=1)
        self.bnd4_3 = nn.BatchNorm2d(256)
        self.relud4_3 = nn.ReLU(inplace=True)
        #stage 3
        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.mconv3 = nn.Conv2d(256 + 128,256,3,padding=1)
        self.mbn3 = nn.BatchNorm2d(256)
        self.mrelu3 = nn.ReLU(inplace=True)

        self.convd3_1 = nn.Conv2d(512,256,3,padding=1)
        self.bnd3_1 = nn.BatchNorm2d(256)
        self.relud3_1 = nn.ReLU(inplace=True)

        self.convd3_2 = nn.Conv2d(256,256,3,padding=1)
        self.bnd3_2 = nn.BatchNorm2d(256)
        self.relud3_2 = nn.ReLU(inplace=True)

        self.convd3_3 = nn.Conv2d(256,128,3,padding=1)
        self.bnd3_3 = nn.BatchNorm2d(128)
        self.relud3_3 = nn.ReLU(inplace=True)
        #stage 2
        self.mconv2 = nn.Conv2d(128 + 128,128,3,padding=1)
        self.mbn2 = nn.BatchNorm2d(128)
        self.mrelu2 = nn.ReLU(inplace=True)

        self.convd2_1 = nn.Conv2d(256,128,3,padding=1)
        self.bnd2_1 = nn.BatchNorm2d(128)
        self.relud2_1 = nn.ReLU(inplace=True)

        self.convd2_2 = nn.Conv2d(128,128,3,padding=1)
        self.bnd2_2 = nn.BatchNorm2d(128)
        self.relud2_2 = nn.ReLU(inplace=True)

        self.convd2_3 = nn.Conv2d(128,64,3,padding=1)
        self.bnd2_3 = nn.BatchNorm2d(64)
        self.relud2_3 = nn.ReLU(inplace=True)
        #stage 1
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)

        self.mconv1 = nn.Conv2d(64 + 128,64,3,padding=1)
        self.mbn1 = nn.BatchNorm2d(64)
        self.mrelu1 = nn.ReLU(inplace=True)

        self.convd1_1 = nn.Conv2d(128,64,3,padding=1)
        self.bnd1_1 = nn.BatchNorm2d(64)
        self.relud1_1 = nn.ReLU(inplace=True)

        self.convd1_2 = nn.Conv2d(64,64,3,padding=1)
        self.bnd1_2 = nn.BatchNorm2d(64)
        self.relud1_2 = nn.ReLU(inplace=True)

        self.convd1_3 = nn.Conv2d(64,64,3,padding=1)
        self.bnd1_3 = nn.BatchNorm2d(64)
        self.relud1_3 = nn.ReLU(inplace=True)

        #-------------Upsampling--------------#
        self.upscore64 = nn.Upsample(scale_factor=64,mode='bilinear',align_corners=True)
        self.upscore32 = nn.Upsample(scale_factor=32,mode='bilinear',align_corners=True)
        self.upscore16 = nn.Upsample(scale_factor=16,mode='bilinear',align_corners=True)
        self.upscore8 = nn.Upsample(scale_factor=8,mode='bilinear',align_corners=True)
        self.upscore4 = nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)

        #-------------Label output--------------#
        self.label_outconv1 = nn.Conv2d(512,1,1,padding=0)
        self.label_outconv2 = nn.Conv2d(512,1,1,padding=0)
        self.label_outconv3 = nn.Conv2d(512,1,1,padding=0)
        self.label_outconv4 = nn.Conv2d(512,1,1,padding=0)
        self.label_outconv5 = nn.Conv2d(256,1,1,padding=0)
        self.label_outconv6 = nn.Conv2d(128,1,1,padding=0)
        self.label_outconv7 = nn.Conv2d(64,1,1,padding=0)
        self.label_outconv8 = nn.Conv2d(64,1,1,padding=0)
        self.label_outconv9 = nn.Conv2d(1024,1,1,padding=0)

        #-------------Edge output--------------#
        self.edge_outconv1 = nn.Conv2d(128,1,1,padding=0)

    def forward(self,x):

        score1_1 = x
        score2 = self.poola(score1_1)
        score3 = self.poolb(score1_1)

        #-------------Multiscale Feature Extraction--------------#
        #part 1
        score1_1 = self.encoder1(self.prerelu(self.prebn(self.preconv(score1_1))))
        score1_2 = self.encoder2(score1_1)
        score1_3 = self.encoder3(score1_2)
        score1_4 = self.encoder4(score1_3)

        score1_5 = self.poolc(score1_4)
        score1_5 = self.encoder5_1(score1_5)
        score1_5 = self.encoder5_2(score1_5)
        score1_5 = self.encoder5_3(score1_5)
        #part 2
        score2 = self.encoder1(self.prerelu(self.prebn(self.preconv(score2))))
        score2 = self.encoder2(score2)
        score2 = self.encoder3(score2)
        score2 = self.encoder4(score2)

        score2 = self.poolc(score2)
        score2 = self.encoder5_1(score2)
        score2 = self.encoder5_2(score2)
        score2 = self.encoder5_3(score2)
        #part 3
        score3 = self.encoder1(self.prerelu(self.prebn(self.preconv(score3))))
        score3 = self.encoder2(score3)
        score3 = self.encoder3(score3)
        score3 = self.encoder4(score3)

        score3 = self.poolc(score3)
        score3 = self.encoder5_1(score3)
        score3 = self.encoder5_2(score3)
        score3 = self.encoder5_3(score3)

        #-------------Cascaded feature fusion module--------------#
        score3 = self.cat_relu11(self.cat_bn11(self.cat_conv11(score3)))
        score3 = self.cat_relu12(self.cat_bn12(self.cat_conv12(score3)))
        score3 = self.cat_relu13(self.cat_bn13(self.cat_conv13(score3)))

        score3_up1 = self.upscore2(score3)

        score2 = score3_up1 + score2
        score2 = self.cat_relu21(self.cat_bn21(self.cat_conv21(score2)))
        score2 = self.cat_relu22(self.cat_bn22(self.cat_conv22(score2)))
        score2 = self.cat_relu23(self.cat_bn23(self.cat_conv23(score2)))

        score2_up = self.upscore2(score2)

        score3_up2 = self.upscore4(score3)

        score1_5 = score2_up + score3_up2 + score1_5
        score1_5 = self.cat_relu31(self.cat_bn31(self.cat_conv31(score1_5)))
        score1_5 = self.cat_relu32(self.cat_bn32(self.cat_conv32(score1_5)))
        score1_5 = self.cat_relu33(self.cat_bn33(self.cat_conv33(score1_5)))

        #-------------Edge--------------#
        score1_5_up = self.upscore8(score1_5)
        score1_5_up = self.up_bn(self.up_conv(score1_5_up))

        score_edge = score1_2 +score1_5_up
        score_edge = self.edge_relu(self.edge_bn(self.edge_conv(score_edge)))

        #-------------Bridge--------------#
        Fea2 = self.bdg_bn1(self.bdg_conv1(score1_5))
        Fea4 = self.bdg_bn2(self.bdg_conv2(score1_5))
        Fea8 = self.bdg_bn3(self.bdg_conv3(score1_5))
        Fea16 = self.bdg_bn4(self.bdg_conv4(score1_5))

        score_bdg = torch.cat((Fea2,Fea4,Fea8,Fea16,score1_5),1)

        #-------------deep feature aggregation module--------------#
        #stage 5
        t = self.pool5(score_edge)
        t = torch.cat((t,score_bdg),1)
        t = self.mrelu5(self.mbn5(self.mconv5(t)))

        scored5 = self.relud5_1(self.bnd5_1(self.convd5_1(t)))
        scored5 = self.relud5_2(self.bnd5_2(self.convd5_2(scored5)))
        scored5 = self.relud5_3(self.bnd5_3(self.convd5_3(scored5)))

        scored5_up = self.upscore2(scored5)
        #stage 4
        t = self.pool4(score_edge)
        t = torch.cat((t,score1_4),1)
        t = self.mrelu4(self.mbn4(self.mconv4(t)))

        scored4 = torch.cat((scored5_up,t),1)
        scored4 = self.relud4_1(self.bnd4_1(self.convd4_1(scored4)))
        scored4 = self.relud4_2(self.bnd4_2(self.convd4_2(scored4)))
        scored4 = self.relud4_3(self.bnd4_3(self.convd4_3(scored4)))

        scored4_up = self.upscore2(scored4)
        #stage 3
        t = self.pool3(score_edge)
        t = torch.cat((t,score1_3),1)
        t = self.mrelu3(self.mbn3(self.mconv3(t)))

        scored3 = torch.cat((scored4_up,t),1)
        scored3 = self.relud3_1(self.bnd3_1(self.convd3_1(scored3)))
        scored3 = self.relud3_2(self.bnd3_2(self.convd3_2(scored3)))
        scored3 = self.relud3_3(self.bnd3_3(self.convd3_3(scored3)))

        scored3_up = self.upscore2(scored3)
        #stage 2
        t = torch.cat((score1_2,score_edge),1)
        t = self.mrelu2(self.mbn2(self.mconv2(t)))

        scored2 = torch.cat((scored3_up,t),1)
        scored2 = self.relud2_1(self.bnd2_1(self.convd2_1(scored2)))
        scored2 = self.relud2_2(self.bnd2_2(self.convd2_2(scored2)))
        scored2 = self.relud2_3(self.bnd2_3(self.convd2_3(scored2)))

        scored2_up = self.upscore2(scored2)
        #stage 1
        t = self.up1(score_edge)
        t = torch.cat((score1_1,t),1)
        t = self.mrelu1(self.mbn1(self.mconv1(t)))

        scored1 = torch.cat((scored2_up,t),1)
        scored1 = self.relud1_1(self.bnd1_1(self.convd1_1(scored1)))
        scored1 = self.relud1_2(self.bnd1_2(self.convd1_2(scored1)))
        scored1 = self.relud1_3(self.bnd1_3(self.convd1_3(scored1)))

        #-------------Label output--------------#
        label1_out = self.label_outconv1(score3)
        label1_out = self.upscore64(label1_out)

        label2_out = self.label_outconv2(score2)
        label2_out = self.upscore32(label2_out)

        label3_out = self.label_outconv3(score1_5)
        label3_out = self.upscore16(label3_out)

        label4_out = self.label_outconv4(scored5)
        label4_out = self.upscore16(label4_out)

        label5_out = self.label_outconv5(scored4)
        label5_out = self.upscore8(label5_out)

        label6_out = self.label_outconv6(scored3)
        label6_out = self.upscore4(label6_out)

        label7_out = self.label_outconv7(scored2)
        label7_out = self.upscore2(label7_out)

        label8_out = self.label_outconv8(scored1)

        labelbdg_out = self.label_outconv9(score_bdg)
        labelbdg_out = self.upscore16(labelbdg_out)

        #-------------Edge output--------------#
        edge_out = self.edge_outconv1(score_edge)
        edge_out = self.upscore2(edge_out)

        #-------------Return--------------#
        return torch.sigmoid(edge_out), torch.sigmoid(label1_out), torch.sigmoid(label2_out), torch.sigmoid(label3_out), torch.sigmoid(labelbdg_out), \
        torch.sigmoid(label4_out), torch.sigmoid(label5_out), torch.sigmoid(label6_out), torch.sigmoid(label7_out), torch.sigmoid(label8_out)