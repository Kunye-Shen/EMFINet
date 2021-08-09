import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import glob

from Data_loader import RescaleT
from Data_loader import ToTensorLab
from Data_loader import SalObjDataset

from model import EMFINet

import pytorch_ssim
import pytorch_iou

bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def muti_loss(pred,target):

    bce_out = bce_loss(pred,target)
    ssim_out = 1 - ssim_loss(pred,target)
    iou_out = iou_loss(pred,target)

    loss = bce_out + ssim_out + iou_out

    return loss

def muti_loss_fusion(d1,d2,d3,d4,d5,d6,d7,d8,d9, labels_v):

    loss1 = muti_loss(d1,labels_v)
    loss2 = muti_loss(d2,labels_v)
    loss3 = muti_loss(d3,labels_v)
    loss4 = muti_loss(d4,labels_v)
    loss5 = muti_loss(d5,labels_v)
    loss6 = muti_loss(d6,labels_v)
    loss7 = muti_loss(d7,labels_v)
    loss8 = muti_loss(d8,labels_v)
    loss9 = muti_loss(d9,labels_v)
    print("l1: %.3f, l2: %.3f, l3: %.3f, l4: %.3f, l5: %.3f, l6: %.3f, l7: %.3f , l8:%.3f , l9:%.3f"%(loss1.item(),loss2.item(),\
        loss3.item(),loss4.item(),loss5.item(),loss6.item(),loss7.item(),loss8.item(),loss9.item()))

    loss_label = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 +loss7 + loss8 + loss9

    return loss_label

tra_image_dir = "/train_images/"
tra_label_dir = "/train_labels/"
tra_edge_dir= "/train_edges/"

image_ext = '.jpg'
label_ext = '.png'
edge_ext = '.png'

model_dir = "/model_save/"

epoch_num = 1000
batch_size_train = 4
train_num = 0

tra_img_name_list = glob.glob(tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
tra_edge_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split("/")[-1]

    a = img_name.split(".")
    b = a[0:-1]
    imidx = b[0]
    for i in range(1,len(b)):
        imidx = imidx + "." + b[i]

    tra_lbl_name_list.append(tra_label_dir + imidx + label_ext)
    tra_edge_name_list.append(tra_edge_dir + imidx + edge_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("train edges:  ",len(tra_edge_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    edge_name_list=tra_edge_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

net = EMFINet()
if torch.cuda.is_available():
    net.cuda()

print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

def main():
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    ite_num4val = 0

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels, edges = data['image'], data['label'], data['edge']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            edges = edges.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)
                edges_v = Variable(edges.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
                edges_v = Variable(edges, requires_grad=False)

            optimizer.zero_grad()

            de,d1,d2,d3,d4,d5,d6,d7,d8,d9 = net(inputs_v)
            loss_label = muti_loss_fusion(d1,d2,d3,d4,d5,d6,d7,d8,d9, labels_v)
            loss_edge = bce_loss(de,edges_v)
            print("l1_e: %.3f"%(loss_edge.item()))
            loss = loss_label + loss_edge
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            del de,d1,d2,d3,d4,d5,d6,d7,d8,d9,loss,loss_label,loss_edge
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %.3f \n" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val))

            if ite_num % 1000 == 0:

                torch.save(net.state_dict(), model_dir + "MYNet_%d_%d.pth" % (ite_num, epoch))
                running_loss = 0.0
                net.train()
                ite_num4val = 0
    print('-------------Congratulations! Training Done!!!-------------')
if __name__ == '__main__':
    main()