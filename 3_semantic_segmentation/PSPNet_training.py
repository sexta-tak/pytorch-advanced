#%%
import random
import math
import time
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

#%%
from utils.dataloader import make_datapath_list, DataTransform, VOCDataset

rootpath = "./data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath=rootpath)

color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))
val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

batch_size = 8
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# %%
from utils.pspnet import PSPNet

net = PSPNet(n_classes=21)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

net.apply(weights_init)
print(net)

# %%
class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight

    def forward(self, outputs, targets):
        loss = F.cross_entropy(outputs[0], targets, reduction="mean")
        loss_aux = F.cross_entropy(outputs[1], targets, reduction="mean")

        return loss+self.aux_weight*loss_aux

criterion = PSPLoss(aux_weight=0.4)
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0001)

def lambda_epoch(epoch):
    max_epoch = 150
    return math.pow((1-epoch/max_epoch), 0.9)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

# %%
def train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    net.to(device)
    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloaders_dict["train"].dataset)
    num_val_imgs = len(dataloaders_dict["val"].dataset)
    batch_size = dataloaders_dict["train"].batch_size

    iteration = 1
    logs = []
    batch_multiplier = 3

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                scheduler.step()
                optimizer.zero_grad()
                print(phase)

            else:
                if((epoch+1)%5==0):
                    net.eval()
                    print(phase)
                else:
                    continue

            count = 0
            for imges, anno_class_imges in tqdm(dataloaders_dict[phase]):
                if imges.size()[0] == 1:
                    continue

                imges = imges.to(device)
                anno_class_imges = anno_class_imges.to(device)

                if (phase == "train") and (count == 0):
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(imges)
                    loss = criterion(outputs, anno_class_imges.long()) / batch_multiplier

                    if phase == "train":
                        loss.backward()
                        count -= 1
                        epoch_train_loss += loss.item() * batch_multiplier
                        iteration += 1

                    else:
                        epoch_val_loss += loss.item() * batch_multiplier

        t_epoch_finish = time.time()
        duration = t_epoch_finish - t_epoch_start
        print("Epoch: {}/{} || Train_Loss: {:.4f} || Valid_Loss: {:.4f} || Timr: {:.4f} sec.".format(
            epoch+1,
            num_epochs,
            epoch_train_loss/num_train_imgs,
            epoch_val_loss/num_val_imgs,
            duration
            ))
        t_epoch_start = time.time()

        log_epoch = {"epoch": epoch+1, "train_loss": epoch_train_loss/num_train_imgs, "val_loss": epoch_val_loss/num_val_imgs}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

        if (epoch+1)%30 == 0:
            torch.save(net.state_dict(), "weights/pspnet50_" + str(epoch+1) + ".pth")



num_epochs = 150
train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs=num_epochs)

# %%
