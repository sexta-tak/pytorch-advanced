#%%
import os.path as osp
import random
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
from utils.ssd_model import make_datapath_list, VOCDataset, DataTransform, Anno_xml2list, od_collate_fn

rootpath = "./data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)
voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
color_mean = (104, 117, 123)
input_size = 300

train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))
val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

batch_size = 32
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn)
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

#%%
from utils.ssd_model import SSD, MultiBoxLoss
ssd_cfg = {
    'num_classes': 21,  # 背景クラスを含めた合計クラス数
    'input_size': 300,  # 画像の入力サイズ
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
    'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

net = SSD(phase="train", cfg=ssd_cfg)

vgg_weights = torch.load("./weights/vgg16_reducedfc.pth")
net.vgg.load_state_dict(vgg_weights)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

# %%
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    torch.backends.cudnn.benchmark = True

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    for epoch in range(num_epochs+1):
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print("---------------")
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("---------------")

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                if((epoch+1) % 10 == 0):
                    net.eval()
                    print("---------------")
                    print(" (val) ")
                else:
                    continue

            for images, targets in tqdm(dataloaders_dict[phase]):
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(images)

                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == "train":
                        loss.backward()
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
                        optimizer.step()

                        if (iteration % 10 == 0):
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print("iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec.".format(iteration, loss.item(), duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1

                    else:
                        epoch_val_loss += loss.item()

            t_epoch_finish = time.time()
            print("---------------")
            print("epoch {} || Epoch_Train_Loss:{:.4f} || Epoch_Val_Loss:{:.4f}".format(epoch+1, epoch_train_loss, epoch_val_loss))
            print("timer: {:.4f} sec.".format(t_epoch_finish - t_epoch_start))
            t_epoch_start = time.time()

            log_epoch = {"epoch": epoch+1, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss}
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv("log_output.csv")

            epoch_train_loss = 0.0
            epoch_val_loss = 0.0

            if ((epoch+1) % 50 == 0):
                torch.save(net.state_dict(), "weights/ssd300_" + str(epoch+1) + ".pth")

num_epochs=250
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
# %%
