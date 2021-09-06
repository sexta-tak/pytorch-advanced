#%%
import os.path as osp
from PIL import Image
import torch.utils.data as data

# %%
def make_datapath_list(rootpath):
    imgpath_template = osp.join(rootpath, "JPEGImages", "%s.jpg")
    annopath_template = osp.join(rootpath, "SegmentationClass", "%s.png")

    train_id_names = osp.join(rootpath + "ImageSets/Segmentation/train.txt")
    val_id_names = osp.join(rootpath + "ImageSets/Segmentation/val.txt")

    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id =line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list

rootpath = "./data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

# print(train_img_list[0])
# print(train_anno_list[0])

# %%
from utils.data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor

class DataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            "train":Compose([
                Scale(scale=[0.5, 1.5]),
                RandomRotation(angle=[-10, 10]),
                RandomMirror(),
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ]),
            "val":Compose([
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)

class VOCDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)

        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img

color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))
val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

# print(val_dataset.__getitem__(0)[0].shape)
# print(val_dataset.__getitem__(0)[1].shape)
# print(val_dataset.__getitem__(0))

# %%
batch_size = 8
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

iterator = iter(dataloaders_dict["val"])
imges, anno_class_imges = next(iterator)
print(imges.size())
print(anno_class_imges.size())

# %%
import numpy as np
import matplotlib.pyplot as plt

index = 0
imges, anno_class_imges = train_dataset.__getitem__(index)

img_val = imges
img_val = img_val.numpy().transpose((1, 2, 0))
plt.imshow(img_val)
plt.show()

anno_file_path = train_anno_list[0]
anno_class_img = Image.open(anno_file_path)
p_palette = anno_class_img.getpalette()

anno_class_img_val = anno_class_imges.numpy()
anno_class_img_val = Image.fromarray(np.uint8(anno_class_img_val), mode="P")
anno_class_img_val.putpalette(p_palette)
plt.imshow(anno_class_img_val)
plt.show()

# %%
index = 0
imges, anno_class_imges = val_dataset.__getitem__(index)

img_val = imges
img_val = img_val.numpy().transpose((1, 2, 0))
plt.imshow(img_val)
plt.show()

anno_file_path = train_anno_list[0]
anno_class_img = Image.open(anno_file_path)
p_palette = anno_class_img.getpalette()

anno_class_img_val = anno_class_imges.numpy()
anno_class_img_val = Image.fromarray(np.uint8(anno_class_img_val), mode="P")
anno_class_img_val.putpalette(p_palette)
plt.imshow(anno_class_img_val)
plt.show()
# %%
