#%%
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class Generator(nn.Module):
    def __init__(self, z_dim=20, image_size=64):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, image_size*8, kernel_size=4, stride=1),
            nn.BatchNorm2d(image_size*8),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(image_size*8, image_size*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size*4),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(image_size*4, image_size*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(image_size*2, image_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True),
        )

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out

class Discriminator(nn.Module):
    def __init__(self, z_dim=20, image_size=64):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, image_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(image_size, image_size*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(image_size*2, image_size*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(image_size*4, image_size*8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.last = nn.Conv2d(image_size*8, 1, kernel_size=4, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out

def make_datapath_list():
    train_img_list = list()
    for img_idx in range(200):
        img_path = "./data/img_78/img_7_" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)

        img_path = "./data/img_78/img_8_" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)

    return train_img_list

class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __call__(self, img):
        return self.data_transform(img)


class GAN_Img_Dataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        
        if self.transform is not None:
            img = self.transform(img)

        return img


train_img_list = make_datapath_list()

mean, std = (0.5,), (0.5,)
train_dataset = GAN_Img_Dataset(file_list=train_img_list, transform=ImageTransform(mean, std))

batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

    elif classname.find("BatchNrom") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

G = Generator()
G.apply(weights_init)
D = Discriminator()
D.apply(weights_init)

def train_model(G, D, dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    g_lr, d_lr = 1e-4, 4e-4
    beta1, beta2 = 0.0, 0.9
    g_optimizer = optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = optim.Adam(D.parameters(), d_lr, [beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    z_dim = 20
    mini_batch_size = 64

    G.to(device)
    D.to(device)

    G.train()
    D.train()

    torch.backends.cudnn.benchmark = True
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.bath_size

    iteration = 1
    logs = []

    for epochs in range(num_epochs):
        t_epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        print("Epoch {}/{}".format(epochs, num_epochs))
        print("train")

        for imgs in tqdm(dataloader):
        """Discriminatorの学習"""
        imgs = imgs.to(device)
        mini_batch_size = imgs.size()[0]
        label_real = torch.full((mini_batch_size,), 1).to(device)
        label_fake = torch.full((mini_batch_size,), 0).to(device)

        d_out_real = D(imgs)

        input_z = torch.randn(mini_batch_size, z_dim).to(device)
        input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
        fake_imgs = G(input_z)
        d_out_fake = D(fake_imgs)

        d_loss_real = criterion(d_out_real.view(-1), label_real)
        d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
        d_loss = d_loss_real + d_loss_fake

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        d_loss.backward()
        d_optimizer.step()

        """Generatorの学習"""
        input_z = torch.randn(mini_batch_size, z_dim).to(device)
        input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
        fake_imgs  G(input_z)
        d_out_fake = D(fake_imgs)

        g_loss = criterion(d_out_fake.view(-1), label_real)

        g_optimzier.zero_grad()
        d_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        iteration += 1

    t_epoch_finish = time.time()
    print("epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}".format(epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size))
    print("timer:  {:/4f} sec.".format(t_epoch_finish - t_epoch_start))
    t_epoch_start = time.time()

return G, D


num_epochs = 200
G_update, D_update = train_model(G, D, dataloader=train_dataloader, num_epochs=num_epochs)