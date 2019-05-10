import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

# G(z)
latent_dim = 100
ngf = 32
ndf = 32
num_channels = 3
num_classes = 1000
class generator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(generator, self).__init__()
        self.deconv1_1    = nn.ConvTranspose2d(latent_dim, ngf*4, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(ngf*4)
        self.deconv1_2    = nn.ConvTranspose2d(num_classes, ngf*4, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(ngf*4)
        self.deconv2      = nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1)
        self.deconv2_bn   = nn.BatchNorm2d(ngf*4)
        self.deconv3      = nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1)
        self.deconv3_bn   = nn.BatchNorm2d(ngf*2)
        self.deconv4      = nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1)
        self.deconv4_bn   = nn.BatchNorm2d(ngf)
        self.deconv5      = nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.deconv1_1_bn(self.deconv1_1(input)), 0.2)
        y = F.leaky_relu(self.deconv1_2_bn(self.deconv1_2(label)), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = torch.tanh(self.deconv5(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(num_channels, int(ndf/2), 4, 2, 1)
        self.conv1_2 = nn.Conv2d(num_classes, int(ndf/2), 4, 2, 1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(ndf*2)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(ndf*4)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(ndf*8)
        self.conv5 = nn.Conv2d(ndf*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# program params
MODEL_CHECKPOINT = 2
LEARNING_RATE_UPDATE = 8
IMAGE_SAVE_INTERVAL = 1000

# training parameters
batch_size = 16
lr = 0.0002
train_epoch = 30 
ngf = 32
ndf = 32
latent_dim = 100
num_classes = 1000

# data_loader
img_size = 64
transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

data_root = "/scratch/ks4883/dl_data/supervised/train"
raw_data = datasets.ImageFolder(data_root, transform=transform)
classes, classes_to_idx = raw_data._find_classes(data_root)
idx_to_class = {}
for class_name in classes_to_idx:
    index = classes_to_idx[class_name]
    idx_to_class[index] = class_name
print("num classes: ", len(classes))

train_loader = torch.utils.data.DataLoader(
    raw_data,
    batch_size=batch_size,
    shuffle=True,
)
print("loaded data, num batches=", len(train_loader), ", batch_size=", batch_size)

# network
print("about to make nets")
G = generator(ngf)
D = discriminator(ndf)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()
print("moved nets to cuda")

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
root = 'gen_img'
model = 'SSL_'

if not os.path.isdir(root):
    os.mkdir(root)

model_dir = 'saved_models'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# label preprocess
num_classes = 1000
latent_dim = 100
onehot = torch.zeros(num_classes, num_classes)
onehot = onehot.scatter_(1, torch.LongTensor(list(range(num_classes))).view(num_classes,1), 1).view(num_classes, num_classes, 1, 1)
fill = torch.zeros([num_classes, num_classes, img_size, img_size])
for i in range(num_classes):
    fill[i, i, :, :] = 1

def generate_images(epoch, num_batches):
    
    # generate images in batches of size 10
    gen_batch_size = 10
    batch_first_label = 0
    batch_last_label = batch_first_label + gen_batch_size

    for i in range(100):
        fake_ys = torch.LongTensor(list(range(batch_first_label, batch_last_label))).squeeze()
        fake_y_labels = onehot[fake_ys]
        fake_y_labels = Variable(fake_y_labels.cuda())
        z = torch.randn((gen_batch_size, latent_dim)).view(-1, latent_dim, 1, 1)
        z = Variable(z.cuda())
        
        generated_imgs = G(z, fake_y_labels)
        
        for num_label, img in zip(fake_ys, generated_imgs):
            class_label = idx_to_class[num_label.item()]

            root = "gen_img/"
            if not os.path.exists(root+class_label):
                os.mkdir(root+class_label)

            save_image(img.data, "gen_img/{0}/{0}_{1}_{2}.png".format(class_label, epoch, num_batches))

        batch_first_label += gen_batch_size
        batch_last_label += gen_batch_size

print('>training start')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    if epoch >= 10 and epoch % LEARNING_RATE_UPDATE == 0:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print(">learning rate change!")

    # model checkpointing
    if epoch > 0 and epoch % MODEL_CHECKPOINT == 0:
        os.mkdir(model_dir + '/{0}'.format(epoch))
        torch.save(G.state_dict(), model_dir+'/e{0}/g{0}.pth'.format(epoch))
        torch.save(D.state_dict(), model_dir+'/e{0}/d{0}.pth'.format(epoch))

    num_batches = 0
    epoch_start_time = time.time()
    y_real_ = torch.ones(batch_size)
    y_fake_ = torch.zeros(batch_size)
    y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())

    for x_, y_ in train_loader:
        num_batches += 1
        # train discriminator D
        D.zero_grad()

        mini_batch = x_.size()[0]

        if mini_batch != batch_size:
            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)
            y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())

        y_fill_ = fill[y_]
        x_, y_fill_ = Variable(x_.cuda()), Variable(y_fill_.cuda())

        D_result = D(x_, y_fill_).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, latent_dim)).view(-1, latent_dim, 1, 1)
        y_ = (torch.rand(mini_batch, 1) * num_classes).type(torch.LongTensor).squeeze()
        y_label_ = onehot[y_]
        y_fill_ = fill[y_]
        z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(y_label_.cuda()), Variable(y_fill_.cuda())

        G_result = G(z_, y_label_)
        D_result = D(G_result, y_fill_).squeeze()

        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.data.item())

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, latent_dim)).view(-1, latent_dim, 1, 1)
        y_ = (torch.rand(mini_batch, 1) * num_classes).type(torch.LongTensor).squeeze()
        y_label_ = onehot[y_]
        y_fill_ = fill[y_]
        z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(y_label_.cuda()), Variable(y_fill_.cuda())

        G_result = G(z_, y_label_)
        D_result = D(G_result, y_fill_).squeeze()

        G_train_loss = BCE_loss(D_result, y_real_)

        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data.item())

        if num_batches % IMAGE_SAVE_INTERVAL == 0:
            generate_images(epoch, num_batches)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    #fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    #show_result((epoch+1), save=True, path=fixed_p)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))

print("Training finish!... save training results")
torch.save(G.state_dict(), 'g_final.pth')
torch.save(D.state_dict(), 'd_final.pth')

with open('train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

'''
show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)
'''
