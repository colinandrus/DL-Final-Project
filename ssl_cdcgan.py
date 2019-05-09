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
from torch.autograd import Variable

# G(z)
latent_dim = 100
ngf = 128
ndf = 128
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

print("defined stuff")

latent_dim = 100
num_classes = 1000
# fixed noise & label
'''
temp_z_ = torch.randn(num_classes, latent_dim)
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(num_classes, 1)
for i in range(num_classes-1):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(num_classes, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)

print("fz shaped: ", fixed_z_.shape)
print("fy shaped: ", fixed_y_.shape)
fixed_z_ = fixed_z_.view(-1, latent_dim, 1, 1)
print("after view fz shaped: ", fixed_z_.shape)
fixed_y_label_ = torch.zeros(num_classes**2, num_classes)
fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
fixed_y_label_ = fixed_y_label_.view(-1, num_classes, 1, 1)
fixed_z_, fixed_y_label_ = Variable(fixed_z_.cuda(), volatile=True), Variable(fixed_y_label_.cuda(), volatile=True)
def show_result(num_epoch, show = False, save = False, path = 'result.png'):

    G.eval()
    test_images = G(fixed_z_, fixed_y_label_)
    G.train()

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
'''

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters
batch_size = 8
lr = 0.0002
train_epoch = 20
ngf = 4
ndf = 4

# data_loader
img_size = 64
transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder("/home/thekingh/final_projects/gans/dl_data/supervised/train", transform=transform),
    batch_size=batch_size,
    shuffle=True,
)
print("loaded data, batch len=", len(train_loader))

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
root = 'ssLresults/'
model = 'SSL_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

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

print('>training start')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    if (epoch+1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print(">learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print(">learning rate change!")

    epoch_start_time = time.time()
    y_real_ = torch.ones(batch_size)
    y_fake_ = torch.zeros(batch_size)
    y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
    for x_, y_ in train_loader:
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
        D_real_los = BCE_loss(D_result, y_real_)

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
torch.save(G.state_dict(), 'g.pth')
torch.save(D.state_dict(), 'd.pth')
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)
