from ..core.utils import *
from ..core.const import *
from ..data import *


class DCGAN():
    def __init__(self, dataset,\
    noise_size, noise_type, device='auto', \
    conv_dim=(32,32), \
    num_conv=(3,3), \
    label_smooth=(0.9, 0.3), \
    learning_rate=(0.001, 0.0001), \
    beta1=(0.5, 0.5), \
    beta2=(0.999, 0.999),\
    train_ratio=(1,1),\
    w_loss=False,
    ):
        self.dataset = dataset
        self.label_smooth=label_smooth
        self.noise_type=noise_type
        self.noise_size=noise_size
        self.num_img_channels = self.dataset.num_output_channels
        self.img_size = self.dataset.img_size
        self.num_img_channels = self.dataset.num_output_channels
        self.learning_rate = learning_rate
        self.beta1=beta1
        self.beta2=beta2
        self.train_ratio=train_ratio
        self.w_loss=w_loss
        self.discriminator = DCDiscriminator(input_img_size=self.img_size, num_input_channels= self.num_img_channels, conv_dim=conv_dim[0], num_conv=num_conv[0])
        self.generator = DCGenerator(noise_size=noise_size, conv_dim=conv_dim[1], output_img_size=self.img_size, num_output_channels= self.num_img_channels, num_tconv=num_conv[1])
        self.device = select_device(device)
        if self.w_loss:
          self.discriminator_optimizer=optim.RMSprop(self.discriminator.parameters(), lr=self.learning_rate[0])
          self.generator_optimizer=optim.RMSprop(self.generator.parameters(), lr=self.learning_rate[1])
        else:
          self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), self.learning_rate[0], [self.beta1[0], self.beta2[0]])
          self.generator_optimizer = optim.Adam(self.generator.parameters(), self.learning_rate[1], [self.beta1[1], self.beta2[1]])

    def generate_noise(self, size, type, batch_size):
        if type == 'normal': return torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size, size))).float()
        elif type == 'gaussian': return torch.from_numpy(np.random.normal(0, 1, size=(batch_size, size))).float()
        else: raise ValueError('Noise type not specified/recognized. Please check.')

    def generate_fixed_samples(self, size):
        return self.generate_noise(size=self.noise_size, type=self.noise_type, batch_size=self.dataset.batch_size)

    def calculate_real_loss(self, X, device, label_smooth=False, w_loss=False):
        batch_size = X.size(0)
        if label_smooth:
            labels = torch.ones(batch_size)*label_smooth
        elif w_loss:
            labels = torch.ones(batch_size)*-1
        else:
            labels = torch.ones(batch_size)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(X.squeeze(), labels.to(device))
        return loss

    def calculate_fake_loss(self, X, device, label_smooth=False, w_loss=False):
        batch_size = X.size(0)
        if label_smooth:
            labels = torch.ones(batch_size)*label_smooth
        elif w_loss:
            labels = torch.ones(batch_size)
        else:
            labels=torch.zeros(batch_size)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(X.squeeze(), labels.to(device))
        return loss

    def rescale_images(self, X, range=(-1,1)):
        min, max = range
        X = X * (max - min) + min
        return X

    def fit(self, epochs=10):
        self.samples = []
        total_d_loss = []
        d_real_loss = []
        d_fake_loss = []
        total_g_loss = []
        fixed_noise = (self.generate_fixed_samples(16)).to(self.device)

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        self.generator.train()
        self.discriminator.train()

        for e in tqdm(range(epochs), total=epochs):

            discriminator_loss = 0
            generator_loss = 0
            discriminator_loss_real = 0
            discriminator_loss_fake= 0

            for i, (idx, real_images, labels) in enumerate(self.dataset.loaders['train']):
                real_images = self.rescale_images(real_images)
                real_images = real_images.to(self.device)

                for r in range(self.train_ratio[0]):
                    # Train Discriminator on Real Images
                    self.discriminator_optimizer.zero_grad()

                    real_logits = self.discriminator(real_images)
                    real_loss = self.calculate_real_loss(real_logits, label_smooth=self.label_smooth[0], device=self.device, w_loss=self.w_loss)

                    # Train Discriminator on Real Images
                    noise = self.generate_noise(self.noise_size, self.noise_type, self.dataset.batch_size)
                    fake_images = self.generator(noise.to(self.device))
                    fake_logits = self.discriminator(fake_images)
                    fake_loss = self.calculate_fake_loss(fake_logits,label_smooth=self.label_smooth[1], device=self.device, w_loss=self.w_loss)

                    # Total Discriminator loss
                    d_loss = real_loss + fake_loss
                    if self.w_loss:
                        f = torch.mean(fake_logits.view(-1))
                        r = torch.mean(real_logits.view(-1))
                        d_loss = -(r-f)
                    d_loss.backward()
                    self.discriminator_optimizer.step()

                    if self.w_loss:
                        for p in self.discriminator.parameters(): p.data.clamp_(-0.01, 0.01)


                for r in range(self.train_ratio[1]):
                # Train Generator
                    self.generator_optimizer.zero_grad()
                    g_fake_images = self.generator(noise.to(self.device))
                    g_fake_logits = self.discriminator(g_fake_images)
                    g_loss = self.calculate_real_loss(g_fake_logits, device=self.device, w_loss=self.w_loss) #flipped labels
                    if self.w_loss:
                        g_loss = -torch.mean(g_fake_logits.view(-1))

                    g_loss.backward()
                    self.generator_optimizer.step()

                # Keep track of losses
                discriminator_loss += d_loss.item()
                generator_loss += g_loss.item()
                discriminator_loss_real += real_loss.item()
                discriminator_loss_fake += fake_loss.item()

            epoch_d_loss = discriminator_loss/len(self.dataset.loaders['train'])
            epoch_g_loss = generator_loss/len(self.dataset.loaders['train'])
            epoch_d_real_loss = discriminator_loss_real/len(self.dataset.loaders['train'])
            epoch_d_fake_loss = discriminator_loss_fake/len(self.dataset.loaders['train'])

            total_d_loss.append(epoch_d_loss)
            total_g_loss.append(epoch_g_loss)
            d_real_loss.append(epoch_d_real_loss)
            d_fake_loss.append(epoch_d_fake_loss)

            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.5f} | d_loss(real): {:6.5f} | d_loss(fake): {:6.5f} | g_loss: {:6.5f}'.format(
                    e+1, epochs, epoch_d_loss, epoch_d_real_loss, epoch_d_fake_loss, epoch_g_loss))

            # Generate samples to view later
            with torch.no_grad():
                self.generator.eval()
                sample_images = self.generator(fixed_noise)
                self.samples.append(sample_images)
                self.generator.train()

        self.train_logs=pd.DataFrame({"d_loss": total_d_loss, "d_loss_real":d_real_loss, "d_loss_fake":d_fake_loss,  "g_loss" : total_g_loss})

    def view_samples(self, epoch=-1, cmap='gray', num_images=8, figsize=(16,4)):
        if isinstance(epoch, int):
            fig, axes = plt.subplots(figsize=figsize, nrows=2, ncols=8, sharey=True, sharex=True)
            for ax, img in zip(axes.flatten(), self.samples[epoch]):
                img = img.detach().cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                im = ax.imshow(img.reshape((self.img_size,self.img_size,self.num_img_channels)), cmap=cmap)
        else:
          fig, axes = plt.subplots(figsize=figsize, nrows=len(epoch), ncols=8, sharey=True, sharex=True)
          imgs = torch.cat([self.samples[i][:num_images] for i in epoch], 0)
          for ax, img in zip(axes.flatten(), imgs):
              img = img.detach().cpu().numpy()
              img = np.transpose(img, (1, 2, 0))
              ax.xaxis.set_visible(False)
              ax.yaxis.set_visible(False)
              im = ax.imshow(img.reshape((self.img_size,self.img_size,self.num_img_channels)), cmap=cmap)


    def view_train_logs(self, data='all', figsize=(12,8)):
        plt.figure(figsize=figsize)
        sns.set_style("darkgrid")
        if data == 'all': p = sns.lineplot(data = self.train_logs)
        else: p = sns.lineplot(data = self.train_logs[data].tolist())
        p.set_xlabel("epoch", fontsize = 10)
        p.set_ylabel("loss", fontsize = 10);


class DCDiscriminator(nn.Module):
    def __init__(self, input_img_size, num_input_channels=3, conv_dim=32, num_conv=3):
        super(DCDiscriminator, self).__init__()
        self.input_img_size = input_img_size
        self.num_conv = num_conv
        self.conv_dim = conv_dim
        self.num_input_channels = num_input_channels
        self.output_img_size = self.get_end_img_size(self.input_img_size, self.num_conv)

        layers = [('conv0', self.conv_unit(in_channels=self.num_input_channels, out_channels=self.conv_dim, kernel_size=4, batch_norm=False, leaky_relu=True, dropout=0.5))]

        for i in range(self.num_conv-1):
            layers.append(('conv'+str(i+1), self.conv_unit(in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=4, batch_norm=True, leaky_relu=True, dropout=0.5)))
            conv_dim = conv_dim*2
        self.conv_layers = nn.Sequential(OrderedDict(layers))
        self.fc = nn.Linear(in_features=(self.conv_dim*(2**(self.num_conv-1))*self.output_img_size*self.output_img_size), out_features=1)

    def get_end_img_size(self, input_img_size, num_conv):
        for i in range(num_conv):
            input_img_size = input_img_size/2
            if input_img_size < 4:
                raise ValueError('Cannot use this number of conv operations. Min conv final image size must be at least 4x4. Please use a smaller number of conv.')
        return int(input_img_size)

    def conv_unit(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, leaky_relu=True, dropout=False):
        layers = []
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        layers.append(conv_layer)
        if batch_norm:layers.append(nn.BatchNorm2d(out_channels))
        if leaky_relu:layers.append(nn.LeakyReLU())
        if dropout:layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.conv_dim*(2**(self.num_conv-1))*self.output_img_size*self.output_img_size)
        x = self.fc(x)
        return x


class DCGenerator(nn.Module):
    def __init__(self, noise_size, conv_dim, output_img_size, num_output_channels, num_tconv):
        super(DCGenerator, self).__init__()
        self.noise_size=noise_size
        self.conv_dim=conv_dim
        self.output_img_size=output_img_size
        self.num_output_channels=num_output_channels
        self.num_tconv=num_tconv
        self.start_img_size = self.get_start_img_size(self.output_img_size, self.num_tconv)

        self.fc = nn.Linear(in_features=self.noise_size, out_features=(self.conv_dim * (2**(self.num_tconv-1)) * self.start_img_size * self.start_img_size))
        self.t_conv_layers = self.create_tconv_seq()

    def get_start_img_size(self, target_size, num_conv):
        for i in range(num_conv):
            target_size = target_size/2
            if target_size < 4:
                raise ValueError('Cannot use this number of t_conv operations. Starting image size must be at least 4x4. Please use a smaller number of t_conv.')
        return int(target_size)

    def tconv_unit(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, relu=True):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if batch_norm: layers.append(nn.BatchNorm2d(out_channels))
        if relu: layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def create_tconv_seq(self):
        layers=[]
        z = 2**(self.num_tconv-1)
        for i in range (self.num_tconv-1):
            layers.append(('t_conv'+str(i), self.tconv_unit(in_channels=self.conv_dim*z, out_channels=self.conv_dim*int(z/2))))
            z = int(z/2)
        layers.append(('t_conv'+str(self.num_tconv-1), self.tconv_unit(in_channels=self.conv_dim, out_channels=self.num_output_channels, batch_norm=False, relu=False)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1,self.conv_dim * (2**(self.num_tconv-1)) , self.start_img_size , self.start_img_size)
        x = self.t_conv_layers(x)
        x = torch.tanh(x)
        return x
