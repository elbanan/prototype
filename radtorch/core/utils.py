import torch, torchvision, itertools, glob, os, pydicom, copy, cv2, uuid

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import torchvision.transforms as transforms
import torchvision.models as models

from statistics import mean
from copy import deepcopy
from PIL import Image
from sklearn import metrics
from tqdm.notebook import tqdm
from pathlib import Path
from torch.utils.data.dataset import Dataset
from collections import OrderedDict
from sklearn.utils import resample
from torch.utils.model_zoo import load_url
from torchinfo import summary
from matplotlib import cm
from datetime import datetime
from torchvision.utils import make_grid

from .const import *


###### GENERAL ######

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def select_device(device='auto'):
    if device=='auto':
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device)

def current_time():
    dt_string = (datetime.now()).strftime("[%d-%m-%Y %H:%M:%S]")
    return dt_string

def set_random_seed(seed):
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        log('Random seed '+str(seed)+' set successfully')
    except:
        raise TypeError('Error. Could not set Random Seed. Please check again.')
        pass

def save_checkpoint(classifier, output_file):
    if classifier.classifier_type == 'torch':
        checkpoint = {'type':classifier.classifier_type,
                      'model':classifier.best_model,
                      'optimizer_state_dict' : classifier.optimizer.state_dict(),
                      'train_losses': classifier.train_losses,
                      'valid_losses': classifier.valid_losses,
                      'valid_loss_min': classifier.valid_loss_min,}

    elif classifier.classifier_type == 'sklearn':
        checkpoint = {'type':classifier.classifier_type,
                      'model':classifier.best_model}

    torch.save(checkpoint, output_file)


###### GRAPH ######

def show_values_on_bars(axs):
    #https://stackoverflow.com/a/51535326
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = int(p.get_height())
            ax.text(_x, _y, value, ha="center")
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def show_stack(imgs, figsize):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False,figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

###### DATA ######

def find_classes(directory):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def root_to_data(root, path_col, label_col, ext):
    df = pd.DataFrame()
    list_path = []
    list_label =[]
    for file in glob.glob(root + "**/*."+ext, recursive=True):
        list_path.append(file)
        list_label.append(Path(os.path.join(root, file)).parent.name)
    df[path_col] = list_path
    df[label_col] = list_label
    return df

def id_to_path(df, root, id_col, path_col = 'img_path', ext='.dcm'):
    if root[-1] != '/' : root = root+'/'
    df[path_col] = [root+r[id_col]+ext for i,r in df.iterrows()]
    return df

def split_data(table, valid_percent=False, test_percent=False):
    num_train = len(table)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    if valid_percent:
        valid_size = int(np.floor(valid_percent * num_train))
        if test_percent:
            test_size = int(np.floor(test_percent * num_train))
            test_idx, valid_idx, train_idx = indices[:test_size], indices[test_size:test_size+valid_size],indices[test_size+valid_size:]
            return test_idx, valid_idx, train_idx
        else:
            split = int(np.floor(valid_percent * num_train))
            train_idx, valid_idx = indices[split:], indices[:split]
            return train_idx, valid_idx
    else:
        raise ValueError('Split percentages not provided. Please check.')

def check_zero_image(table, path_col):
    zero_img = []
    for i, r in table.iterrows():
        if np.max(pydicom.read_file(r[path_col]).pixel_array) == 0:
            zero_img.append(True)
        else:
            zero_img.append(False)
    table['zero_img'] = zero_img
    return table[table['zero_img']==False]

def balance(df, method, **kwargs):
    counts=df.groupby(kwargs['label_col']).count()
    classes=kwargs['classes']
    max_class_num=counts.max()[0]
    max_class_id=counts.idxmax()[0]
    min_class_num=counts.min()[0]
    min_class_id=counts.idxmin()[0]
    if method=='upsample':
        resampled_subsets = [df[df[kwargs['label_col']]==max_class_id]]
        for i in [x for x in classes if x != max_class_id]:
          class_subset=df[df[kwargs['label_col']]==i]
          upsampled_subset=resample(class_subset, n_samples=max_class_num, random_state=100)
          resampled_subsets.append(upsampled_subset)
    elif method=='downsample':
        resampled_subsets = [df[df[kwargs['label_col']]==min_class_id]]
        for i in [x for x in classes if x != min_class_id]:
          class_subset=df[df[kwargs['label_col']]==i]
          upsampled_subset=resample(class_subset, n_samples=min_class_num, random_state=100)
          resampled_subsets.append(upsampled_subset)
    resampled_df = pd.concat(resampled_subsets)
    return resampled_df

def dict_to_data(table_dict, **kwargs):
    data_table = {}
    classes = []
    for i in ['train', 'valid', 'test']:
        if i not in table_dict: print ('Warning:', i, 'label table was not provided. Skipping.')
        else:
            data_table[i] = table_dict[i]
            classes += table_dict[i][kwargs['label_col']].unique().tolist()
    classes = np.unique(np.array(kwargs['classes'])).tolist()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(kwargs['classes'])}
    return data_table, classes, class_to_idx

def calculate_mean_std(dataloader):
    '''
    Source
    -------
    https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
    '''
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, labels, paths in tqdm(dataloader, total=len(dataloader)):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return (mean, std)

def wl_array(array, WW, WL): #https://storage.googleapis.com/kaggle-forum-message-attachments/1010629/17014/convert_to_jpeg_for_kaggle.py
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(array.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

def dicom_to_hu(img_path):
    dcm_data = pydicom.read_file(img_path)
    p = dcm_data.pixel_array
    s = dcm_data.RescaleSlope
    i = dcm_data.RescaleIntercept
    return (p*s+i)

def Normalize_0_1(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def Normalize_255(array):
    return (255*(array - np.min(array))/np.ptp(array)).astype(int)

def Normalize_1_1(array):
    return 2.*(array - np.min(array))/np.ptp(array)-1

def dicom_handler(img_path, num_output_channels=1, WW=None, WL=None):

    dcm_data = pydicom.read_file(img_path)
    modality = dcm_data.Modality
    num_source_channel = dcm_data.SamplesPerPixel

    if dcm_data.Modality == 'CT':
        array = dicom_to_hu(img_path)

        if num_output_channels == 1:

            if all(i!=None for i in [WW, WL]):
                img = Image.fromarray(wl_array(array, WW, WL))
            else:
                img = Image.fromarray(dicom_to_hu(img_path).astype('float32'), 'F')

        elif num_output_channels == 3:

            if all(i!=None for i in [WW, WL]):
                channels = [wl_array(array, WW=WW[c], WL=WL[c]) for c in range(num_output_channels)]
                img = Image.fromarray(np.dstack(channels))
            else:
                channels = [array for c in range(num_output_channels)]
                img_stacked = np.dstack(channels)
                img_stacked = Normalize_0_1(img_stacked)*255
                img = Image.fromarray(img_stacked.astype(np.uint8))


        else:
            raise ValueError('Only 1 or 3 channels is supported.')


    else:

        if num_source_channel != 3:

            if num_output_channels == 1:
                    img = Image.fromarray(dcm_data.pixel_array.astype('float32'))

            elif num_output_channels == 3:
                    array = dcm_data.pixel_array
                    channels = [array for c in range(num_output_channels)]
                    img_stacked = np.dstack(channels)
                    img_stacked = Normalize_0_1(img_stacked)*255
                    img = Image.fromarray(img_stacked.astype(np.uint8))

            else:
                raise ValueError('Only 1 or 3 channels is supported.')

        else:
            img = Image.fromarray(dcm_data.pixel_array)

    return img

def check_wl(WW, WL):

    if all (type(i)==list and type(i[0])==str for i in [WW, WL]):
        WW = [v['window'] for k, v in CT_window_level.items() if k in WW ]
        WL = [v['level'] for k, v in CT_window_level.items() if k in WL ]
    return WW, WL

def add_uid_column(df, length=10):
    df['uid'] = [int(str(uuid.uuid1().int)[:length]) for i in range (len(df)) ]
    return df


###### CLASSIFIER ######

def create_seq_classifier(fc, i, l, o, batch_norm=True): #needs documentation
    layers = {}
    layers['fc0']= nn.Linear(i, l[0])
    layers['r0']= nn.ReLU()
    if batch_norm: layers['bn0']= nn.BatchNorm1d(l[0])
    for i in range (0, len(l)-1):
        layers['fc'+str(i+1)]= nn.Linear(l[i], l[i+1])
        layers['r'+str(i+1)]= nn.ReLU()
        if batch_norm: layers['bn'+str(i+1)]= nn.BatchNorm1d(l[i+1])
    layers['fc'+ str(fc)]= nn.Linear(l[-1],o)
    return nn.Sequential(OrderedDict([(k,v) for k, v in layers.items()]))

def init_w(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.normal_(mean=0.0, std=y)
        m.bias.data.fill_(0)

def grab_pytorch_model(model_arch, pretrained):
    model = eval('models.'+model_arch+ "()")
    if pretrained:
        state_dict = load_url(model_url[model_arch], progress=True)
        model.load_state_dict(state_dict)
    return model

def remove_last_layers(model, model_arch):
    if 'vgg' in model_arch: model.classifier = model.classifier[0]
    elif 'resnet' in model_arch : model.fc = Identity()
    elif 'alexnet' in model_arch: model.classifier = model.classifier[:2]
    elif 'inception' in model_arch: model.fc = Identity()
    return model

def model_info(model, list=False, batch_size=1, channels=3, img_dim=224):
    if isinstance(model, str): model = eval('models.'+model+ "()")
    if list:
        return list(model.named_children())
    else:
        return summary(model, input_size=(batch_size, channels, img_dim, img_dim), depth=channels, col_names=["input_size", "output_size", "num_params"],)



###### GAN ######
def get_img_dim(img_size, divisions):
    for i in range(divisions):
        img_size = img_size/2
    return img_size

def generate_noise(self, noise_size, noise_type, batch_size=25):
    if noise_type =='normal':
        noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
    elif noise_type == 'gaussian':
        noise = np.random.normal(0, 1, size=(batch_size, noise_size))
    else:
        log('Noise type not specified/recognized. Please check.')
        pass
    noise=torch.from_numpy(noise).float()
    return noise

def conv_unit(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, leaky_relu=True):
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    if leaky_relu:
        return nn.LeakyReLU(nn.Sequential(*layers))
    else:
        return nn.Sequential(*layers)

def deconv_unit(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
    layers = []
    deconv_layer = nn.ConvTranspose2d(in_channels, out_channels,kernel_size, stride, padding, bias=False)
    layers.append(deconv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)
