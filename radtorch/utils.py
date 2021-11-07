import torch, torchvision, itertools, glob, os, pydicom, copy, cv2

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

from const import *


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

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

def create_seq_classifier(fc, i, l, o, batch_norm=True):
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

def init_w(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.normal_(mean=0.0, std=y)
        m.bias.data.fill_(0)

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

def dicom_to_array(filepath, HU=False, window=None, level=None):
    dcm_info =  pydicom.read_file(filepath)
    if HU:
        try:
            hu_img = dcm_info.pixel_array*dcm_info.RescaleSlope + dcm_info.RescaleIntercept
            return hu_img.astype(float)
        except:
            raise ValueError('File {:} could not be converted to HU. Please check that modality is CT and DICOM header contains RescaleSlope and RescaleIntercept.'.format(filepath))
    elif window:
        if len(window) != 1:
            raise ValueError('Argument "wl" can only accept 1 combination of W and L when "WIN" mode is selected')
        try:
            img = dcm_info.pixel_array*dcm_info.RescaleSlope + dcm_info.RescaleIntercept # try to convert to HU before windowing
        except:
            img = dcm_info.pixel_array
        lower = level - (width / 2)
        upper = level + (width / 2)
        img[img<=lower] = lower
        img[img>=upper] = upper
        return img.astype(float)
    else:
        return dcm_info.pixel_array.astype(float)

def dicom_array_to_pil(pixel_array, mode=None):
    img = Image.fromarray(pixel_array)
    if mode:
        return img.convert(mode)
    else:
        return img

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

def set_random_seed(seed):
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        log('Random seed '+str(seed)+' set successfully')
    except:
        raise TypeError('Error. Could not set Random Seed. Please check again.')
        pass

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
