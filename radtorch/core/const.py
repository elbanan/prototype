
requirements = ['imageio==2.9.0',
      'imagesize==1.2.0',
      'matplotlib==3.4.2',
      'matplotlib-inline==0.1.2',
      'numpy==1.19.5',
      'opencv-python==4.5.3.56',
      'pandas==1.3.0',
      'Pillow==8.3.1',
      'pydicom==2.1.2',
      'scikit-image==0.18.2',
      'scikit-learn==0.24.2',
      'scipy==1.7.0',
      'seaborn==0.11.1',
      'torch==1.9.0',
      'torchinfo==1.5.3',
      'torchvision==0.10.0',
      'tqdm==4.61.2',
      'urllib3==1.26.6',
      'wrapt==1.12.1',
      'xgboost==1.4.2',]

subsets = ['train', 'valid', 'test']

model_url = {
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
    "inception_v3": "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
    "efficientnet_b2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
    "efficientnet_b3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
    "efficientnet_b4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
    "efficientnet_b5": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
    "efficientnet_b6": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
    "efficientnet_b7": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
    }

supported_models= list(model_url.keys())

CT_window_level = {  #https://radiopaedia.org/articles/windowing-ct?lang=us
"brain": {"window":80, "level":40},
"subdural": {"window":200, "level":75},
"stroke_1": {"window":8, "level":32},
"stroke_2": {"window":40, "level":40},
"temporal_1": {"window":2800, "level":600},
"temporal_2": {"window":4000, "level":700},
"head_soft": {"window":380, "level":40},
"lungs": {"window":1500, "level":-600},
"mediastinum": {"window":350, "level":50},
"abdomen_soft": {"window":350, "level":50},
"liver": {"window":150, "level":30},
"spine_soft": {"window":250, "level":50},
"spine_bone": {"window":1800, "level":400},
}

imagenet_mean = (0.485, 0.456, 0.406) #https://github.com/pytorch/vision/issues/1439
imagenet_std = (0.229, 0.224, 0.225) #https://github.com/pytorch/vision/issues/1439

norm_mean = (0.5, 0.5, 0.5)
norm_std = (0.5, 0.5, 0.5)



image_classification_pipe_allowed_keys = ['root', 'model', 'model_arch','ext','label_table','path_col','label_col','num_output_channels','transform','WW','WL','split','ignore_zero_img', 'sample', 'train_balance','batch_size', 'output_subset', 'optimizer', 'criterion','device']
dataset_allowed_keys = ['root', 'ext','label_table','path_col','label_col','num_output_channels','transform','WW','WL','split','ignore_zero_img', 'sample', 'train_balance','batch_size', 'output_subset']
classifier_allowed_keys = ['model', 'dataset', 'feature_extractor_arch', 'criterion', 'optimizer', 'device']
