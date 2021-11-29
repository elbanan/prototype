from .utils import *



def show_stack(imgs, figsize):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False,figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


class DICOMDataset():

    def __init__(
    self, root, ext='dcm', type='file', \
    label_table=None, path_col='path', study_col='study_id', \
    label_col='label', num_output_channels=1, \
    transform=False, WW=None, WL=None, split=None, \
    ignore_zero_img=False, sample=False, train_balance=False, batch_size=16, output_subset='all'):


        if root.endswith('/'):self.root = root
        else: self.root = root+'/'

        self.ext = ext
        self.type=type
        self.label_col = label_col
        self.path_col = path_col
        self.study_col=study_col
        self.num_output_channels=num_output_channels

        self.idx = {}
        self.ignore_zero_img = ignore_zero_img
        self.sample = sample
        self.train_balance=train_balance
        self.batch_size = batch_size


        self.window, self.level = check_wl(WW, WL)


        if isinstance(label_table, dict): #NEEDS CHECK FOR DICOM DIRECTORY STRUCTURE
            self.data_table, self.classes, self.class_to_idx = dict_to_data(table_dict=label_table, classes=self.classes, label_col = self.label_col)
            for k,v in self.data_table:
                v = add_uid_column(v, length=10)

        else:
            if isinstance(label_table, pd.DataFrame):
                self.data_table= {}
                table = label_table
                self.classes = label_table[self.label_col].unique().tolist()
                self.class_to_idx =  {cls_name: i for i, cls_name in enumerate(self.classes)}
            else:
                self.data_table = {}
                if self.type=='file':
                    self.classes, self.class_to_idx = find_classes(self.root)
                    table = root_to_data(root=self.root, ext=self.ext, path_col=self.path_col, label_col=self.label_col)
                    if len(table) == 0:
                        raise ValueError('No .{:} files were found in {:}. Please check.'.format(self.ext, self.root))
                elif self.type=='directory':
                    self.classes, self.class_to_idx, table = self.dicom_dir_to_table(self.root, self.ext)


            if split:
                self.valid_percent = split['valid']
                if 'test' in split:
                    self.test_percent = split['test']
                    self.train_percent = 1.0-(self.valid_percent+self.test_percent)
                    self.idx['test'], self.idx['valid'], self.idx['train'] = split_data(table, valid_percent=self.valid_percent, test_percent=self.test_percent)
                    for i in subsets:
                        self.data_table[i] = table.loc[self.idx[i],:]
                        self.data_table[i] = add_uid_column(self.data_table[i], length=10)
                else:
                    self.train_percent = 1.0-self.valid_percent
                    self.idx['train'], self.idx['valid'] = self.split_data(table, valid_percent=self.valid_percent, test_percent=False)
                    for i in subsets[:2]:
                        self.data_table[i] = table.loc[self.idx[i],:]
                        self.data_table[i] = add_uid_column(self.data_table[i], length=10)

            else:
                self.data_table['train'] = table


        if self.ignore_zero_img:
            if self.type != 'file':
                raise ValueError('Ignore empty images feature is currently available with DICOMDatasets with "file" type.')
            else:
                for i in self.ignore_zero_img:
                    self.data_table[i] = check_zero_image(table=self.data_table[i], path_col=self.path_col)

        if self.sample:
            for k, v in self.data_table.items():
                self.data_table[k] = v.sample(frac=self.sample, random_state=100)

        if self.train_balance:
            self.data_table['train'] = balance(df=self.data_table['train'], method=self.train_balance, label_col=self.label_col, classes=self.classes, )

        self.data_table['train'] = add_uid_column(self.data_table['train'], length=10)

        if isinstance(transform, dict):
            self.transforms = transform
        else:
            self.transforms = {k:transforms.Compose([transforms.ToTensor()]) for k,v in self.data_table.items()}

        self.loaders = self.get_loaders(batch_size=batch_size, subset=output_subset)
        if self.type == 'file':
            self.img_size = self.loaders['train'].dataset[0][1].shape[1]

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        for i in ['train', 'valid','test']:
            try: info.loc[len(info.index)] = [i+' dataset size', len(self.data_table[i])]
            except: pass
        return info

    def dicom_dir_to_table(self, root, ext):
        classes, class_to_idx = find_classes(root)
        table = pd.DataFrame(columns=[self.study_col, self.path_col, 'num_images', self.label_col])
        for c in classes:
            study_dir = os.path.join(root, c)
            study_idx = [x for x in os.walk(os.path.join(root, c))][0][1]
            study_paths = [x[0] for x in os.walk(os.path.join(root, c))][1:]
            for i in range(len(study_idx)):
                table.loc[len(table.index)] = [study_idx[i], study_paths[i], len([file for file in glob.glob(study_paths[i]+"/"+ "**/*."+ext, recursive=True)]), c ]
        return classes, class_to_idx, table

    def get_loaders(self, batch_size=16, shuffle=True, subset='all'):
        if subset == 'all':
            output = {}
            for k, v in self.data_table.items():
                output[k] = DICOMProcessor(root=self.root, ext=self.ext, num_output_channels=self.num_output_channels, table=v, type=self.type, class_to_idx = self.class_to_idx, path_col=self.path_col, study_col=self.study_col, label_col=self.label_col, \
                transform=self.transforms[k], window=self.window, level=self.level).get_loaders(batch_size=batch_size, shuffle=shuffle)
            return output
        else:
            return {subset: DICOMProcessor(root=self.root, ext=self.ext, num_output_channels=self.num_output_channels, table=self.data_table[subset], type=self.type, class_to_idx = self.class_to_idx, study_col=self.study_col, path_col=self.path_col, label_col=self.label_col, \
            transform=self.transforms[subset], window=self.window, level=self.level).get_loaders(batch_size=batch_size, shuffle=shuffle)}

    def view_batch(self, data='train', figsize = (25,5), study_id=0, rows=2, batch_size=16, shuffle=True, num_images=False, cmap='gray'):
        loader = DICOMProcessor(root=self.root, ext=self.ext, num_output_channels=self.num_output_channels, table=self.data_table[data], type=self.type, class_to_idx = self.class_to_idx, path_col=self.path_col, study_col=self.study_col, label_col=self.label_col, transform=self.transforms[data], \
        window=self.window, level=self.level).get_loaders(batch_size=batch_size, shuffle=shuffle)

        if self.type == 'file':
            uidx, images, labels  = (iter(loader)).next()
            images = images.cpu().numpy()
            labels = labels.cpu().numpy()
            batch = images.shape[0]
            if num_images:
                if num_images > batch:
                    print('Warning: Selected number of images is less than batch size. Displaying a batch instead.')
                else:
                    batch = num_images

            fig = plt.figure(figsize=figsize)
            for i in np.arange(batch):
                ax = fig.add_subplot(rows, int(batch/rows), i+1, xticks=[], yticks=[])
                if images[i].shape[0] == 3:
                    ax.imshow(np.transpose(images[i], (1, 2, 0)), cmap=cmap)
                elif images[i].shape[0] ==1:
                    ax.imshow(np.squeeze(images[i]), cmap=cmap)
                ax.set_title(self.classes[labels[i]])
        elif self.type == 'directory':
            imgs = loader.dataset[study_id][1]
            if num_images:
                imgs = imgs[:num_images]
            grid = make_grid(imgs)
            show_stack(grid, figsize)


    def view_multichannel_image(self, data='train', idx=0, figsize = (25,5), batch_size=16, shuffle=True, cmap='gray'):
        loader = DICOMProcessor(root=self.root, ext=self.ext, table=self.data_table[data], type='file', class_to_idx = self.class_to_idx, path_col=self.path_col, label_col=self.label_col, transform=self.transforms[data], \
        num_output_channels = self.num_output_channels, window=self.window, level=self.level).get_loaders(batch_size=batch_size, shuffle=shuffle)
        uidx, images, labels  = (iter(loader)).next()

        images = images.cpu().numpy()
        labels = labels.cpu().numpy()

        img = images[idx]

        if img.shape[0] == 1:
            print ('Warning: Selected image does not have 3 channels. Please check.')

        num_channels = img.shape[0]
        fig = plt.figure(figsize=figsize)
        img = np.transpose(img, (1, 2, 0))

        channels=[img[:,:,i] for i in range (0, num_channels)]

        for i in range(0, num_channels):
            ax = fig.add_subplot(1, num_channels, i+1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(channels[i]), cmap=cmap)
            ax.set_title('channel '+str(i))

    def header_info(self, data='train', limit=10):
        table = self.data_table[data]
        header_col = []
        for c in self.classes:
            g = pydicom.read_file((table[table[self.label_col] == c]).iloc[0][self.path_col])
            header_col += [g[k].keyword for k in g.keys()]
        df = pd.DataFrame(columns = (list(set(header_col))).sort())
        s = 0
        for i, r in self.data_table[data].iterrows():
            if s in range(0, limit):
                d = pydicom.read_file(r[self.path_col])
                df = df.append(dict([(d[k].keyword, d[k].value) for k in d.keys() if d[k].keyword != 'PixelData']), ignore_index=True)
                s +=1
        return df

    def examine_img(self, data='train', figure_size=(15,15), resize=(128,128), img_idx=0, cmap='gray'):

        img = pydicom.read_file(self.data_table[data][self.path_col].tolist()[img_idx]).pixel_array
        img = cv2.resize(img, dsize=resize, interpolation=cv2.INTER_CUBIC)
        fig = plt.figure(figsize = figure_size)
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap=cmap)
        width, height = img.shape
        thresh = img.max()/2.5
        for x in range(width):
            for y in range(height):
                val = round(img[x][y],2) if img[x][y] !=0 else 0
                ax.annotate(str(val), xy=(y,x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white' if img[x][y]<thresh else 'black')

    def data_stat(self, plot=False, figure_size=(8, 6)):
        d, c, i, n = [],[],[],[]
        for k, v in self.data_table.items():
            for l, j in self.class_to_idx.items():
                d.append(k)
                c.append(l)
                i.append(j)
                n.append(v[self.label_col].value_counts()[l].sum())
        df = pd.DataFrame(list(zip(d, c, i, n)), columns=['Dataset', 'Class', 'Class_idx', 'Count'])
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=figure_size)
            ax =  sns.barplot(x="Dataset", y="Count", hue="Class", data=df, palette="viridis")
            show_values_on_bars(ax)
        else:
            return df


class DICOMProcessor(Dataset):

    def __init__(self, root, ext='dcm', type='file', table=None, class_to_idx = None, path_col=None, label_col=None, study_col=None, transform=None, num_output_channels=1, window=None, level=None, split=None, ):
        self.ext = ext
        self.type=type
        if root.endswith('/'):self.root = root
        else: self.root = root+'/'
        self.num_output_channels=num_output_channels
        self.window= window
        self.level = level
        self.class_to_idx = class_to_idx

        if path_col:self.path_col = path_col
        else: self.path_col = 'path'

        if label_col:self.label_col = label_col
        else: self.label_col = 'label'

        if path_col:self.study_col = study_col
        else: self.study_col = 'study_id'

        if isinstance(table, pd.DataFrame):
            self.table = table
            if self.class_to_idx:
                self.classes = [k for k, v in self.class_to_idx.items()]
            else:
                self.classes = self.table[self.label_col].unique().tolist()
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            if self.type == 'file':
                if self.class_to_idx:
                    self.classes = [k for k, v in self.class_to_idx.items()]
                else:
                    self.classes, self.class_to_idx = find_classes(self.root)
                self.table = root_to_data(root=self.root, ext=self.ext, path_col=self.path_col, label_col=self.label_col)
                if len(self.table) == 0:
                    raise ValueError('No .{:} files were found in {:}. Please check.'.format(ext, self.root))
            elif self.type == 'directory':
                self.classes, self.class_to_idx, self.table = self.dicom_dir_to_table(self.root, self.ext)

        if transform == None:
            self.transforms = transforms.Compose([transforms.ToTensor()])
        else:
            self.transforms = transform

    def dicom_dir_to_table(self, root, ext):
        classes, class_to_idx = find_classes(root)
        table = pd.DataFrame(columns=[self.study_col, self.path_col, 'num_images', self.label_col])
        for c in classes:
            study_dir = os.path.join(root, c)
            study_idx = [x for x in os.walk(os.path.join(root, c))][0][1]
            study_paths = [x[0] for x in os.walk(os.path.join(root, c))][1:]
            for i in range(len(study_idx)):
                table.loc[len(table.index)] = [study_idx[i], study_paths[i], len([file for file in glob.glob(study_paths[i]+"/"+ "**/*."+ext, recursive=True)]), c ]
        return classes, class_to_idx, table

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        return info

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        path = self.table.iloc[idx][self.path_col]
        label = self.table.iloc[idx][self.label_col]
        label_id = self.class_to_idx[label]

        if self.type == 'file':
            if self.ext != 'dcm': img=Image.open(P)
            else: img = dicom_handler(img_path=P, num_output_channels=self.num_output_channels, WW=self.window, WL=self.level)
            if self.transforms: img=self.transforms(img)

        elif self.type=='directory':
            img_list = sorted([(file, float(pydicom.read_file(file).SliceLocation)) for file in glob.glob(path+"/"+  "**/*."+self.ext, recursive=True)],key=lambda x: x[1], reverse=True)
            img_list = [i[0] for i in img_list]
            img = [dicom_handler(img_path=i, num_output_channels=self.num_output_channels, WW=self.window, WL=self.level) for i in img_list]
            if self.transforms:
                img = [self.transforms(i) for i in img]
                img = torch.stack(img)

        try:
            uid = self.table.iloc[idx]['uid']
        except:
            uid = self.table.index.values.tolist()[idx]
        return  uid, img, label_id

    def get_loaders(self, batch_size=16, shuffle=True):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle)




# Allow to select subset of the CT stack
# check 3 w/l on  CT stack
