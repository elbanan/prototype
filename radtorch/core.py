from .utils import *

# %matplotlib inline
# %config InlineBackend.figure_format='retina'


class DICOMDataset():

    def __init__(self, root, ext='dcm', label_table=None, path_col='img_path', label_col='img_label', transform=None, HU=False, window=None, level=None, split=None, ignore_zero_img=False, sample=False, train_balance=False):

        subsets = ['train', 'valid', 'test']
        if root.endswith('/'):self.root = root
        else: self.root = root+'/'
        self.ext = ext
        self.label_col = label_col
        self.path_col = path_col
        self.transforms = {}
        self.HU = HU

        self.idx = {}
        self.ignore_zero_img = ignore_zero_img
        self.sample = sample
        self.train_balance=train_balance

        self.window= window
        self.level = level
        self.multilevelwindow=False
        if isinstance(self.window, dict):
            self.multilevelwindow = True
        if isinstance(self.level, dict):
            self.multilevelwindow = True


        if isinstance(label_table, dict):
            self.data_table, self.classes, self.class_to_idx = dict_to_data(table_dict=label_table, classes=self.classes, label_col = self.label_col)
        else:
            if isinstance(label_table, pd.DataFrame):
                self.data_table= {}
                table = label_table
                self.classes = label_table[self.label_col].unique().tolist()
                self.class_to_idx =  {cls_name: i for i, cls_name in enumerate(self.classes)}
            else:
                self.data_table = {}
                self.classes, self.class_to_idx = find_classes(self.root)
                table = root_to_data(root=self.root, ext=self.ext, path_col=self.path_col, label_col=self.label_col)
                table['uid'] = table.index.tolist()
                if len(table) == 0:
                    raise ValueError('No .{:} files were found in {:}. Please check.'.format(self.ext, self.root))
            if split:
                self.valid_percent = split['valid']
                if 'test' in split:
                    self.test_percent = split['test']
                    self.train_percent = 1.0-(self.valid_percent+self.test_percent)
                    self.idx['test'], self.idx['valid'], self.idx['train'] = split_data(table, valid_percent=self.valid_percent, test_percent=self.test_percent)
                    for i in subsets:
                        self.data_table[i] = table.loc[self.idx[i],:]
                else:
                    # self.test_percent=None
                    self.train_percent = 1.0-self.valid_percent
                    self.idx['train'], self.idx['valid'] = self.split_data(table, valid_percent=self.valid_percent, test_percent=False)
                    for i in subsets[:2]:
                        self.data_table[i] = table.loc[self.idx[i],:]
            else:
                self.data_table['train'] = table

        if self.ignore_zero_img:
            for i in self.ignore_zero_img:
                self.data_table[i] = check_zero_image(table=self.data_table[i], path_col=self.path_col)

        if self.sample:
            for k, v in self.data_table.items():
                self.data_table[k] = v.sample(frac=self.sample, random_state=100)

        if self.train_balance:
            self.data_table['train'] = balance(df=self.data_table['train'], method=self.train_balance, label_col=self.label_col, classes=self.classes, )

        if type(transform) is dict:
            self.transforms = transform
        else:
            for k,v in self.data_table.items():
                self.transforms[k] = transforms.Compose([transforms.ToTensor()])

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        for i in ['train', 'valid','test']:
            try: info.loc[len(info.index)] = [i+' dataset size', len(self.data_table[i])]
            except: pass
        return info

    def get_loaders(self, batch_size=16, shuffle=True):
        if 'loaders' in self.__dict__.keys():
            return self.loaders
        else:
            output = {}
            for k, v in self.data_table.items():
                output[k] = DICOMProcessor(root=self.root, ext=self.ext, table=v, path_col=self.path_col, label_col=self.label_col, transform=self.transforms[k], HU=self.HU, window=self.window, level=self.level).get_loaders(batch_size=batch_size, shuffle=shuffle)
            return output

    def view_batch(self, data='train', figsize = (25,5), rows=2, batch_size=16, shuffle=False, num_images=None):
        loader = DICOMProcessor(root=self.root, ext=self.ext, table=self.data_table[data], path_col=self.path_col, label_col=self.label_col, transform=self.transforms[data], \
        HU=self.HU, window=self.window, level=self.level).get_loaders(batch_size=batch_size, shuffle=shuffle)
        uidx, images, labels  = (iter(loader)).next()

        images = images.cpu().numpy()
        labels = labels.cpu().numpy().tolist()

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
                img = images[i]
                img = img / 2 + 0.5
                ax.imshow(np.transpose(img, (1, 2, 0)))

            elif images[i].shape[0] ==1:
                ax.imshow(np.squeeze(images[i]), cmap='gray')

            label = self.classes[labels[i]]
            ax.set_title(label)

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

    def __init__(self, root, ext='dcm', table=None, class_to_idx = None, path_col=None, label_col=None, transform=None, HU=False, window=None, level=None, split=None, ):
        self.ext = ext
        if root.endswith('/'):self.root = root
        else: self.root = root+'/'
        self.HU = HU
        self.window= window
        self.level = level
        self.multilevelwindow = False
        if isinstance(self.window, dict):
            self.multilevelwindow = True
        if isinstance(self.level, dict):
            self.multilevelwindow = True
        self.class_to_idx = class_to_idx

        if path_col:self.path_col = path_col
        else: self.path_col = 'img_path'

        if label_col:self.label_col = label_col
        else: self.label_col = 'img_label'

        if isinstance(table, pd.DataFrame):
            self.table = table
            if self.class_to_idx:
                self.classes = [k for k, v in self.class_to_idx.items()]
            else:
                self.classes = self.table[self.label_col].unique().tolist()
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            if self.class_to_idx:
                self.classes = [k for k, v in self.class_to_idx.items()]
            else:
                self.classes, self.class_to_idx = find_classes(self.root)
            self.table = root_to_data(root=self.root, ext=self.ext, path_col=self.path_col, label_col=self.label_col)
            if len(self.table) == 0:
                raise ValueError('No .{:} files were found in {:}. Please check.'.format(ext, self.root))

        if transform == None:
            self.transforms = {'train': transforms.Compose([transforms.ToTensor()]), 'test':transforms.Compose([transforms.ToTensor()])}
        else:
            self.transforms = transform

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        return info

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        P = self.table.iloc[idx][self.path_col]
        L = self.table.iloc[idx][self.label_col]
        L_id = self.classes[L]
        if self.ext != 'dcm':
            img=Image.open(P)
        else:
            if self.multilevelwindow:
                w = self.window[L]
                l = self.level[L]
                img=dicom_to_array(filepath=P, HU=self.HU, window=w, level=l)
            else:
                img=dicom_to_array(filepath=P, HU=self.HU, window=self.window, level=self.level)
            img=dicom_array_to_pil(img)
        if self.transforms:
            img=self.transforms(img)
        try:
            uid = self.table.iloc[idx]['uid']
        except:
            uid = P
        # return  uid, img, [v for k, v in self.class_to_idx.items() if k == self.table.iloc[idx][self.label_col]][0]
        return  uid, img, L_id


    def get_loaders(self, batch_size=16, shuffle=True):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle)

class ImageClassifier():

    def __init__(self,model,loaders,criterion, optimizer):
        self.loaders = loaders
        self.classes = self.class_to_idx()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        for i in ['train', 'valid','test']:
            try: info.loc[len(info.index)] = [i+' dataset size', len(self.loaders[i].dataset)]
            except: pass
        return info

    def class_to_idx(self):
        classes = self.loaders['train'].dataset.classes
        return dict(zip(classes, [i for i in range(0,len(classes))]))

    def fit(self, epochs=20, valid=True, verbose_level=('epoch', 1), output_model='best_model.pt'):
        self.valid_loss_min = np.Inf
        print ("Starting model training on "+str(self.device))
        self.training_model = deepcopy(self.model)
        self.training_model = self.training_model.to(self.device)
        self.train_losses, self.valid_losses = [], []
        steps = 0
        verbose, print_every = verbose_level
        for e in tqdm(range(0,epochs)):
            train_loss = 0
            valid_loss = 0
            #TRAIN MODEL
            self.training_model.train()
            for i, (idx, images, labels) in enumerate(self.loaders['train']):
                steps += 1
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad();
                output = self.training_model(images);
                loss = self.criterion(output, labels);
                loss.backward();
                self.optimizer.step();
                train_loss += loss.item()*images.size(0)
                if verbose == 'batch':
                    if i % print_every == 0:
                        print (
                              "Epoch: {:4}/{:4} |".format(e+1, epochs),
                              "Batch: {:4}/{:4} |".format(i, len(self.loaders['train'])),
                              "Train Loss:{:.5f} |".format(train_loss/len(self.loaders['train'].dataset))
                              )
            #VALID MODEL
            if valid:
                with torch.no_grad():
                    self.training_model.eval()
                    for ii, (idx, images, labels) in enumerate(self.loaders['valid']):
                        images, labels = images.to(self.device), labels.to(self.device)
                        output = self.training_model(images)
                        loss = self.criterion(output, labels)
                        valid_loss += loss.item()*images.size(0)
                self.train_losses.append(train_loss/len(self.loaders['train'].dataset))
                self.valid_losses.append(valid_loss/len(self.loaders['valid'].dataset))


                # SAVE MODEL IF VALID_LOSS IS DECREASING
                epoch_valid_loss = valid_loss/len(self.loaders['valid'].dataset)
                if epoch_valid_loss < self.valid_loss_min:
                    self.best_model = deepcopy(self.training_model)
                    self.valid_loss_min = epoch_valid_loss
                    checkpoint = {'type':'NN',
                                  'model':self.best_model,
                                  'optimizer_state_dict' : self.optimizer.state_dict(),
                                  'train_losses': self.train_losses,
                                  'valid_losses': self.valid_losses,
                                  'valid_loss_min': self.valid_loss_min,
#                       'state_dict': train_model.state_dict()
                                 }
                    torch.save(checkpoint, output_model)
                    save_status, validation_decrease_status = True, True
                else:
                    save_status, validation_decrease_status = False, False


            #PRINT RESULTS
            if valid:
                if verbose == 'epoch' :
                    if e % print_every == 0:
                        print (
                                "Epoch: {:4}/{:4} |".format(e, epochs),
                                "Train Loss: {:.5f} |".format(train_loss/len(self.loaders['train'].dataset)),
                                "Valid Loss: {:.5f} (best: {:.5f}) |".format(valid_loss/len(self.loaders['valid'].dataset),self.valid_loss_min),
                                "Valid Loss dec: {:5} |".format(str(validation_decrease_status)),
                                "Model saved: {:5} ".format(str(save_status)))

                else:
                    if verbose:
                        print (
                              "Epoch: {:4}/{:4} |".format(e, epochs),
                              "Train Loss :{:.5f} |".format(train_loss/len(self.loaders['train'].dataset)),
                              "Valid Loss: {:.5f} (best: {:.5f}) |".format(valid_loss/len(self.loaders['valid'].dataset),self.valid_loss_min),
                              "Valid Loss dec: {:5} |".format(str(validation_decrease_status)),
                              "Model saved: {:5} ".format(str(save_status)))

            else:
                if verbose == 'epoch' :
                    if e % print_every == 0:
                        print (
                                "Epoch: {:4}/{:4} |".format(e, epochs),
                                "Train Loss: {:.5f} |".format(train_loss/len(self.loaders['train'].dataset)))

                else:
                    if verbose:
                        print (
                              "Epoch: {:4}/{:4} |".format(e, epochs),
                              "Train Loss :{:.5f} |".format(train_loss/len(self.loaders['train'].dataset)))

            if e+1 == epochs:
                print ('Training Finished Successfully!')

        self.train_logs=pd.DataFrame({"train": self.train_losses, "valid" : self.valid_losses})
        self.metrics = Metrics(self, self.device, use_best=True)
        self.inference = Inference(self)

    def weight_init(self):
        self.model.apply(init_w)
        print('Model weight initialization applied successfully.')

    def view_train_logs(self, data='all', figsize=(12,8)):
        plt.figure(figsize=figsize)
        sns.set_style("darkgrid")
        if data == 'all':
            p = sns.lineplot(data = self.train_logs)
        else:
            p = sns.lineplot(data = self.train_logs[data].tolist())

        p.set_xlabel("epoch", fontsize = 10)
        p.set_ylabel("loss", fontsize = 10);

    def export(self, path):
        try:
            torch.save(self, path)
            print ('Export done successfully.')
        except:
            raise ValueError('Cannot Export.')

class Metrics():

    def __init__(self, classifier, device, use_best=True):
        self.classifier = classifier
        self.device = device
        self.auc='Please run metrics.roc() to generate roc auc.'
        self.accuracy = 'Please run metrics.test() to generate accuracy.'
        self.use_best = use_best

    def get_predictions(self, target_loader):
        if self.use_best:
            self.selected_model = self.classifier.best_model
        else:
            self.selected_model = self.classifier.training_model

        true_labels = []
        pred_labels = []
        for i, (idx, imgs, labels) in tqdm(enumerate(self.classifier.loaders[target_loader]), total=len(self.classifier.loaders[target_loader])):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            true_labels = true_labels+labels.tolist()
            with torch.no_grad():
                self.selected_model.to(self.device)
                self.selected_model.eval()
                out = self.selected_model(imgs)
                pr = [(i.tolist()).index(max(i.tolist())) for i in out]
                pred_labels = pred_labels+pr
        return true_labels, pred_labels

    def test(self):
        if self.use_best:
            self.selected_model = self.classifier.best_model
        else:
            self.selected_model = self.classifier.training_model

        test_loss = 0.0
        class_correct = list(0. for i in range(len(self.classifier.classes)))
        class_total = list(0. for i in range(len(self.classifier.classes)))

        test_model = self.selected_model.to(self.device)

        with torch.no_grad():
            test_model.eval()
            for idx, data, target in self.classifier.loaders['test']:
                data, target = data.to(self.device), target.to(self.device)
                output = test_model(data)
                loss = self.classifier.criterion(output, target)
                test_loss += loss.item()*data.size(0)
                _, pred = torch.max(output, 1)
                correct = np.squeeze(pred.eq(target.data.view_as(pred)))
                for i in range(data.shape[0]):
                    label = target.data[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1

            test_loss = test_loss/len(self.classifier.loaders['test'].dataset)
            print('Overall Test Loss: {:.6f}\n'.format(test_loss))

            for i in range(len(self.classifier.classes)):
                c = next((k for k, v in self.classifier.classes.items() if v == i), None)
                if class_total[i] > 0:
                    print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                        c , 100 * class_correct[i] / class_total[i],
                        np.sum(class_correct[i]), np.sum(class_total[i])))

                else:
                    print('Test Accuracy of %5s: N/A (no training examples)' % (c))

            print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
                100. * np.sum(class_correct) / np.sum(class_total),
                np.sum(class_correct), np.sum(class_total)))

            self.accuracy = 100. * np.sum(class_correct) / np.sum(class_total)

    def confusion_matrix(self, target_loader='test', figure_size=(8,6), cmap='Blues', percent=False):
        #https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
        true_labels, pred_labels = self.get_predictions(target_loader=target_loader)
        cm = metrics.confusion_matrix(true_labels, pred_labels)
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
        if len(cm)==2:
            precision = cm[1,1] / sum(cm[:,1])
            recall    = cm[1,1] / sum(cm[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}\nMisclassified={:}".format(accuracy, misclass)
        plt.figure(figsize=figure_size)
        sns.set_style("darkgrid")
        if percent:
            sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap=cmap, xticklabels=self.classifier.classes,yticklabels=self.classifier.classes, linewidths=1, linecolor='black')
        else:
            sns.heatmap(cm, annot=True, cmap=cmap, xticklabels=self.classifier.classes,yticklabels=self.classifier.classes,  linewidths=1, linecolor='black')
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
        plt.title('Confusion Matrix', fontweight='bold')

    def roc(self, target_loader= 'test', figure_size=(8,6)):
        true_labels, pred_labels = self.get_predictions(target_loader=target_loader)
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_labels)
        auc = metrics.roc_auc_score(true_labels, pred_labels)
        sns.set_style("darkgrid")
        fig = plt.figure(figsize=figure_size)
        plt.plot([0, 0.5, 1.0], [0, 0.5, 1.0], linestyle=':')
        plt.plot(fpr, tpr, linestyle='--', lw=1.1,  label = "ROC AUC = {:0.3f}".format(auc))
        plt.xlabel('False Positive Rate (1-Specificity)')
        plt.ylabel('True Positive Rate (Senstivity)')
        plt.title('Receiver Operating Characteristic Curve',y=-0.2 , fontweight='bold')
        plt.legend()
        plt.show()
        self.auc = auc

    def all(self):
        self.test()
        self.confusion_matrix()
        self.roc()

class FeatureExtractor():

    def __init__(self, model_arch, dataset):

        self.dataset = dataset
        self.model_arch = model_arch
        if self.model_arch not in supported_models:
            raise ValueError('Model not yet Supported. For list of supported models please use supported_models. Thanks')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = grab_pytorch_model(model_arch=self.model_arch, pretrained=True)
        self.model = remove_last_layers(model=self.model, model_arch=self.model_arch)
        self.model = self.model.to(self.device)

    def run(self, batch_size=16):
        self.batch_size= batch_size
        self.loaders = self.dataset.get_loaders(batch_size=batch_size)
        with torch.no_grad():
            self.model.eval()
            uid_list = torch.IntTensor([])
            for i, (uid, images, labels) in tqdm(enumerate(self.loaders['train']), total=len(self.loaders['train'])):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                if i == 0:
                    features = deepcopy(output)
                    uid_list = deepcopy(uid)
                else:
                    features = torch.cat((features,output), 0)
                    uid_list = torch.cat((uid_list, uid), 0)
        self.features = pd.DataFrame(features.cpu().numpy())
        self.features['uid'] = uid_list.numpy()
        self.feature_names = self.features.columns.tolist()[:-1]

    def num_features(self):
        return self.features.shape[1]

    def plot_features(self, annotations=False, figure_size=(10,10), cmap="YlGnBu"):
        plt.figure(figsize=figure_size)
        plt.title("Average Rating of Games Across All Platforms")
        sns.heatmap(data=self.features, annot=annotations,cmap=cmap);

    def model_info(self):
        batch  = (next(iter(self.loaders['train'])))[1]
        batch_size, channels, img_dim, img_dim = batch.shape
        return model_info(self.model, list=False, batch_size=batch_size, channels=channels, img_dim=img_dim)

    def hybrid_table(self, sklearn_ready=False):
        h = pd.merge(self.features, self.dataset.data_table['train'], on='uid')
        if sklearn_ready:
            f, l = self.features[self.feature_names], h[self.dataset.label_col]
            return f, l
        else:
            return h

class Inference():
    def __init__(self, classifier=False, feature_extractor=False, mode='non-sklearn', specific_transform=False):
        self.mode = mode
        if self.mode == 'non-sklearn':
            self.classifier = classifier
            self.model = self.classifier.best_model
            self.ds = self.classifier.loaders['test'].dataset
            self.transforms = self.ds.transforms

        elif self.mode == 'sklearn':
            self.classifier = classifier
            self.feature_extractor = feature_extractor
            self.model = self.feature_extractor.model
            self.ds = self.feature_extractor.dataset
            self.transforms = self.ds.transforms['train']

        if specific_transform:self.transforms=specific_transform
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def predict(self, img_path, top_predictions='all', display_image=False, human=True):
        if top_predictions == 'all':
            top = len(self.ds.classes)
        else:
            top = top_predictions
            if top > len(self.ds.classes):
                raise ValueError('Number of top predictions is more than number of classes. Please check')

        name, ext = os.path.splitext(img_path)
        if ext != '.dcm':
            img=Image.open(img_path)
        else:
            img=dicom_to_array(filepath=img_path, HU=self.ds.HU, window=self.ds.window, level=self.ds.level)
            img=dicom_array_to_pil(img)
        img = self.transforms(img)
        img = torch.unsqueeze(img, 0)
        model, img = self.model.to(self.device), img.to(self.device)
        with torch.no_grad():
            model.eval()
            model_output = model(img)
        if self.mode == 'non-sklearn':
            raw_pred = torch.nn.functional.softmax(model_output, dim=1).cpu().numpy()
        elif self.mode == 'sklearn':
            raw_pred = self.classifier.predict_proba(model_output.cpu().numpy())

        predictions = []
        s=0
        for i in raw_pred:
            o = []
            for k, v in self.ds.class_to_idx.items():
                o.append({'id':s, 'class':k, 'class_id':v, 'prob':i.tolist()[v]})
                o = sorted(o, key = lambda i: i['prob'], reverse=True)[:top]
            s = s+1
            predictions.append(o)

        if display_image:
            if ext != '.dcm':
                plt.grid(False)
                plt.imshow(mpimg.imread(img_path))
            else:
                plt.grid(False)
                plt.imshow((pydicom.dcmread(img_path).pixel_array), cmap='gray');

        if human:
            for class_pred in predictions:
                for i in class_pred:
                    print('class: {:4} [prob: {:.2f}%]'.format(i['class'], i['prob']*100))


        return predictions




# FIX uid with upsample and downsample
# metrics for all sklearn and nn models



# self mean/std
# resume training
# set random seed
# T-Sne visualization
#select only certain classes
