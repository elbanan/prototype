from .utils import *
from .metrics import *
from .inference import *
from .feature_extractor import *


class ImageClassifier():

    def __init__(self, model, dataset, feature_extractor_arch='vgg16', criterion=None, optimizer=None):

        self.loaders = dataset.loaders
        self.class_to_idx = dataset.class_to_idx
        self.dataset=dataset

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(type(model))[8:].startswith('sklearn'):
            self.classifier_type = 'sklearn'
            self.feature_extractors = {}
            for i in ['train', 'valid', 'test']:
                self.feature_extractors[i] = FeatureExtractor(model_arch=feature_extractor_arch, dataset=dataset, subset=i)

        else:
            self.classifier_type = 'torch'

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        for i in ['train', 'valid','test']:
            try: info.loc[len(info.index)] = [i+' dataset size', len(self.loaders[i].dataset)]
            except: pass
        return info

    def fit(self, epochs=20, valid=True, verbose_level= ('epoch', 1), output_model='best_model.pt',):

        if self.classifier_type == 'torch':
            self.train_nn(epochs=epochs, valid=valid, verbose_level=verbose_level, output_model=output_model)

        elif self.classifier_type == 'sklearn':
            if hasattr(self, 'train_features'): print (current_time(), 'Using pre-extracted training features.')
            else:
                print (current_time(), 'Running Feature Extraction using model architecture', self.feature_extractors['train'].model_arch)
                self.feature_extractors['train'].run()
                self.train_features, self.train_labels = self.feature_extractors['train'].hybrid_table(sklearn_ready=True)
            self.train_sklearn(output_model=output_model)


    def train_sklearn(self,**kwargs):
        output_model=kwargs['output_model']
        print (current_time(), "Starting model training on "+str(self.device))
        self.training_model = deepcopy(self.model)
        self.best_model = self.training_model
        self.best_model.fit(self.train_features, self.train_labels)
        save_checkpoint(classifier=self, output_file=output_model)
        print (current_time(), 'Training completed successfully.')

    def train_nn(self,**kwargs):
        epochs=kwargs['epochs']
        valid=kwargs['valid']
        verbose_level=kwargs['verbose_level']
        output_model=kwargs['output_model']

        self.valid_loss_min = np.Inf
        print (current_time(), "Starting model training on "+str(self.device))
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
                        print (current_time(),
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
                    save_checkpoint(classifier=self, output_file=output_model)
#                     checkpoint = {'type':'NN',
#                                   'model':self.best_model,
#                                   'optimizer_state_dict' : self.optimizer.state_dict(),
#                                   'train_losses': self.train_losses,
#                                   'valid_losses': self.valid_losses,
#                                   'valid_loss_min': self.valid_loss_min,
# #                       'state_dict': train_model.state_dict()
#                                  }
#                     torch.save(checkpoint, output_model)
                    save_status, validation_decrease_status = True, True
                else:
                    save_status, validation_decrease_status = False, False


            #PRINT RESULTS
            if valid:
                if verbose == 'epoch' :
                    if e % print_every == 0:
                        print (current_time(),
                                "Epoch: {:4}/{:4} |".format(e, epochs),
                                "Train Loss: {:.5f} |".format(train_loss/len(self.loaders['train'].dataset)),
                                "Valid Loss: {:.5f} (best: {:.5f}) |".format(valid_loss/len(self.loaders['valid'].dataset),self.valid_loss_min),
                                "Valid Loss dec: {:5} |".format(str(validation_decrease_status)),
                                "Model saved: {:5} ".format(str(save_status)))

                else:
                    if verbose:
                        print (current_time(),
                              "Epoch: {:4}/{:4} |".format(e, epochs),
                              "Train Loss :{:.5f} |".format(train_loss/len(self.loaders['train'].dataset)),
                              "Valid Loss: {:.5f} (best: {:.5f}) |".format(valid_loss/len(self.loaders['valid'].dataset),self.valid_loss_min),
                              "Valid Loss dec: {:5} |".format(str(validation_decrease_status)),
                              "Model saved: {:5} ".format(str(save_status)))

            else:
                if verbose == 'epoch' :
                    if e % print_every == 0:
                        print (current_time(),
                                "Epoch: {:4}/{:4} |".format(e, epochs),
                                "Train Loss: {:.5f} |".format(train_loss/len(self.loaders['train'].dataset)))

                else:
                    if verbose:
                        print (current_time(),
                              "Epoch: {:4}/{:4} |".format(e, epochs),
                              "Train Loss :{:.5f} |".format(train_loss/len(self.loaders['train'].dataset)))

            if e+1 == epochs:
                print (current_time(), 'Training Finished Successfully!')

        self.train_logs=pd.DataFrame({"train": self.train_losses, "valid" : self.valid_losses})

    def weight_init(self):
        if self.classifier_type == 'torch':
            self.model.apply(init_w)
            print(current_time(), 'Model weight initialization applied successfully.')
        else:
            raise ValueError('Weight initialization not available with sklearn classifiers.')

    def view_train_logs(self, data='all', figsize=(12,8)):
        if self.classifier_type == 'torch':
            plt.figure(figsize=figsize)
            sns.set_style("darkgrid")
            if data == 'all':
                p = sns.lineplot(data = self.train_logs)
            else:
                p = sns.lineplot(data = self.train_logs[data].tolist())

            p.set_xlabel("epoch", fontsize = 10)
            p.set_ylabel("loss", fontsize = 10);

        else:
            raise ValueError('Train Logs not available with sklearn classifiers.')

    def export(self, path):
        try:
            torch.save(self, path)
            print (current_time(), 'Export done successfully.')
        except:
            raise ValueError(current_time(), 'Cannot Export.')
