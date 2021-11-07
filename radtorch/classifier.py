from .utils import *
from .metrics import *
from .inference import *

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

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


# FIX uid with upsample and downsample
# metrics for all sklearn and nn models



# self mean/std
# resume training
# set random seed
# T-Sne visualization
#select only certain classes
