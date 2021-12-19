# 12/19/2021

from ..core.utils import *
from ..core.const import *
from .metrics import *
from .inference import *
from ..feature import *


class ImageClassifier():
    """
    Container class to create/train a pytorch neural network image classifer or a combined pytorch neural network feature extractor with an sklearn classifier model.

    Parameters
    ----------
    model : sklearn model or pytorch nn
        the classifer model object that will be trained.

    dataset : radtorch dataset object
        the dataset object created using radtorch.data.DICOMDataset.

    device: str
        device that will be used for training. Can be 'cpu' or 'cuda'. default 'auto' to automaically detect and use 'cuda' if available.

    feature_extractor_arch: str
        the architecture of the feature extractor that will be used to extract imaging features if using an sklearn classifier model. This value will be ignored if training using a pytorch nn model.

    criterion: torch nn
        loss function/criterion used for training. Only when training a pytorch nn model, otherwise ignored.

    optimizer: torch nn
        optimizer used for training. Only when training a pytorch nn model, otherwise ignored.


    Attributes
    ----------
    device : str
        device used for training.

    valid_loss_min : float
        best/lowest validation loss achieved during training.

    train_logs: pandas dataframe
        dataframe containing trainig and validation loss.

    best_model: pytorch nn or sklearn classifier model.
        the best trained model achieved during training.
    """

    def __init__(self, model, dataset, device='auto', feature_extractor_arch='vgg16', criterion=None, optimizer=None):
        self.dataset = dataset
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = select_device(device)

        if str(type(model))[8:].startswith('sklearn'):
            self.classifier_type, self.feature_extractors = 'sklearn', {i:FeatureExtractor(model_arch=feature_extractor_arch, dataset=dataset, subset=i) for i in ['train', 'valid', 'test']}
        else:
            self.classifier_type = 'torch'

    def info(self):
        """
        Returns class relevant information/parameters
        """
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        for i in ['train', 'valid','test']:
            try:
                info.loc[len(info.index)] = [i+' dataset size', len(self.dataset.loaders[i].dataset)]
            except:
                pass
        return info

    def fit(self, **kwargs):
        """
        Method to train the image classifier object.

        Parameters
        ----------
        epochs : int
            number of training epochs. default= 20. Only when training a pytorch nn model, otherwise ignored.

        valid : boolean
            determine validation step during training or not. default= True. Only when training a pytorch nn model, otherwise ignored.

        print_every : int
            number of epochs to print training verbose. default=1. Only when training a pytorch nn model, otherwise ignored.

            Explaining the verbose:

            [19-12-2021 17:14:31] epoch:   0/  50 | t_loss: 0.80455 | v_loss: 0.82266 (best: 0.82266) | v_loss dec: True  | v_loss below target: False | model saved: False

            [19-12-2021 17:14:31]: timestamp
            epoch: current epoch/total epochs
            t_loss: training loss
            v_loss: validation loss (best achieved validation loss)
            v_loss dec: True if validation loss is decreasing
            v_loss below target: True if validation loss is below specified target value.
            model saved: True if saved.

        target_valid_loss: float
            target validation loss below which the best model will be saved automatically. Every time the model's validation loss the saved model will be overriden. default = 'lowest' which automatically saves the trained model with lowest validation loss.  Only when training a pytorch nn model, otherwise ignored.

        output_model: str
            path and name of output trained model to be saved. By default, for pytorch nn models, the best model with lowest validation loss will be selected as best. default='best_model.pt'.
        """

        if self.classifier_type == 'torch':
            train_nn(self, **{k:v for k, v in kwargs.items() if k in ['epochs', 'valid', 'print_every', 'target_valid_loss', 'output_model' ]})

        elif self.classifier_type == 'sklearn':
            if hasattr(self, 'train_features'):
                print (current_time(), 'Using pre-extracted training features.')
            else:
                print (current_time(), 'Running Feature Extraction using model architecture', self.feature_extractors['train'].model_arch)
                self.feature_extractors['train'].run()
                self.train_features, self.train_labels = self.feature_extractors['train'].hybrid_table(sklearn_ready=True)
            train_sklearn(self, **{k:v for k, v in kwargs.items() if k in ['output_model']})

    def view_train_logs(self, data='all', figsize=(12,8)):
        """
        Method to display training logs (training and validation loss during training process)

        Parameters
        ----------

        data: str
            type of data to be displayed. default='all'. can be 'all', 'train', or 'valid'.

        figsize: tuple
            size of the displayed graph. default=(12,8)
        """

        if self.classifier_type == 'torch':
            plt.figure(figsize=figsize)
            sns.set_style("darkgrid")
            if data == 'all': p = sns.lineplot(data = self.train_logs)
            else: p = sns.lineplot(data = self.train_logs[data].tolist())
            p.set_xlabel("epoch", fontsize = 10)
            p.set_ylabel("loss", fontsize = 10);
        else:
            raise ValueError('Train Logs not available with sklearn classifiers.')

    def export(self, output_file):
        """
        Exports the whole classifier object for later use.

        Parameters
        ----------

        output_file: str
            name and path of the saved output file.
        """

        try:
            torch.save(self, output_file)
            print (current_time(), 'Export done successfully.')
        except:
            raise ValueError(current_time(), 'Cannot Export.')
