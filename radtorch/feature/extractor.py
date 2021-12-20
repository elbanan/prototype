# 12/19/2021

from ..core.utils import *


class FeatureExtractor():
    """
    Container class for extraction of features from image dataset using pretrained pytorch neural network models.

    Parameters
    ----------
    model_arch: str
        the architecture of the neural network model to be used e.g. vgg16. For supported models, please see radtorch.core.const.supported_models

    dataset: radtorch dataset object

    subset: str
        the subset of the dataset to perform feature extraction on. default='train'.


    Attributes
    ----------
    model: pytorch nn object
        the model that will be used for feature extraction. The last layers of these models have been modfied to remove the classifier portion of the code.

    loader: pytorch dataloader
        dataloader object used for feature extraction.

    features: dataframe
        table of extracted features.

    feature_names: list
        list of extracted feature names.

    """

    def __init__(self, model_arch, dataset, subset='train'):
        self.dataset = dataset
        self.subset = subset
        self.model_arch = model_arch
        if self.model_arch not in supported_models:
            raise ValueError('Model not yet Supported. For list of supported models please use supported_models. Thanks')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = grab_pytorch_model(model_arch=self.model_arch, pretrained=True)
        self.model = remove_last_layers(model=self.model, model_arch=self.model_arch)
        self.model = self.model.to(self.device)
        self.loader = self.dataset.loaders[self.subset]


    def run(self,):
        """
        Runs the feature extraction process.
        """

        with torch.no_grad():
            self.model.eval();
            print(current_time(), 'Starting Feature extraction of subset =', self.subset, 'using model architecture =', self.model_arch)
            for i, (uid, images, labels) in tqdm(enumerate(self.loader), total=len(self.loader)):
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
        print(current_time(), 'Feature extraction completed successfully.')


    def num_features(self):
        """
        Returns the number of features to be extracted.
        """

        return self.features.shape[1]

    def plot_features(self, annotations=False, figure_size=(10,10), cmap="YlGnBu"):
        """
        Plots a matrix of the extracted features.

        Parameters
        ----------

        annotations: boolean
            To display values of features on the plot. default=False.

        figure_size: tuple
            size of the displayed graph. default=(10,10)

        cmap: str
            color map for the generated features. default = 'YlGnBu'. Please refer to https://matplotlib.org/stable/tutorials/colors/colormaps.html
        """

        plt.figure(figsize=figure_size)
        plt.title("Extracted Features")
        sns.heatmap(data=self.features, annot=annotations,cmap=cmap);

    def model_info(self):
        """
        Returns information of the pytorch nn model used for feature extraction.
        """

        batch  = (next(iter(self.loaders['train'])))[1]
        batch_size, channels, img_dim, img_dim = batch.shape
        return model_info(self.model, list=False, batch_size=batch_size, channels=channels, img_dim=img_dim)

    def hybrid_table(self, sklearn_ready=False, label_id=True):
        """
        Method to generate a custom table that can be used directly into sklearn classifier i.e. table of features and labels only.

        Parameters
        ----------

        sklearn_ready: boolean
            True to generate an sklearn ready table. False to only add the label column to the native feature table.

        label_id: boolean
            determines if either label id or label name will be used in the label column of the hybrid table. True uses label_id not the class name.

        Returns
        -------
        pandas dataframe

        """

        h = pd.merge(self.features, self.dataset.data_table[self.subset], on='uid')
        h['label_id'] = [self.dataset.class_to_idx[r[self.dataset.label_col]] for i, r in h.iterrows()]
        if sklearn_ready:
            if label_id:
                f, l = self.features[self.feature_names], h['label_id']
            else:
                f, l = self.features[self.feature_names], h[self.dataset.label_col]
            return f, l
        else:
            return h
