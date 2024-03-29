# 12/19/2021

from ..core.utils import *


class Inference():
    """
    Container class that contains all inference methods related to making predictions/inference using a trained model.


    Parameters
    ----------

    classifier : radtorch classifier object
        a trained classifier object. Please see radtorch.classification.classifier

    specific_transform : pytorch nn transforms
        List of specific pytorch transforms to be applied to images for inference.
        If no specific tranforms are selected, the Inference class will try to use the specified `test` transforms of the dataset used for the classifier training.
        If `test` transforms are not available, `train` transforms will be used instead.
        if the model of the trained Classifier is an sklearn model, the Inference class will by default use the `train` transforms of the training dataset.

    """

    def __init__(self, classifier=False, specific_transform=False):

        self.classifier = classifier
        self.dataset = self.classifier.dataset

        try:
            self.transforms = self.dataset.transforms['test']
        except:
            self.transforms = self.dataset.transforms['train']

        if self.classifier.classifier_type == 'sklearn':
            self.feature_extractor = self.classifier.feature_extractors['train']

        self.model = self.classifier.best_model

        if specific_transform:
            self.transforms=specific_transform

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def predict(self, img_path, top_predictions='all', display_image=False, human=True):
        """
        Method to predict class/label of an image used a trained classifier.

        Parameters
        ----------

        img_path : str
            path to image to perfeorm inference operation on. Image type must be similar to image type used in original training set i.e. DICOM or non DICOM.

        top_predictions : int
            Number of top predictions to be returned. default='all'.

        display_image : boolean
            Either to display the image or not. default=False.

        human : boolean
            Display the predictions in human readable format. default=True.


        Returns
        -------
        Only if human parameter = False.
        This method will return a a list of predictions for each image supplied.
        The predictions for each image will be in the form of list of dictionaries of each predictions as follows:
        [
            [ {'id':image1 id, 'class': class1 text, 'class__id':class1 id, 'prob': probability float},
              {'id':image1 id, 'class': class2 text, 'class__id':class2 id, 'prob': probability float},...
            ],
            [ {'id':image2 id, 'class': class1 text, 'class__id':class1 id, 'prob': probability float},
              {'id':image2 id, 'class': class2 text, 'class__id':class2 id, 'prob': probability float},...
        ]

        """

        if top_predictions == 'all':
            top = len(self.dataset.classes)
        else:
            top = top_predictions
            if top > len(self.dataset.classes):
                raise ValueError('Number of top predictions is more than number of classes. Please check')


        name, ext = os.path.splitext(img_path)
        if ext != '.dcm':
            img=Image.open(img_path)
        else:
            img = dicom_handler(img_path=img_path, num_output_channels=self.dataset.num_output_channels, WW=self.dataset.window, WL=self.dataset.level)

        img = self.transforms(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)

        if self.classifier.classifier_type == 'sklearn':
            with torch.no_grad():
                self.feature_extractor.model.eval()
                nn_output = self.feature_extractor.model(img)
                img_features = pd.DataFrame(nn_output.cpu().numpy())
                raw_pred = self.classifier.best_model.predict_proba(img_features)

        elif self.classifier.classifier_type == 'torch':
            with torch.no_grad():
                self.model.eval()
                nn_output = self.model(img)
                raw_pred = torch.nn.functional.softmax(nn_output, dim=1).cpu().numpy()

        predictions = []
        s=0
        for i in raw_pred:
            o = []
            for k, v in self.dataset.class_to_idx.items():
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
        else:
            return predictions
