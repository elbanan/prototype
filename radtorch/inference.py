from .utils import *

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

class Inference():
    def __init__(self, classifier=False, feature_extractor=False, mode='non-sklearn', specific_transform=False):
        self.mode = mode
        if self.mode == 'non-sklearn':
            self.classifier = classifier
            self.model = self.classifier.best_model
            self.dataset = self.classifier.loaders['test'].dataset
            self.transforms = self.ds.transforms

        elif self.mode == 'sklearn':
            self.classifier = classifier
            self.feature_extractor = feature_extractor
            self.model = self.feature_extractor.model
            self.dataset = self.feature_extractor.dataset
            self.transforms = self.ds.transforms['train']

        if specific_transform:self.transforms=specific_transform
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def predict(self, img_path, top_predictions='all', display_image=False, human=True):
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
            img = dicom_handler(img_path=img_path, modality=self.dataset.modality, num_output_channels=self.dataset.num_output_channels, w=self.dataset.window, l=self.dataset.level)

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


        return predictions




# FIX uid with upsample and downsample
# metrics for all sklearn and nn models



# self mean/std
# resume training
# set random seed
# T-Sne visualization
#select only certain classes
