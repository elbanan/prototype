from .utils import *


class Inference():
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
