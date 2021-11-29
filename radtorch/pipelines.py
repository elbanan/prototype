from .core import *


class ImageClassificationPipeline():
    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in image_classification_pipe_allowed_keys)
        self.dataset = DICOMFileDataset(**{k:v for k, v in self.__dict__.items() if k in dataset_allowed_keys})
        self.classifier = ImageClassifier(**{k:v for k, v in self.__dict__.items() if k in classifier_allowed_keys})

    def fit(self, **kwargs):
        self.classifier.fit(**kwargs)
        self.metrics = Metrics(classifier=self.classifier, use_best=True)
        self.inference = Inference(classifier=self.classifier)

    def export(self, output_file):
        torch.save(self, output_file)

class CompareClassifiers(): #In Progress
    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in image_classification_pipe_allowed_keys)
        self.dataset = DICOMFileDataset(**{k:v for k, v in self.__dict__.items() if k in dataset_allowed_keys})
        self.classifiers = []
        sk_models = [{'model':i[0], 'feature_extractor_arch':i[1], 'type':'sklearn'} for i in list(itertools.product([i for i in self.model if str(type(i))[8:].startswith('sklearn')], self.feature_extractor_arch))]
        nn_models = [{'model':i, 'type':'nn'} for i in [i for i in self.model if not str(type(i))[8:].startswith('sklearn')]]

        self.classifier_table = pd.DataFrame((sk_models+nn_models))

        print(current_time(), 'Creating Classifier Objects List.')
        for i, r in tqdm (self.classifier_table.iterrows(), total=len(self.classifier_table)):
            args = {k:v for k, v in self.__dict__.items() if k in classifier_allowed_keys and k not in ['model', 'feature_extractor_arch'] }
            args['model'] = r['model']
            args['feature_extractor_arch'] = r['feature_extractor_arch']
            self.classifiers.append(ImageClassifier(**args))

    def fit(self, **kwargs):
        print(current_time(), 'Training Classifiers.')
        self.metrics = []
        self.inference = []
        for c in tqdm(self.classifiers, total=len(self.classifiers)):
            c.fit(**kwargs)
            self.metrics.append(Metrics(classifier=c, use_best=True))
            self.inference.append(Inference(classifier=c))
            print ('---')

    def find_best(self, roc_auc=True):
        self.roc_auc_list = []
        if roc_auc:

            for i in tqdm(self.metrics, total=len(self.metrics)):
                print(current_time(), 'Calculating ROC AUC for trained models.' )
                i.roc(plot=False)
                self.roc_auc_list.append(i.auc)
        self.best_classifier_id = self.roc_auc_list.index(max(self.roc_auc_list))
        print(current_time(), 'Best Model id:', self.best_classifier_id )
        print(current_time(), 'Best Model:', self.classifier_table.iloc[self.best_classifier_id]['model'], 'with', self.classifier_table.iloc[self.best_classifier_id]['feature_extractor_arch'] )
        print(current_time(), 'Best Model ROC AUC:', max(self.roc_auc_list))





## TO DO LIST
# 1. resume training checkpoints
# 2. CompareClassifiers
# 3. GAN
