from ..core import *
from ..data import *
from ..classification import *



class ImageClassificationPipeline():
    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in image_classification_pipe_allowed_keys)
        self.dataset = DICOMDataset(**{k:v for k, v in self.__dict__.items() if k in dataset_allowed_keys})
        self.classifier = ImageClassifier(**{k:v for k, v in self.__dict__.items() if k in classifier_allowed_keys})

    def fit(self, **kwargs):
        self.classifier.fit(**kwargs)
        self.metrics = Metrics(classifier=self.classifier, use_best=True)
        self.inference = Inference(classifier=self.classifier)

    def export(self, output_file):
        torch.save(self, output_file)
