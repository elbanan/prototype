from .core import *


class ImageClassificationPipeline():
    def __init__(self, **kwargs):

        pipeline_allowed_keys = ['root', 'model', 'model_arch','ext','label_table','path_col','label_col','num_output_channels','transform','WW','WL','split','ignore_zero_img', 'sample', 'train_balance','batch_size', 'output_subset', 'optimizer', 'criterion','device']
        dataset_allowed_keys = ['root', 'ext','label_table','path_col','label_col','num_output_channels','transform','WW','WL','split','ignore_zero_img', 'sample', 'train_balance','batch_size', 'output_subset']
        classifier_allowed_keys = ['model', 'dataset', 'feature_extractor_arch', 'criterion', 'optimizer', 'device']

        self.__dict__.update((k, v) for k, v in kwargs.items() if k in pipeline_allowed_keys)
        self.dataset = DICOMDataset(**{k:v for k, v in self.__dict__.items() if k in dataset_allowed_keys})
        self.classifier = ImageClassifier(**{k:v for k, v in self.__dict__.items() if k in classifier_allowed_keys})


    def fit(self, **kwargs):
        self.classifier.fit(**kwargs)
        self.metrics = Metrics(classifier=self.classifier, use_best=True)
        self.predictor = Inference(classifier=self.classifier)

    def export(self, output_file):
        torch.save(self, output_file)
