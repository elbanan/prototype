from .core import *


class ImageClassificationPipeline():
    def __init__(self, root, model, model_arch=None,
        ext='dcm', \
        label_table=None, path_col='img_path', label_col='img_label', \
        num_output_channels=1,transform=None, WW=None, WL=None, \
        split=None, \
        ignore_zero_img=False, sample=False, train_balance=False, batch_size=16, output_subset='all', optimizer=None, criterion = None, ):

        self.dataset = DICOMDataset(root=root, ext=ext, \
        label_table=label_table, path_col=path_col, \
        label_col=label_col, num_output_channels=num_output_channels, \
        transform=transform, WW=WW, WL=WL, split=split, \
        ignore_zero_img=ignore_zero_img, sample=sample, train_balance=train_balance, batch_size=batch_size, output_subset=output_subset)

        self.classifier = ImageClassifier(model, dataset=self.dataset, feature_extractor_arch=model_arch, criterion=criterion, optimizer=optimizer)


    def fit(self, **kwargs):
        self.classifier.fit(**kwargs)
        self.metrics = Metrics(classifier=self.classifier, use_best=True)
        self.predictor = Inference(classifier=self.classifier)

    def export(self, output_file):
        torch.save(self, output_file)
