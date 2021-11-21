from .utils import *

class Metrics():

    def __init__(self, classifier, use_best=True):

        self.classifier = classifier
        self.use_best = use_best

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.auc='Please run metrics.roc() to generate roc auc.'
        self.accuracy = 'Please run metrics.test() to generate accuracy.'


        if self.classifier.classifier_type == 'torch':
            if self.use_best:
                self.selected_model = self.classifier.best_model
            else:
                self.selected_model = self.classifier.training_model
        else:
            self.selected_model = self.classifier.feature_extractors['train'].model


    def get_predictions(self, target_loader, raw=False):
        true_labels = []
        pred_labels = []

        if self.classifier.classifier_type == 'torch':
            if self.use_best:
                self.selected_model = self.classifier.best_model
            else:
                self.selected_model = self.classifier.training_model
        else:
            self.selected_model = self.classifier.feature_extractors['train'].model

        for i, (idx, imgs, labels) in tqdm(enumerate(self.classifier.dataset.loaders[target_loader]), total=len(self.classifier.dataset.loaders[target_loader])):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            true_labels = true_labels+labels.tolist()

            with torch.no_grad():
                self.selected_model.to(self.device)
                self.selected_model.eval()
                out = self.selected_model(imgs)
                if self.classifier.classifier_type == 'torch':
                    if raw:
                        pr = [i.tolist() for i in out.cpu().numpy()]
                    else:
                        pr = [(i.tolist()).index(max(i.tolist())) for i in out.cpu().numpy()]
                elif self.classifier.classifier_type == 'sklearn':
                    if raw:
                        pr = [i.tolist() for i in self.classifier.best_model.predict_proba(out.cpu().numpy())]
                    else:
                        pr = [(i.tolist()).index(max(i.tolist())) for i in self.classifier.best_model.predict_proba(out.cpu().numpy())]

                pred_labels = pred_labels+pr

        return true_labels, pred_labels


    def test(self, target_loader='test'):
        test_loss = 0.0
        class_correct = list(0. for i in range(len(self.classifier.dataset.class_to_idx.keys())))
        class_total = list(0. for i in range(len(self.classifier.dataset.class_to_idx.keys())))

        if self.classifier.classifier_type == 'torch':
            with torch.no_grad():
                self.selected_model.eval()
                for idx, imgs, labels in self.classifier.dataset.loaders[target_loader]:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    output = self.selected_model(imgs)
                    loss = self.classifier.criterion(output, labels)
                    test_loss += loss.item()*imgs.size(0)
                    _, pred = torch.max(output, 1)
                    correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
                    for i in range(imgs.shape[0]):
                        label = labels.data[i]
                        class_correct[label] += correct[i].item()
                        class_total[label] += 1

                test_loss = test_loss/len(self.classifier.dataset.loaders['test'].dataset)
                print('Overall Test Loss: {:.6f}\n'.format(test_loss))


        elif  self.classifier.classifier_type == 'sklearn':
            true_labels, predictions = self.get_predictions(target_loader=target_loader)
            true_labels = torch.LongTensor(true_labels)
            predictions = torch.FloatTensor(predictions)
            correct = np.squeeze(predictions.eq(true_labels.data.view_as(predictions)))
            for i in range(len(true_labels)):
                label = true_labels.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1


        for i in range(len(self.classifier.dataset.class_to_idx)):
            c = next((k for k, v in self.classifier.dataset.class_to_idx.items() if v == i), None)
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    c , 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))

            else:
                print('Test Accuracy of %5s: N/A (no examples)' % (c))

        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))


    def confusion_matrix(self, target_loader='test', figure_size=(8,6), cmap='Blues', percent=False): #https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
        true_labels, pred_labels = self.get_predictions(target_loader=target_loader)
        cm = metrics.confusion_matrix(true_labels, pred_labels)
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
        if len(cm)==2:
            precision = cm[1,1] / sum(cm[:,1])
            recall    = cm[1,1] / sum(cm[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}\nMisclassified={:}".format(accuracy, misclass)
        plt.figure(figsize=figure_size)
        sns.set_style("darkgrid")
        if percent:
            sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap=cmap, xticklabels=self.classifier.dataset.class_to_idx.keys() ,yticklabels=self.classifier.dataset.class_to_idx.keys(), linewidths=1, linecolor='black')
        else:
            sns.heatmap(cm, annot=True, cmap=cmap, xticklabels=self.classifier.dataset.class_to_idx.keys(),yticklabels=self.classifier.dataset.class_to_idx.keys(),  linewidths=1, linecolor='black')
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
        plt.title('Confusion Matrix', fontweight='bold')


    def roc(self, target_loader= 'test', figure_size=(8,6)):
        true_labels, pred_labels = self.get_predictions(target_loader=target_loader)
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_labels)
        auc = metrics.roc_auc_score(true_labels, pred_labels)
        sns.set_style("darkgrid")
        fig = plt.figure(figsize=figure_size)
        plt.plot([0, 0.5, 1.0], [0, 0.5, 1.0], linestyle=':')
        plt.plot(fpr, tpr, linestyle='--', lw=1.1,  label = "ROC AUC = {:0.3f}".format(auc))
        plt.xlabel('False Positive Rate (1-Specificity)')
        plt.ylabel('True Positive Rate (Senstivity)')
        plt.title('Receiver Operating Characteristic Curve',y=-0.2 , fontweight='bold')
        plt.legend()
        plt.show()
        self.auc = auc

    def all(self):
        self.test()
        self.confusion_matrix()
        self.roc()
