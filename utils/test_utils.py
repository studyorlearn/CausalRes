import time
from matplotlib import pyplot as plt
import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, cohen_kappa_score, 
                             confusion_matrix, mean_absolute_error, 
                             matthews_corrcoef)

def get_cost_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print('[INFO] The running time is: %ss\n' % (end - start))
        return res
    return wrapper


class Tester():
    def __init__(self, model, epoch):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("The device is: {}".format(self.device))
        self.model = model.to(self.device)
        self.epoch = epoch
        print("The tester initialization is completed.")
        self.model_name = model._get_name()
        self.flatten = True if self.model_name in ['MLP', ] else False

    @get_cost_time
    def train(self, train_loader):
        
        self.lossFn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.train_losses = []
        self.train_accuracy = []
        self.val_losses = []
        self.val_accuracy = []
        
        trainSteps = len(train_loader.dataset)

        for e in range(self.epoch):
            
            self.model.train()
            totalTrainLoss, trainCorrect, trainAcc = 0, 0, 0

            for (x, y) in train_loader:
                (x, y) = (x.to(self.device), y.to(self.device))
                
                if self.flatten:
                    batch_size = x.shape[0]
                    x = x.permute(0, 2, 1).contiguous()
                    x = x.reshape(batch_size, -1)
                
                pred = self.model(x)
                loss = self.lossFn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                totalTrainLoss += loss
                trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()     
         
            avgTrainLoss = totalTrainLoss / trainSteps
            trainAcc = trainCorrect /trainSteps
            self.train_losses.append(avgTrainLoss)
            self.train_accuracy.append(trainAcc)

            if((e + 1) % 2 >= 0):
                print("[INFO] EPOCH: {}/{}".format(e +1, self.epoch))
                print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainAcc))
    
    @get_cost_time            
    def test(self, test_loader):
        correct, total = 0, 0
        self.preds = []
        self.all_test_label = []
        self.test_losses = 0
        self.test_accuracy = 0
        with torch.no_grad():
            self.model.eval()
            for inputs, labels in test_loader:               
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.flatten:
                    batch_size = inputs.shape[0]
                    inputs = inputs.permute(0, 2, 1).contiguous()
                    inputs = inputs.reshape(batch_size, -1)
                       
                pred = self.model(inputs)
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
                self.preds.extend(pred.argmax(axis=1).cpu().numpy())
                total += labels.size(0)
                self.all_test_label.extend(labels.cpu())
                self.test_losses += self.lossFn(pred, labels)
                
            self.test_accuracy = 100 * correct / total
            print("[INFO] Test loss: {:.6f}, Test accuracy: {:.4f}".format(self.test_losses/len(self.preds), self.test_accuracy))
            
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def calculate_performance_metrice(self):
        self.metrics = {
            'Accuracy': accuracy_score(self.all_test_label, self.preds),
            'F1-score': f1_score(self.all_test_label, self.preds, average='macro'),
            'Kappa': cohen_kappa_score(self.all_test_label, self.preds),
            'MAE': mean_absolute_error(self.all_test_label, self.preds),
            'Precision': precision_score(self.all_test_label, self.preds, average='macro'),
            'Recall': recall_score(self.all_test_label, self.preds, average='macro'),
            'Specificity': calculate_specificity(self.all_test_label, self.preds),
            'Confusion Matrix': confusion_matrix(self.all_test_label, self.preds),
        }
    
    def plot_loss(self, ):
        plt.figure(dpi=600, figsize=[10, 5])
        plt.plot(range(1, self.epoch + 1), self.train_losses)
        plt.plot(range(1, self.epoch + 1), self.val_losses)
        plt.title(f"Train and Validation Loss ({self.model_name})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(["Train", "Validation"])
        plt.show()
    
    def plot_accuracy(self):
        plt.figure(dpi=600, figsize=[10, 5])
        plt.plot(range(1, self.epoch + 1), self.train_accuracy)
        plt.title(f"Train Accuracy ({self.model_name})")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show()
        
    def plot_confusion_matrix(self):
        cm = np.array(self.metrics['Confusion Matrix'])
        cm_log = np.log1p(cm)
        plt.figure(dpi=600, figsize=[8, 6])
        plt.imshow(cm_log, cmap=plt.cm.Blues)
        plt.colorbar()
        plt.title(f"Confusion Matrix (log scale, {self.model_name})")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.show()
        
        
def calculate_specificity(y_true, y_pred):
    """Calculate the specificity of a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    specificity = tn / (tn + fp)
    return specificity
