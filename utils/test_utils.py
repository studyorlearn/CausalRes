import json
import time

import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, cohen_kappa_score, 
                             confusion_matrix, mean_absolute_error, 
                             matthews_corrcoef)

def get_cost_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        cost_time = end - start
        print('[INFO] The running time is: %ss\n' % cost_time)
        return cost_time
    return wrapper


class Tester():
    def __init__(self, model, epoch):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("The device is: {}".format(self.device))
        self.model = model.to(self.device)
        self.epoch = epoch
        print("The tester initialization is completed.\n")
        self.model_name = model._get_name()
        self.flatten = True if self.model_name in ['MLP', ] else False
        self.train_time_cost = 0
        self.test_time_cost = 0

    @get_cost_time
    def train(self, train_loader):
        print("The model is: {}.".format(self.model_name), "Waitting...")
        self.lossFn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.train_losses = []
        self.train_accuracy = []
        
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
         
            avgTrainLoss = totalTrainLoss.item() / trainSteps
            trainAcc = trainCorrect /trainSteps
            self.train_losses.append(avgTrainLoss)
            self.train_accuracy.append(trainAcc)

            if((e + 1) % 10 == 0):
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
            self.test_losses = self.test_losses / len(self.preds)    
            self.test_accuracy = 100 * correct / total
            print("[INFO] Test loss: {:.6f}, Test accuracy: {:.4f}".format(self.test_losses, self.test_accuracy))
    
    
    
    def save_model(self, num, tag="low_probability"):
        self.num = num
        self.tag = tag
        path = f"/root/CausalRes/results/pths/model_{self.tag}_{self.model_name}_{self.num}"
        torch.save(self.model.state_dict(), path)
    
    def calculate_performance_metrice(self):
        self.metrics = {
            'model_name': self.model_name,
            'epoch': self.epoch,
            'data_info': self.tag,
            'loop_counter': self.num,
            
            'Accuracy': accuracy_score(self.all_test_label, self.preds),
            'F1-score': f1_score(self.all_test_label, self.preds, average='macro'),
            'Kappa': cohen_kappa_score(self.all_test_label, self.preds),
            'MAE': mean_absolute_error(self.all_test_label, self.preds),
            'Precision': precision_score(self.all_test_label, self.preds, average='macro'),
            'Recall': recall_score(self.all_test_label, self.preds, average='macro'),
            'Specificity': calculate_specificity(self.all_test_label, self.preds),
            'Confusion Matrix': confusion_matrix(self.all_test_label, self.preds).tolist(),
            
            'traing_time': self.train_time_cost,
            'test_time': self.test_time_cost,
            'train_accuracy': self.train_accuracy,
            'test_accuracy': self.test_accuracy,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses.item(),     
        }
    
    def save_metrics_to_json(self):
        filename = f"/root/CausalRes/results/jsons/model_{self.tag}_{self.model_name}_metrics.json"
        try:
            with open(filename, 'r') as f:
                all_metrics = json.load(f)
                if self.num==1:
                    all_metrics = []
        except FileNotFoundError:
            all_metrics = []

        all_metrics.append(self.metrics)
        with open(filename, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Save metrics to model_{self.model_name}_{self.num}_metrics.json file successfully!\n")
    
    def save_predict_and_label(self):
        filename = f"/root/CausalRes/results/predict_labels/model_{self.tag}_{self.model_name}_{self.num}_predict_and_label.npy"
        res = np.array([self.preds, self.all_test_label])
        np.save(filename, res)
        print(f"Save predict and label to model_{self.model_name}_{self.num}_predict_and_label.npy file successfully!\n")
        
    
    def plot_loss(self, ):
        plt.figure(dpi=600, figsize=[10, 5])
        plt.plot(range(1, self.epoch + 1), self.train_losses)
        plt.title(f"Train Loss ({self.model_name})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
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
        
def calculate_macro_specificity(conf_matrix):
    num_classes = conf_matrix.shape[0]
    specificities = np.zeros(num_classes)   
    for i in range(num_classes):
        true_negatives = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
        false_positives = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
        if (true_negatives + false_positives) > 0:
            specificities[i] = true_negatives / (true_negatives + false_positives)
        else:
            specificities[i] = 0.0
    macro_specificity = np.mean(specificities)
    
    return macro_specificity
