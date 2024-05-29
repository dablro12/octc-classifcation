import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd 
import os
import torch 
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score

def plot_combined_roc_curves(models_data):
    """
    Plots ROC curves for multiple models on the same plot.
    
    Parameters:
    models_data (list of tuples): Each tuple should contain (true_labels, predicted_scores, label) for a model.
    """
    plt.figure(figsize=(8, 6))
    
    for true_labels, predicted_scores, label in models_data:
        # Ensure predicted_scores is a numpy array
        predicted_scores = np.array(predicted_scores)
        
        fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves')
    plt.legend(loc='lower right')
    plt.show()
    # To save the plot, uncomment the following line and adjust the filename as needed.
    # plt.savefig("path/to/save/roc_curve_combined.png")
    plt.close()
    
def metric_abstract(total, correct, labels, outputs,pred_labels, pred_scores, true_labels):
    pred = (torch.sigmoid(outputs) >0.5).float()
    total += labels.size(0)
    correct += (pred == labels).sum().item()
    true_labels.extend(labels.cpu().numpy())
    pred_labels.extend(pred.cpu().numpy())
    pred_scores.extend(torch.sigmoid(outputs).cpu().numpy())
    
    return true_labels, pred_labels, pred_scores

def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def npv_score(y_true, y_pred):
    tn, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0

def metric_func(y_true, y_pred, y_score, metrics_dict):
    metrics_dict['acc'].append(accuracy_score(y_true, y_pred))
    metrics_dict['f1_score'].append(f1_score(y_true, y_pred, zero_division=1))
    metrics_dict['recall'].append(recall_score(y_true, y_pred, zero_division=1))
    metrics_dict['specificity'].append(specificity_score(y_true, y_pred))
    metrics_dict['ppv'].append(precision_score(y_true, y_pred, zero_division=1))  # ppv is the same as precision
    metrics_dict['npv'].append(npv_score(y_true, y_pred))
    metrics_dict['roc_auc_score'].append(roc_auc_score(y_true, y_score))
    return metrics_dict

def save_metric(metrics_dict, save_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Convert the dictionary into a DataFrame for easy CSV saving
    df = pd.DataFrame.from_dict(metrics_dict)
    df.to_csv(save_path, index=False)