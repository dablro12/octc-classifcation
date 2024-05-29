import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd 
import os
import torch 
import seaborn as sns
import seaborn
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
from scipy import stats 
# def plot_combined_roc_curves(models_data):
#     """
#     Plots ROC curves for multiple models on the same plot.
    
#     Parameters:
#     models_data (list of tuples): Each tuple should contain (true_labels, predicted_scores, label) for a model.
#     """
#     plt.figure(figsize=(8, 6))
    
#     for true_labels, predicted_scores, label in models_data:
#         # Ensure predicted_scores is a numpy array
#         predicted_scores = np.array(predicted_scores)
        
#         fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
#         roc_auc = auc(fpr, tpr)
        
#         plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    
#     plt.plot([0, 1], [0, 1], 'k--', label='Chance')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Combined ROC Curves')
#     plt.legend(loc='lower right')
#     plt.show()
#     # To save the plot, uncomment the following line and adjust the filename as needed.
    # plt.savefig("/home/eiden/eiden/octc-classification/models/binary_metric/roc_curve_combined.png")
#     plt.close()

def plot_combined_roc_curves(models_data):
    """
    Plots ROC curves for multiple models on the same plot using Seaborn styles and marks the best threshold,
    without adding the best threshold markers to the legend.
    
    Parameters:
    models_data (list of tuples): Each tuple should contain (true_labels, predicted_scores, label) for a model.
    """
    sns.set(style="whitegrid")  # Seaborn 스타일 설정
    plt.figure(figsize=(8, 6))
    
    for true_labels, predicted_scores, label in models_data:
        predicted_scores = np.array(predicted_scores)
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
        roc_auc = auc(fpr, tpr)
        
        # Youden's J statistic을 계산하여 최적의 임계값을 찾습니다.
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr, optimal_tpr = fpr[optimal_idx], tpr[optimal_idx]
        
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')
        # 최적의 지점에 마커 추가하지만, 범례에는 포함시키지 않습니다.
        plt.plot(optimal_fpr, optimal_tpr, marker='o', color='red', markersize=8)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    sns.despine()  # 그래프의 상단 및 오른쪽 축을 제거
    plt.show()
    # 그래프 저장
    plt.savefig("/home/eiden/eiden/octc-classification/models/binary_metric/roc_curve_combined_seaborn.png")
    plt.close()

def calculate_youden_index(true_labels, predicted_scores):
    true_labels, predicted_scores = np.array(true_labels), np.array(predicted_scores)
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    return best_threshold

    
def metric_abstract(labels, outputs, pred_scores, true_labels):
    true_labels.extend(labels.cpu().numpy())
    pred_scores.extend(torch.sigmoid(outputs).cpu().numpy())
    return true_labels, pred_scores

def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def npv_score(y_true, y_pred):
    tn, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0


def metric_func(y_true, y_score, metrics_dict):
    best_thr = calculate_youden_index(y_true, y_score)
    print(best_thr)
    y_pred = (y_score > best_thr).astype(int)
    
    metrics_dict['Best Threshold'] = [best_thr]
    metrics_dict['acc'].append(float(f"{accuracy_score(y_true, y_pred):.3f}"))
    metrics_dict['f1_score'].append(float(f"{f1_score(y_true, y_pred, zero_division=1):.3f}"))
    metrics_dict['recall'].append(float(f"{recall_score(y_true, y_pred, zero_division=1):.3f}"))
    metrics_dict['specificity'].append(float(f"{specificity_score(y_true, y_pred):.3f}"))
    metrics_dict['ppv'].append(float(f"{precision_score(y_true, y_pred, zero_division=1):.3f}"))  # ppv is the same as precision
    metrics_dict['npv'].append(float(f"{npv_score(y_true, y_pred):.3f}"))
    metrics_dict['roc_auc_score'].append(float(f"{roc_auc_score(y_true, y_score):.3f}"))
    return metrics_dict

def save_metric(metrics_dict, save_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Convert the dictionary into a DataFrame for easy CSV saving
    df = pd.DataFrame.from_dict(metrics_dict)
    df.to_csv(save_path, index=False)


def calculate_p_value(original_labels, predicted_scores, n_permutations=1000):
    original_labels, predicted_scores = np.array(original_labels), np.array(predicted_scores)
    original_auc = roc_auc_score(original_labels, predicted_scores)
    permuted_aucs = []
    for _ in range(n_permutations):
        shuffled_labels = np.random.permutation(original_labels)
        auc = roc_auc_score(shuffled_labels, predicted_scores)
        permuted_aucs.append(auc)
    
    p_value = np.sum(original_auc <= np.array(permuted_aucs)) / n_permutations
    return p_value

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

def delong_roc_variance(ground_truth, predictions):
    """
    계산된 AUC의 분산을 계산하는 함수입니다.
    """
    n1 = sum(ground_truth)
    n2 = len(ground_truth) - n1
    auc = roc_auc_score(ground_truth, predictions)
    
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    
    return auc, (auc * (1 - auc) +
                 (n1 - 1) * (q1 - auc**2) +
                 (n2 - 1) * (q2 - auc**2)) / (n1 * n2)

def delong_test(preds1, preds2, labels1, labels2):
    """
    두 모델 간의 DeLong 테스트를 수행합니다.
    """
    auc1, var1 = delong_roc_variance(labels1, preds1)
    auc2, var2 = delong_roc_variance(labels2, preds2)
    
    z_stat = (auc1 - auc2) / np.sqrt(var1 + var2)
    p_value = norm.sf(abs(z_stat)) * 2
    return z_stat, p_value
