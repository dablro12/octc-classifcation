from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd 
def plot_roc_curve(true_labels, predicted_scores):
    # predicted_scores를 numpy 배열로 변환
    predicted_scores = np.array(predicted_scores)
    
    fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
    roc_auc = auc(fpr, tpr)
    
    # ROC Curve 그리기
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    # plt.savefig(f"../../../models/metric/oci-gan/{checkpoint_path.split('/multi/')[-1].split('/')[0]}.png")
    plt.close()

def metric_func(true_labels, predicted_labels, predicted_scores, metrics_dict):
    accuracy = accuracy_score(true_labels, predicted_labels)
    metrics_dict["acc"] = accuracy

    # F1 Score 계산
    f1 = f1_score(true_labels, predicted_labels)
    metrics_dict["f1-score"] = f1

    # Recall 계산
    recall = recall_score(true_labels, predicted_labels)
    metrics_dict["recall"] = recall

    # Specificity 계산
    specificity = recall_score(true_labels, predicted_labels, pos_label=0)
    metrics_dict["specificity"] = specificity

    # Precision 계산
    precision = precision_score(true_labels, predicted_labels)
    metrics_dict["ppv"] = precision

    # NPV 계산
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    npv = tn / (tn + fn) if (tn + fn) != 0 else np.nan
    metrics_dict["npv"] = npv

    # AUROC 계산
    auroc = roc_auc_score(true_labels, predicted_scores)
    metrics_dict["auroc"] = auroc
    
    return metrics_dict
 
def save_metric(metrics_dict, save_path):
    # 데이터프레임 생성
    df = pd.DataFrame(metrics_dict)

    # CSV 파일로 저장
    df.to_csv(save_path, index=False)

