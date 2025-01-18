import random
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import pearsonr as pearsonr_array
from sklearn import metrics
from iDeepB.iDeepB.utils.utils import seed_everything
from iDeepB.iDeepB.utils.functions import find_peaks
import sys
import torch

seed_everything(0)
sys.path.append('/data/xliu/work/') 

def randomPadN(df, NLen=101, column="seq", signal="signal"):
    processedData = []
    for index, peakElement in df.iterrows():
        tempSignal = np.array(eval(peakElement[signal]), dtype=float)
        if random.choice(["left", "right"]) == "right":
            leftPadN = random.randint(0, NLen//2)
            peakElement[column] = "N"*leftPadN + peakElement[column][leftPadN:(len(peakElement[column]))]
            tempSignal[0:leftPadN] = 0
        else:
            rightPadN = random.randint(0, NLen//2)
            peakElement[column] = peakElement[column][0:((len(peakElement[column])-rightPadN))] + "N"*rightPadN
            tempSignal[(len(tempSignal)-rightPadN):len(tempSignal)] = 0
        peakElement[signal] = str(tempSignal.tolist())
        processedData.append(peakElement.tolist())
    return pd.DataFrame(processedData, columns=df.columns)

# Function to print model parameter count
def get_parameter_number(model_analyse):
    total_num = sum(p.numel() for p in model_analyse.parameters())
    trainable_num = sum(p.numel() for p in model_analyse.parameters() if p.requires_grad)
    return f'Total parameters: {total_num}, Trainable parameters: {trainable_num}'

def get_used_parameters(model, *inputs):
    """
    计算模型中在forward中实际使用的参数量。
    """
    used_params = set()  # 用于存储实际用到的参数

    # 钩子函数，用来在每层被调用时记录其参数
    def hook_fn(module, input, output):
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:  # 只统计可训练参数
                used_params.add(param)

    # 为每一层注册一个forward hook
    hooks = []
    for layer in model.children():
        hooks.append(layer.register_forward_hook(hook_fn))

    # 进行一次forward pass
    with torch.no_grad():
        model(*inputs)

    # 计算参数量
    total_params = sum(p.numel() for p in used_params)
    for hook in hooks:
        hook.remove()  # 移除hook

    return total_params


def calculate_batch_metrics_seq(outputs_np, targets_np):
    roc_auc_values = []
    pr_auc_values = []
    sp_cor_values = []

    for row_predict, row_target in zip(outputs_np, targets_np):
        # Spearman correlation for each batch
        spearman_corr = spearmanr(row_predict, row_target).correlation

        # ROC-AUC for each batch
        signal_peak_mark, _ = find_peaks(row_target)
        fpr, tpr, thresholds = metrics.roc_curve(y_true=signal_peak_mark, y_score=row_predict, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)

        # PRAUC for each batch
        pr_auc = metrics.average_precision_score(y_true=signal_peak_mark, y_score=row_predict, pos_label=1)

        roc_auc_values.append(roc_auc)
        pr_auc_values.append(pr_auc)
        sp_cor_values.append(spearman_corr)

    return roc_auc_values, pr_auc_values, sp_cor_values

# Calculate metrics for the entire batch after flattening
def calculate_batch_metrics_whole_auc(outputs, targets):
    # Flatten the outputs and targets arrays
    outputs_flat = outputs.flatten()
    targets_flat = targets.flatten()

    # Spearman and Pearson correlation for the entire batch
    sp_whole = spearmanr(outputs_flat, targets_flat)[0]
    p_whole = pearsonr_array(outputs_flat, targets_flat)[0]

    # Calculate ROC-AUC for the entire batch
    signal_peak_mark, peak_count = find_peaks(targets_flat, signal_max_threshold=2, signal_sum_threshold=2)
    fpr, tpr, thresholds = metrics.roc_curve(y_true=signal_peak_mark, y_score=outputs_flat, pos_label=1)
    roc_auc_whole = metrics.auc(fpr, tpr)

    # Calculate PRAUC for the entire batch
    pr_auc_whole = metrics.average_precision_score(y_true=signal_peak_mark, y_score=outputs_flat, pos_label=1)

    return roc_auc_whole, pr_auc_whole, sp_whole, p_whole

def calculate_batch_metrics_whole(outputs, targets):
    # Flatten the outputs and targets arrays
    outputs_flat = outputs.flatten()
    targets_flat = targets.flatten()

    # Spearman and Pearson correlation for the entire batch
    sp_whole = spearmanr(outputs_flat, targets_flat)[0]
    p_whole = pearsonr_array(outputs_flat, targets_flat)[0]

    # Calculate ROC-AUC for the entire batch
    signal_peak_mark, peak_count = find_peaks(targets_flat, signal_max_threshold=2, signal_sum_threshold=2)
    fpr, tpr, thresholds = metrics.roc_curve(y_true=signal_peak_mark, y_score=outputs_flat, pos_label=1)
    roc_auc_whole = metrics.auc(fpr, tpr)

    # Calculate PRAUC for the entire batch
    pr_auc_whole = metrics.average_precision_score(y_true=signal_peak_mark, y_score=outputs_flat, pos_label=1)

    return roc_auc_whole, pr_auc_whole, sp_whole, p_whole