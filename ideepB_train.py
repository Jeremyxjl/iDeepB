import os
import sys
import argparse
import gc
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from audtorch.metrics.functional import pearsonr as pearsonr_tensor

# 自定义模块导入
sys.path.append('/data/xliu/work/')
from iDeepB.iDeepB.utils.utils import (
    epochAverageMeter, 
    save_model, 
    seed_everything
)
from iDeepB.iDeepB.utils.functions import onehot_encode
from iDeepB.iDeepB.loss.loss import loss_fun
from iDeepB.iDeepB.operations.train_ops import (
    randomPadN, 
    get_parameter_number, 
    calculate_batch_metrics_seq, 
    calculate_batch_metrics_whole
)
from iDeepB.iDeepB.models.iDeepBModel import iDeepB

# Seed
seed_everything(0)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        prog='ideepB_train.py',
        description='Training script for iDeepB model',
        epilog="Show help message."
    )
    
    # 基本参数
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    parser.add_argument("--bsz", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    
    # 训练控制参数
    parser.add_argument("--lr_decay_count_max", type=int, default=5, 
                       help="Max count for learning rate decay")
    parser.add_argument("--max_stopping_step", type=int, default=8, 
                       help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--lr_min", type=float, default=1e-7, 
                       help="Minimum learning rate")
    
    # 数据参数
    parser.add_argument("--pretrained_model_path", type=str, 
                       help="Path to pretrained model")
    parser.add_argument("--output", dest="output", type=str, required=True,
                       help="Output directory path")
    parser.add_argument("--task", dest="task", type=str)
    parser.add_argument("--CUDA", type=str, default="0", 
                       help="CUDA device ID")
    parser.add_argument("--seqLength", type=int, default=0, 
                       help="Sequence length")
    parser.add_argument("--randomPadNLen", type=int, default=101,
                       help="Random padding length")
    parser.add_argument("--encode", type=str, default="Embedding",
                       choices=["OneHot", "Embedding"], 
                       help="Sequence encoding method")
    parser.add_argument("--lossM", type=str, dest="loss_fun_name",
                       required=True, help="Loss function name")
    parser.add_argument('-train', '--trainFile', required=True,
                       metavar="trainFile", type=str,
                       help='Path to training data file')
    parser.add_argument('-validation', '--validationFile', required=True,
                       metavar="validationFile", type=str,
                       help='Path to validation data file')
    parser.add_argument('--log', action="store_true",
                       help="Apply log transformation to data")
    parser.add_argument('--modelFramework', required=True,
                       choices=["Treat", "Control"],
                       help="Model framework type")
    parser.add_argument('--shuff', action="store_true",
                       help="Enable sequence shuffling augmentation")
    parser.add_argument("--signalMax", type=int, default=0,
                       help="Maximum signal threshold for filtering")
    parser.add_argument('--filter', action="store_true",
                       help="Enable data filtering")
    
    return parser.parse_args()

def read_data(bed_fa_name, signal_max, filter_flag, EncodeMode):
    """加载并预处理数据"""
    try:
        # 读取数据
        bed_fa = pd.read_hdf(bed_fa_name, key='df')
        bed_fa.columns = ["chr", "chromStart", "chromEnd", "peak", "score", 
                         "strand", "signal", "signal_ctl", "seq"]
        
        print(f"Initial subsequence count: {bed_fa.shape[0]}")
        
        # 数据过滤
        if filter_flag:
            print(f"Filtering with signal_max={signal_max}")
            bed_fa["mainPeakMaxSignal"] = np.max(np.array(bed_fa["signal"].tolist()), axis=1)
            bed_fa = bed_fa[bed_fa['mainPeakMaxSignal'] >= signal_max]
        
        # 移除含N的序列
        bed_fa = bed_fa[~bed_fa['seq'].str.contains('N', case=False)]
        bed_fa = bed_fa.dropna()
        print(f"After filtering: {bed_fa.shape[0]} sequences remaining")
        print("Strand distribution:", Counter(bed_fa['strand']))
        
        # 序列处理
        seqs = bed_fa["seq"].str.upper().str.replace('T', 'U').tolist()
        
        if EncodeMode == "OneHot":
            vocab = list("AUGC")
            train_data = onehot_encode(seqs, vocab, 4)
            print(f"One-hot encoded shape: {np.array(train_data).shape}")
        
        return train_data, bed_fa["signal"].tolist(), bed_fa["signal_ctl"].tolist()
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


# 解析参数
args = parse_arguments()

task = args.task
protein,cell,accession = task.split("_")[0], task.split("_")[1],task.split("_")[2]

# Set encode mode and data type
EncodeMode = args.encode

# Learning rate
lr = args.lr

# Filter or not
filter = args.filter

# output directory
outputDir = args.output

# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.CUDA)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {args.device}")

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已经存在。")
        
# Set output directory
create_folder_if_not_exists(outputDir) 
create_folder_if_not_exists(outputDir+'/tensorboard') 

writer = SummaryWriter(outputDir+'/tensorboard')

# Train data
train_data, train_label, train_label_ctl  = read_data(args.trainFile, args.signalMax, args.filter, EncodeMode)

# Apply log transform if specified
if args.log:
    print("Applying log transform to training data")
    train_label = np.log(np.asarray(train_label) + 1)
    train_label_ctl = np.log(np.asarray(train_label_ctl) + 1)
else:
    print("No log transform applied to training data")
    train_label = np.asarray(train_label)
    train_label_ctl = np.asarray(train_label_ctl)

# Convert to PyTorch tensors
train_data = torch.from_numpy(np.asarray(train_data))
train_label = torch.from_numpy(train_label).float()
train_label_ctl = torch.from_numpy(train_label_ctl).float()

print(f"Training data shape: {train_data.shape}")
print(f"Training label shape: {train_label.shape}")
print(f"Training control label shape: {train_label_ctl.shape}")

#############
# Create DataLoader for training data
train_dataset = torch.utils.data.TensorDataset(train_data, train_label, train_label_ctl)
del train_data, train_label, train_label_ctl
gc.collect()

trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, drop_last = True)
del train_dataset
gc.collect()

#############

# Validation data
val_data, val_label, val_label_ctl = read_data(args.validationFile , args.signalMax, args.filter, EncodeMode)    

# Apply log transform if specified
if args.log:
    print("Applying log transform to validation data")
    val_label = np.log(np.asarray(val_label) + 1)
    val_label_ctl = np.log(np.asarray(val_label_ctl) + 1)
else:
    print("No log transform applied to validation data")
    val_label = np.asarray(val_label)
    val_label_ctl = np.asarray(val_label_ctl)

# Convert to PyTorch tensors
val_data = torch.from_numpy(np.asarray(val_data))
val_label = torch.from_numpy(val_label).float()
val_label_ctl = torch.from_numpy(val_label_ctl).float()

print(f"Validation data shape: {val_data.shape}")
print(f"Validation label shape: {val_label.shape}")
print(f"Validation control label shape: {val_label_ctl.shape}")

val_dataset = torch.utils.data.TensorDataset(val_data, val_label, val_label_ctl)
del val_data, val_label, val_label_ctl
gc.collect()

valLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bsz, shuffle=True, drop_last = True)
del val_dataset
gc.collect()

print(f"Initializing {args.modelFramework} model...")
from iDeepB.iDeepB.models.iDeepBModel import iDeepB

if args.modelFramework == "Treat":
    model = iDeepB(dropOut=0.2, control=False).to(args.device)
elif args.modelFramework == "Control":
    model = iDeepB(dropOut=0.2, control=True).to(args.device)

print(f"Model parameter count: {get_parameter_number(model)}")

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_loss_best = np.inf
val_loss_best = np.inf
val_prAUC_best = 0

train_metric_epoch = []
val_metric_epoch = []

# Trainning
max_stopping_step = args.max_stopping_step

print("Trainning epochs: ", args.epochs)
for epoch in range(args.epochs):
    # Training phase
    model.train()
    train_loss_meter = epochAverageMeter()
    train_p_meter = epochAverageMeter()

    with tqdm(total=(len(trainLoader))) as t:   
        t.set_description('epoch:{}/{}'.format(epoch, args.epochs))
        for (inputs, targets, targets_ctl) in trainLoader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            targets_ctl = targets_ctl.to(args.device)

            if(EncodeMode == "OneHot"):
                outputs = model(inputs.float())
            
            if (args.modelFramework == "Control"): 
                w = outputs[2]
                w = w.unsqueeze(axis=1)
                target_predict = outputs[0]
                ctl_predict = outputs[1]

                if(args.log):
                    target_predict = torch.exp(target_predict) - 1
                    ctl_predict = torch.exp(ctl_predict) - 1
                    targets = torch.exp(targets) - 1
                
                treat_predict = (target_predict * w) + (1-w) * ctl_predict
                loss_treat = getattr(loss_fun(), args.loss_fun_name)(treat_predict, targets)
                loss_ctl = getattr(loss_fun(), args.loss_fun_name)(ctl_predict, targets_ctl)

                loss = loss_treat + loss_ctl
                p = pearsonr_tensor(treat_predict, targets, batch_first=True).nanmean()  # 实验组的pearson

            elif(args.modelFramework == "Treat"): 
                if(args.log):
                    outputs = torch.exp(outputs) - 1
                    targets = torch.exp(targets) - 1
                loss = getattr(loss_fun(), args.loss_fun_name)(outputs, targets)
                p = pearsonr_tensor(outputs, targets, batch_first=True).nanmean()

            train_loss_meter.update(loss.item())

            if (not torch.isnan(p)): 
                train_p_meter.update(p.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (args.modelFramework == "Treat"): 
                t.set_postfix(loss='{:.3f}'.format(train_loss_meter.avg), p='{:.3f}'.format(train_p_meter.avg), lr='{:.5f}'.format(lr))
            elif(args.modelFramework == "Control"):  
                t.set_postfix(loss_treat='{:.3f}'.format(loss_treat), loss_ctl='{:.3f}'.format(loss_ctl), weight='{:.3f}'.format(torch.mean(w)), loss='{:.3f}'.format(train_loss_meter.avg), p='{:.3f}'.format(train_p_meter.avg))
            t.update(1)

    val_mse_meter = epochAverageMeter()
    val_loss_meter = epochAverageMeter()
    val_p_meter = epochAverageMeter()
    val_sp_meter = epochAverageMeter()
    val_AUC_meter = epochAverageMeter()
    val_prAUC_meter = epochAverageMeter()

    # validation data
    model.eval()
    for (inputs, targets, targets_ctl) in valLoader:
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        targets_ctl = targets_ctl.to(args.device)

        if(EncodeMode == "OneHot"):
            outputs = model(inputs.float())
        
        if (args.modelFramework == "Control"): 
            w = outputs[2]
            w = w.unsqueeze(axis=1)
            target_predict = outputs[0]
            ctl_predict = outputs[1]

            if(args.log):
                target_predict = torch.exp(target_predict) - 1
                ctl_predict = torch.exp(ctl_predict) - 1
                targets = torch.exp(targets) - 1
            
            treat_predict = (target_predict * w) + (1-w) * ctl_predict
            loss_treat = getattr(loss_fun(), args.loss_fun_name)(treat_predict, targets)
            loss_ctl = getattr(loss_fun(), args.loss_fun_name)(ctl_predict, targets_ctl)

            valLoss = loss_treat + loss_ctl
            p = pearsonr_tensor(treat_predict, targets, batch_first=True).nanmean()  # 实验组的pearson

            outputs_np = treat_predict.cpu().detach().numpy()
            targets_np = targets.cpu().detach().numpy()

        elif(args.modelFramework == "Treat"): 
            if(args.log):
                outputs = torch.exp(outputs) - 1
                targets = torch.exp(targets) - 1
            valLoss = getattr(loss_fun(), args.loss_fun_name)(outputs, targets)
            p = pearsonr_tensor(outputs, targets, batch_first=True).nanmean()
        
            outputs_np = outputs.cpu().detach().numpy()
            targets_np = targets.cpu().detach().numpy()
        
        precise = False
        if precise:
            p_cor = pearsonr_tensor(outputs, targets).nanmean()
            roc_auc_values, pr_auc_values, sp_cor_values = calculate_batch_metrics_seq(outputs_np, targets_np)

            # Mse
            mse_values = np.nanmean((outputs_np - targets_np)**2, axis=None)
            
            sp_cor = np.nanmean(sp_cor_values)
            roc_auc = np.nanmean(roc_auc_values)
            pr_auc = np.nanmean(pr_auc_values)

        else:
            roc_auc, pr_auc, sp_cor, p_cor = calculate_batch_metrics_whole(outputs_np, targets_np)

            # Mse
            mse_values = np.nanmean((outputs_np - targets_np)**2, axis = None)

        val_loss_meter.update(valLoss.item())
        
        if (not np.isnan(p_cor)):
            val_p_meter.update(p_cor.item())
        val_sp_meter.update(sp_cor)
        val_AUC_meter.update(roc_auc)
        if (not np.isnan(pr_auc)):
            val_prAUC_meter.update(pr_auc)
        val_mse_meter.update(mse_values)

    # lr decay
    if val_loss_best > val_loss_meter.avg:
        decay_count = 0
        val_loss_best = val_loss_meter.avg
    else:
        decay_count += 1
        if decay_count >= args.lr_decay_count_max: 
            lr = optimizer.param_groups[0]['lr'] / 2  #
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if lr < args.lr_min:    # stop training
                print(f"Early stoping by lr, epoch={epoch+1}, ")
                break
            decay_count = 0

    # Early stop
    if (val_prAUC_best < val_prAUC_meter.avg): 
        save_model(outputDir, model, optimizer, name="model")
        stopping_step = 0
        val_prAUC_best = val_prAUC_meter.avg
    else:
        stopping_step += 1
        if stopping_step >= max_stopping_step:
            print(f"Early stoping by val_loss, epoch={epoch+1}")
            break

    # writer
    writer.add_scalar('train_loss', train_loss_meter.avg, epoch)
    writer.add_scalar('train_p', train_p_meter.avg, epoch)

    writer.add_scalar('val_loss', val_loss_meter.avg, epoch)
    writer.add_scalar('val_p', val_p_meter.avg, epoch)
    writer.add_scalar('val_sp', val_sp_meter.avg, epoch)
    writer.add_scalar('val_AUC', val_AUC_meter.avg, epoch)
    writer.add_scalar('val_prAUC', val_prAUC_meter.avg, epoch)
    writer.add_scalar('val_prAUC', val_mse_meter.avg, epoch)
    writer.add_scalar('lr', lr, epoch)
    
    val_metric_epoch.append([val_loss_meter.avg, val_p_meter.avg, val_sp_meter.avg, val_AUC_meter.avg, val_prAUC_meter.avg, val_mse_meter.avg])
    train_metric_epoch.append([train_loss_meter.avg, train_p_meter.avg])

val_metric_epoch_df = pd.DataFrame(val_metric_epoch)
train_metric_epoch_df = pd.DataFrame(train_metric_epoch)

val_metric_epoch_df.columns = ["loss", "pearson", "spearman", "ROC_AUC", "PR_AUC", "mse"]
train_metric_epoch_df.columns = ["loss", "pearson"]
val_metric_epoch_df.to_csv(f"{outputDir}/val_metric.txt", sep="\t", index =False)
train_metric_epoch_df.to_csv(f"{outputDir}/train_metric.txt", sep="\t", index =False)

if(args.log):
    print(f"{protein}, {cell}, {args.loss_fun_name}, log, loss={val_loss_meter.avg}, pearson={val_p_meter.avg}, spearman={val_sp_meter.avg}, auc={val_AUC_meter.avg}, prauc={val_prAUC_meter.avg}, mse={val_mse_meter.avg}")
else:
    print(f"{protein}, {cell}, {args.loss_fun_name}, non-log, loss={val_loss_meter.avg}, pearson={val_p_meter.avg}, spearman={val_sp_meter.avg}, auc={val_AUC_meter.avg}, prauc={val_prAUC_meter.avg}, mse={val_mse_meter.avg}")


OP = open(f"iDeepB_train_val_metric.txt", "a+")

if(args.log):
    print(f"{protein}\t{cell}\t{args.loss_fun_name}\tlog\tloss={val_loss_meter.avg}\tpearson={val_p_meter.avg}\tspearman={val_sp_meter.avg}\tauc={val_AUC_meter.avg}\tprauc={val_prAUC_meter.avg}\tmse={val_mse_meter.avg}", file =OP)
else:
    print(f"{protein}\t{cell}\t{args.loss_fun_name}\tnon-log\tloss={val_loss_meter.avg}\tpearson={val_p_meter.avg}\tspearman={val_sp_meter.avg}\tauc={val_AUC_meter.avg}\tprauc={val_prAUC_meter.avg}\tmse={val_mse_meter.avg}", file =OP)
OP.close()
