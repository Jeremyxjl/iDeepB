#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import scipy.stats as stats
import os
import sys
import shutil
import scipy
from Bio.Seq import reverse_complement,transcribe,Seq
import torch
import pandas as pd
#from pybedtools import Interval
import numpy as np
import pysam
import pyBigWig
from sklearn import metrics
from sklearn.metrics import average_precision_score
from tqdm import tqdm
sys.path.append('/data/xliu/work/') 
from iDeepB.iDeepB.utils.functions import onehot_encode

from iDeepB.iDeepB.preprocessing.RBP_util import Track
import matplotlib.pyplot as plt


# In[ ]:


class Track:
    def __init__(self, bigWigPlus, bigWigMinus):
        self._bigWigPlus = pyBigWig.open(bigWigPlus)
        self._bigWigMinus = pyBigWig.open(bigWigMinus)
    def profile(self, chrInfo, startPos, endPos, strand):
        if strand == '+':
            signalList = self._bigWigPlus.values(chrInfo, int(startPos), int(endPos), numpy=False)
            profile = np.nan_to_num(signalList, nan=0.0).tolist()

        elif strand == '-':
            signalList = self._bigWigMinus.values(chrInfo, int(startPos), int(endPos), numpy=False)
            signalList = np.nan_to_num(signalList, nan=0.0)
            profile = list(reversed(signalList))
        else:
            raise ValueError(f'Unspected strand: {strand}')
        return profile

class genomeSeqClasss:
    def __init__(self, genomeFaFile):
        # import genome fa file
        self._genomeFa = pysam.FastaFile(genomeFaFile) #args.genomeFa
    def seqFetch(self, chrInfo, startPos, endPos, strand):
        if strand == '+':
            seq = self._genomeFa.fetch(chrInfo, startPos, endPos)
        elif strand == '-':
            seq = self._genomeFa.fetch(chrInfo, startPos, endPos)
            seq = reverse_complement(seq)
        else:
            raise ValueError(f'Unspected strand: {strand}')
        return seq


# In[ ]:


#def predictFun(seqTranscribed, window_size, signalList):
def predictHead(seqTranscribed, window_size, signalList, model, mode, codeModel = "OneHot"):
    ## padding and extract subSeq or subSignal
    remainder = (len(seqTranscribed) % window_size)

    print(f"## remainder Length is: {remainder}")

    if mode == "Fragment":
        # 计算列表可以分成多少个子列表
        num_sublists = len(seqTranscribed) // window_size

        # 使用列表推导式创建子列表
        sublists = [seqTranscribed[i * window_size: (i + 1) * window_size] for i in range(num_sublists)]

        # 如果剩余元素不为零，将其加入最后一个子列表
        remainder = len(seqTranscribed) % window_size
        if remainder:
            sublists.append(seqTranscribed[-window_size:])
            
    elif mode == "midBase":
        sublists = [seqTranscribed[i: i + window_size] for i in np.arange(0, len(seqTranscribed)-window_size + 1, step=1)]

        # for reminder 直接取第一个和最后一个就好
        # sublists_end = [seqTranscribed[:window_size], seqTranscribed[-window_size:]]
        # sublists = sublists + sublists_end
    # print(sublists)
    if codeModel == "OneHot":
        vocab = list("AUGC")
        seqsInt = onehot_encode(sublists, vocab, 4)
        subseqOH = torch.from_numpy(np.asarray(seqsInt) ).to(args.device)
    elif codeModel == "Embedding":
        seqsInt = [sequence2int(seq) for seq in sublists]
    subseqOH = torch.from_numpy(np.asarray(seqsInt) ).to(args.device)

    model.eval()
    with torch.no_grad():
        temp = []

        if(EncodeMode == "Embedding"):
            subseqOH = subseqOH.int()
        elif(EncodeMode == "OneHot"):
            subseqOH = subseqOH.float()
        #elif(args.modelFramework == "Enformer"):
            #subseqPredicted = model(subseqOH) 
        if mode == "Fragment":
        
            if remainder:
                for subseqOHTurn in subseqOH[:-1].split(500,  0): #预测转录本，数据太大，无法预
                    treat_predict = model(subseqOHTurn) #torch.Size([n, 101])
                    treat_predict = treat_predict.detach().cpu().numpy().tolist()
                    temp.extend(treat_predict)
                temp = np.array(temp).reshape(-1).tolist()
                
                print("first temp len", len(temp))
                # for remainder length
                treat_predict_end = model(subseqOHTurn[-1:]) #torch.Size([n, 101])
                treat_predict_end = treat_predict_end.detach().cpu().numpy().tolist()

                print(remainder, len(temp), len(np.array(treat_predict_end).reshape(-1).tolist()[-remainder:]), len(np.array(treat_predict_end).reshape(-1).tolist()))
                temp = temp + np.array(treat_predict_end).reshape(-1).tolist()[-remainder:]
                print("remainder: ",remainder, len(treat_predict_end[-remainder:]), len(temp))
            else:
                for subseqOHTurn in subseqOH.split(500,  0): #预测转录本，数据太大，无法预
                    treat_predict = model(subseqOHTurn) #torch.Size([n, 101])

                    treat_predict = treat_predict.detach().cpu().numpy().tolist()

                    temp.extend(treat_predict)
                temp = np.array(temp).reshape(-1).tolist()    
        elif mode == "midBase":  #column_values = tensor_2d[:, 1]
            for subseqOHTurn in subseqOH.split(1000,  0): #预测转录本，数据太大，无法预 [0:-0] 非[1:-1]
                subseqPredicted = model(subseqOHTurn) #torch.Size([n, 101])
                treat_predict = subseqPredicted
   
                treat_predict = treat_predict[:, window_size//2]
                treat_predict = treat_predict.detach().cpu().numpy().tolist()

                temp.extend(treat_predict)

            # for remainder length
            # 对于前半段 remainder_index == 0
            remainder_index = 0
            outputs_end = model(subseqOH[remainder_index].unsqueeze(0)) #torch.Size([n, 101])
            w_end = outputs_end[2]
            w_end = w_end.unsqueeze(axis=1)
            
            treat_predict_end = (outputs_end[0]*w_end) + (1-w_end)*outputs_end[1]
            treat_predict_end = treat_predict_end.detach().cpu().numpy()
            
            temp = treat_predict_end[:, :window_size//2].reshape(-1).tolist() + temp

            # 对于后半段
            remainder_index = -1
            treat_predict_end = model(subseqOH[remainder_index].unsqueeze(0)) #torch.Size([n, 101])

            treat_predict_end = treat_predict_end.detach().cpu().numpy()

            temp = treat_predict_end[:, :window_size//2].reshape(-1).tolist() + temp
                

        transcriptPd = np.array(temp) #subseqPredicted.cpu().detach().numpy()
        # cor 1: subseq预测结果合并成一个向量，然后与实际信号计算相关性
        # transcriptPd = subseqPredictedNp.reshape(-1)
        print("Signal lenth of prediction and transcript length !", len(transcriptPd), len(signalList))
        if(len(transcriptPd) != len(signalList)):
            print("Error: signal lenth not equal transcript length !", len(transcriptPd), len(signalList))
            return False
    return transcriptPd


# In[ ]:


def predict2HeadW(seqTranscribed, window_size, signalList, model, mode, codeModel = "OneHot"):  #mode "Fragment" midBase
    ## padding and extract subSeq or subSignal
    remainder = (len(seqTranscribed) % window_size)

    print(f"## remainder Length is: {remainder}")

    if mode == "Fragment":
        # 计算列表可以分成多少个子列表
        num_sublists = len(seqTranscribed) // window_size

        # 使用列表推导式创建子列表
        sublists = [seqTranscribed[i * window_size: (i + 1) * window_size] for i in range(num_sublists)]

        # 如果剩余元素不为零，将其加入最后一个子列表
        remainder = len(seqTranscribed) % window_size
        if remainder:
            sublists.append(seqTranscribed[-window_size:])
            
    elif mode == "midBase":
        sublists = [seqTranscribed[i: i + window_size] for i in np.arange(0, len(seqTranscribed)-window_size + 1, step=1)]

        # for reminder 直接取第一个和最后一个就好
        # sublists_end = [seqTranscribed[:window_size], seqTranscribed[-window_size:]]
        # sublists = sublists + sublists_end
    # print(sublists)
    if codeModel == "OneHot":
        vocab = list("AUGC")
        seqsInt = onehot_encode(sublists, vocab, 4)
        subseqOH = torch.from_numpy(np.asarray(seqsInt) ).to(args.device)
    elif codeModel == "Embedding":
        seqsInt = [sequence2int(seq) for seq in sublists]
    subseqOH = torch.from_numpy(np.asarray(seqsInt) ).to(args.device)

    model.eval()
    with torch.no_grad():
        temp = []

        if(EncodeMode == "Embedding"):
            subseqOH = subseqOH.int()
        elif(EncodeMode == "OneHot"):
            subseqOH = subseqOH.float()
        #elif(args.modelFramework == "Enformer"):
            #subseqPredicted = model(subseqOH) 
        if mode == "Fragment":
        
            if remainder:
                for subseqOHTurn in subseqOH[:-1].split(500,  0): #预测转录本，数据太大，无法预
                    outputs = model(subseqOHTurn) #torch.Size([n, 101])
                    w = outputs[2]
                    w = w.unsqueeze(axis=1)
                    
                    treat_predict = (outputs[0]*w) + (1-w)*outputs[1]
                    treat_predict = treat_predict.detach().cpu().numpy().tolist()

                    temp.extend(treat_predict)
                temp = np.array(temp).reshape(-1).tolist()
                
                print("first temp len", len(temp))
                # for remainder length
                outputs_end = model(subseqOHTurn[-1:]) #torch.Size([n, 101])
                w_end = outputs_end[2]
                w_end = w_end.unsqueeze(axis=1)
                
                treat_predict_end = (outputs_end[0]*w_end) + (1-w_end)*outputs_end[1]
                treat_predict_end = treat_predict_end.detach().cpu().numpy().tolist()

                print(remainder, len(temp), len(np.array(treat_predict_end).reshape(-1).tolist()[-remainder:]), len(np.array(treat_predict_end).reshape(-1).tolist()))
                temp = temp + np.array(treat_predict_end).reshape(-1).tolist()[-remainder:]
                print("remainder: ",remainder, len(treat_predict_end[-remainder:]), len(temp))
            else:
                for subseqOHTurn in subseqOH.split(500,  0): #预测转录本，数据太大，无法预
                    outputs = model(subseqOHTurn) #torch.Size([n, 101])
                    w = outputs[2]
                    w = w.unsqueeze(axis=1)
                    
                    treat_predict = (outputs[0]*w) + (1-w)*outputs[1]
                    treat_predict = treat_predict.detach().cpu().numpy().tolist()

                    temp.extend(treat_predict)
                temp = np.array(temp).reshape(-1).tolist()    
        elif mode == "midBase":  #column_values = tensor_2d[:, 1]
            for subseqOHTurn in subseqOH.split(1000,  0): #预测转录本，数据太大，无法预 [0:-0] 非[1:-1]
                subseqPredicted = model(subseqOHTurn) #torch.Size([n, 101])
                outputs = subseqPredicted
                w = outputs[2]
                w = w.unsqueeze(axis=1)
                
                treat_predict = (outputs[0]*w) + (1-w)*outputs[1]
                treat_predict = treat_predict[:, window_size//2]
                treat_predict = treat_predict.detach().cpu().numpy().tolist()

                temp.extend(treat_predict)

            # for remainder length
            # 对于前半段 remainder_index == 0
            remainder_index = 0
            outputs_end = model(subseqOH[remainder_index].unsqueeze(0)) #torch.Size([n, 101])
            w_end = outputs_end[2]
            w_end = w_end.unsqueeze(axis=1)
            
            treat_predict_end = (outputs_end[0]*w_end) + (1-w_end)*outputs_end[1]
            treat_predict_end = treat_predict_end.detach().cpu().numpy()
            
            temp = treat_predict_end[:, :window_size//2].reshape(-1).tolist() + temp

            # 对于后半段
            remainder_index = -1
            outputs_end = model(subseqOH[remainder_index].unsqueeze(0)) #torch.Size([n, 101])
            w_end = outputs_end[2]
            w_end = w_end.unsqueeze(axis=1)
            
            treat_predict_end = (outputs_end[0]*w_end) + (1-w_end)*outputs_end[1]
            treat_predict_end = treat_predict_end.detach().cpu().numpy()

            temp = treat_predict_end[:, :window_size//2].reshape(-1).tolist() + temp
                

        transcriptPd = np.array(temp) #subseqPredicted.cpu().detach().numpy()
        # cor 1: subseq预测结果合并成一个向量，然后与实际信号计算相关性
        # transcriptPd = subseqPredictedNp.reshape(-1)
        print("Signal lenth of prediction and transcript length !", len(transcriptPd), len(signalList))
        if(len(transcriptPd) != len(signalList)):
            print("Error: signal lenth not equal transcript length !", len(transcriptPd), len(signalList))
            return False
    return transcriptPd


# In[ ]:


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已经存在。")

class args:
    pass

# 参数设置
fasta_genome = "/data/xliu/dataBase/GRCh38.p13.genome.fa"

#Encode_eCLIP_dataSet路径
eCLIP_dataSet_path = "/data/xliu/work/20231211_iDeepB/20240216_trainData/bam2bigwig/Encode_eCLIP_dataSet"


log = ""
loss = "poisson_loss_test" # "poisson_loss_test"  #  kl_divergence_loss  multinomialnll_batch pearson mse poisson_loss_test
model_dir = "ideepB_trained_model"
args.mode = "Fragment"
args.modelFramework =  "ResLSTMMSA" #"CNNLSTM16"
args.outputFolder = "iDeepB_predict_TP_head_ResLSTMMSA"

# 使用tab20c颜色映射
point_colors = plt.cm.tab20c  #tab20c

# create_folder_if_not_exists(f"{output_dir}")

args.genomeFa = "/home/xliu/dataBase/GRCh38.p13.genome.fa"
## genome fasta
genomeFa = pysam.FastaFile(args.genomeFa) #"/home/xliu/dataBase/GRCh38.p13.genome.fa"
genomeFa2 = genomeSeqClasss(args.genomeFa) 
    

# eCLIP_dataSet_array = [folder.split("/")[-1].split("_") for folder in glob.glob(f'{eCLIP_dataSet_path}/*') ]

BamInfo_file = "ENCODE_eCLIP_252_BamInfo.txt"


# bam info input
ENCODE_eCLIP_bamInfo = pd.read_table(BamInfo_file, header=0, sep = "\t")
ENCODE_eCLIP_bamInfo['term_name'] = ENCODE_eCLIP_bamInfo['term_name'].replace('adrenal gland', 'adrenalgland', regex=True)
eCLIP_dataSet_DF = ENCODE_eCLIP_bamInfo # ENCODE_eCLIP_bamInfo[["symbol", "term_name", "accession", "run_type"]]
eCLIP_dataSet_DF.insert(loc=3, column='dataSet', value = "dataSet")
eCLIP_dataSet_DF.insert(loc=4, column='correct', value = "c0")

# main process 
corPearson = []
corSpearman = []
roc_auc = []
roc_auc_prc = []
peakCountList = []
TP_name = []
RBP_name = []


for index ,resies_row in eCLIP_dataSet_DF.iterrows():
    print(f"Doing: {index}")

    if (resies_row.term_name == "adrenal gland"):
        resies_row.term_name = "adrenalgland"

    protein = resies_row.symbol
    cell = resies_row.term_name
    task = f'{protein}_{cell}_{resies_row.accession}'
    runtype = resies_row.run_type
    correct = resies_row.correct
    eclipID = resies_row.accession

    if log == "":
        args.log = False
    if log != "":
        args.log = True

    args.sample = f"{protein}_{cell}"
    args.transcriptBed = f"/data/xliu/work/20231211_iDeepB/20240307_transcript_predict/TP_with_peak/TP_with_peak/{task}_TP_with_peak_rmDup.txt"
    args.window_size = 101
    args.output = f"{args.outputFolder}/{protein}_{cell}_train_101_encode_c0_{loss}_test_{log}_{args.modelFramework}" # f"QKI_{cell}_train_101_encode_c0_{lossFun}_{args.modelFramework}_Fragment_TP"

    args.modelPath =  f"{model_dir}/{task}_c0/model_params.pth" 
  
    
    args.plus_bw = f"/data/xliu/work/20231211_iDeepB/20240216_trainData/bam2bigwig/Encode_eCLIP_dataSet/{task}_dataSet_{correct}_{runtype}/{task}.pos.bigWig"
    args.minus_bw = f"/data/xliu/work/20231211_iDeepB/20240216_trainData/bam2bigwig/Encode_eCLIP_dataSet/{task}_dataSet_{correct}_{runtype}/{task}.neg.bigWig"
    
    bwTrack = Track(args.plus_bw, args.minus_bw)
    
    args.CUDA = "1"
    codeModel = "OneHot"
    EncodeMode = "OneHot"
    mode = args.mode
    window_size = args.window_size
    if(mode == "midBase"):
        subStride = 1
    elif(mode == "Fragment"):
        subStride = window_size

    args.slide_len = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.CUDA)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.Signal_OP = False

    transcriptBed =  pd.read_table(args.transcriptBed, sep="\t", header=None, names=["chrom_TP", "start_TP", "stop_TP", "name_TP", "score_TP", "strand_TP", "chr", "start", "end", "name", "strand", 'signalValue', 'pValue',"Count"])


    # In[3]:
    if args.modelFramework == "ResLSTMMSA":
        print("Model: ", args.modelFramework)
        # from iDeepB.iDeepB.models.ResLSTMMSA import iDeepB  #ResLSTMMSA_v1
        from iDeepB.iDeepB.models.ResLSTMMSA_v1 import iDeepB  #ResLSTMMSA_v1
        model = iDeepB(dropOut=0.).to(args.device)
    elif(args.modelFramework == "ResLSTMMSA2headw"): 
        print("Model: ", args.modelFramework)
        from iDeepB.iDeepB.models.ResLSTMMSA2headw import iDeepB
        model = iDeepB(dropOut = 0.).to(args.device)
    elif(args.modelFramework == "treat"): 
        print("Model: ", args.modelFramework)
        from iDeepB.iDeepB.models.models import iDeepB
        model = iDeepB(dropOut = 0.).to(args.device)
    elif(args.modelFramework == "weight"): 
        print("Model: ", args.modelFramework)
        from iDeepB.iDeepB.models.models import iDeepB
        model = iDeepB(dropOut = 0., control = True).to(args.device)
    # print("Train model: ",model)
    model.load_state_dict(torch.load(args.modelPath), strict=False)
    print(mode)
    ## inpput bw file
    plus_bw = pyBigWig.open(args.plus_bw)#pyBigWig.open("/home/xliu/work/20230103_EPRB_model/TIA1_{cell}/TIA1_{cell}.pos.bigWig") #"ENCFF947JHV.secondR_128.5.pos.sorted.bigWig"
    minus_bw = pyBigWig.open(args.minus_bw)#pyBigWig.open("/home/xliu/work/20230103_EPRB_model/TIA1_{cell}/TIA1_{cell}.neg.bigWig") #"ENCFF947JHV.secondR_128.5.neg.sorted.bigWig"

    ## 需要预测的数据
    #transcriptBed = args.transcriptBed#"/home/xliu/work/20230103_EPRB_model/TIA1_{cell}/RBP.TIA1_{cell}.300.peak_test.transcript.fa.bed"
    # output file
    outputDir = args.output
    if(os.path.isdir(outputDir)):
        shutil.rmtree(outputDir)
        os.makedirs(outputDir)
    else:
        os.makedirs(outputDir)

    if args.Signal_OP:
        predictOPFile = f"{outputDir}/RBP.predict.out" # f"{args.output}.RBP.predict.out"
        predictCorOPFile = f"{outputDir}/RBP.predict.cor.out" # f"{args.output}.RBP.predict.cor.out"
        predictOP = open(predictOPFile, "w+")
        predictCorOP = open(predictCorOPFile, "w+")

    model = model.to(args.device)

    # 检测泛化能力
    slide_len = args.slide_len
    if(slide_len != 0):
        corPearson_slide = []
        corSpearman_slide = []

    # 选择测试集部分
    test_chr = ["chr1", "chr8", "chr15"]
    print(f"Before filter by length: {transcriptBed.shape}")
    transcriptBed = transcriptBed[transcriptBed["chrom_TP"].isin(test_chr)]

    transcriptBed_dup = transcriptBed.drop_duplicates(subset='name_TP')

    transcriptBed_dup["length_TP"] = transcriptBed_dup["stop_TP"] - transcriptBed_dup["start_TP"]

    # 获取满足要求的转录本
    # transcriptBed_dup = transcriptBed_dup[transcriptBed_dup["length_TP"] < 50000]  
    # 筛选的长度，变长，结果会下降很多
    transcriptBed_dup = transcriptBed_dup[(transcriptBed_dup["length_TP"] < 100000) & (transcriptBed_dup["length_TP"] >= 101)]  
    print(f"After filter by length: {transcriptBed_dup.shape}")

    '''
    # main process 
    corPearson = []
    corSpearman = []
    roc_auc = []
    roc_auc_prc = []
    peakCountList = []
    TP_name = []
    '''

    TPList = list(transcriptBed_dup["name_TP"].values)
    for index_TP, transcriptName in tqdm(enumerate(TPList), total = len(TPList)):
        TF_with_peak = transcriptBed[transcriptBed["name_TP"] == transcriptName]

        # TP_name.append(transcriptName)
        # 获得每个转录本，encode peak
        TF_with_peak = TF_with_peak.reset_index()
        for index_peak, transcript in TF_with_peak.iterrows():

            if index_peak == 0:
                label_list = np.array([0]* (transcript["stop_TP"] - transcript["start_TP"]))
            label_start = transcript["start"]- transcript["start_TP"]
            label_end = transcript["end"]- transcript["start_TP"]
            label_list[label_start:label_end] = 1 

        labelList = label_list

        # 如果没有序列，给的基因组位置，直接从基因组获取序列
        seq = genomeFa2.seqFetch(transcript.chrom_TP, int(transcript.start_TP), int(transcript.stop_TP), transcript.strand_TP)
        seqTranscribed = transcribe(seq) #T->U
        signalList = bwTrack.profile(transcript.chrom_TP, int(transcript.start_TP), int(transcript.stop_TP), transcript.strand_TP)

        if "N" in seqTranscribed:
            continue
        RBP_name.append(task)
        TP_name.append(transcriptName)

        if(slide_len != 0):
            seq_slide = genomeFa2.seqFetch(transcript.chrom_TP, int(transcript.start_TP)+slide_len, int(transcript.stop_TP)+slide_len, transcript.strand_TP)
            seqTranscribed_slide = transcribe(seq_slide) #T->U
            #signalList_slide = bwTrack.profile(transcript.chrom_TP, int(transcript.start_TP)+slide_len, int(transcript.stop_TP)+slide_len, transcript.strand_TP)

        # 如果转录本上面没有结合信号，则不进行预测
        if(sum(signalList) == 0):
            corPearson.append(np.nan)
            roc_auc.append(np.nan)
            roc_auc_prc.append(np.nan)
            peakCountList.append(0)
            continue
        print(f"# TP index: {index} {transcriptName}: TP length = {len(signalList)}, signal sum = {sum(signalList)}, signal max = {max(signalList)}")

        if(mode == "Fragment"):
            print(f"mode: {mode}")
            # 预测
            if (args.log):
                transcriptPd = np.exp(predictHead(seqTranscribed, window_size, signalList, model, mode="Fragment"))-1
            else:
                transcriptPd = predictHead(seqTranscribed, window_size, signalList, model, mode="Fragment")

            # transcriptPd = transcriptPd[0]

            # 通过平移sequence，测试模型泛化能力
            if(slide_len>0):
                if (args.log):
                    transcriptPd_slide = np.exp(predictHead(seqTranscribed_slide, window_size, signalList, model, mode="Fragment"))-1
                else:
                    transcriptPd_slide = predictHead(seqTranscribed_slide, window_size, signalList, model, mode="Fragment")

                transcriptPdpearsonCor_slide = scipy.stats.pearsonr(transcriptPd[slide_len:], transcriptPd_slide[0:-slide_len])[0]
                transcriptPdSpearmanCor_slide = scipy.stats.spearmanr(transcriptPd[slide_len:], transcriptPd_slide[0:-slide_len])[0]

                corPearson_slide.append(transcriptPdpearsonCor_slide)
                corSpearman_slide.append(transcriptPdSpearmanCor_slide)

                print("Slide test: ", f"; slide length = {slide_len}; overlap sequence length = {len(transcriptPd[slide_len:])}", "; pearsonCor:", transcriptPdpearsonCor_slide, "; SpearmanCor: ", transcriptPdSpearmanCor_slide)

            elif(slide_len<0):
                if (args.log):
                    transcriptPd_slide = np.exp(predictHead(seqTranscribed_slide, window_size, signalList, model))-1
                else:
                    transcriptPd_slide = predictHead(seqTranscribed_slide, window_size, signalList, model)

                transcriptPdpearsonCor_slide = scipy.stats.pearsonr(transcriptPd[0:-slide_len], transcriptPd_slide[slide_len:])[0]
                transcriptPdSpearmanCor_slide = scipy.stats.spearmanr(transcriptPd[0:-slide_len], transcriptPd_slide[slide_len:])[0]

                corPearson_slide.append(transcriptPdpearsonCor_slide)
                corSpearman_slide.append(transcriptPdSpearmanCor_slide)

                print("Slide test: ", f"; slide length = {slide_len}; overlap sequence length = {len(transcriptPd[slide_len:])}", "; pearsonCor:", transcriptPdpearsonCor_slide, "; SpearmanCor: ", transcriptPdSpearmanCor_slide)
            
            print(f"Sequence length: {len(transcriptPd)}; signalList length: {len(signalList)}", )
            # print signal and predicted signal
            transcriptPdpearsonCor = scipy.stats.pearsonr(transcriptPd, signalList) #np.exp()-1
            transcriptPdSpearmanCor = scipy.stats.spearmanr(transcriptPd, signalList)

            #print(f"The M1 spearmanCor of {transcriptName} is {subseqPadPdMergeCor[0]}")
            corPearson.append(transcriptPdpearsonCor[0])
            corSpearman.append(transcriptPdSpearmanCor[0])
            # print(f"The M1 correlation of {transcriptName}:  pearson={transcriptPdpearsonCor[0]};  spearman={transcriptPdSpearmanCor[0]}")
            #print(transcriptName,(np.around(np.exp(transcriptPd)-1, decimals=2)).tolist(),signalList, file=predictOP, sep="\t")

            '''
            signalPeakMark, peakCount = peakMark(signalList)
            peakCountList.append(peakCount)
            
            if(peakCount == 0):
                #roc_auc.append(0)
                #roc_auc_prc.append(0)
                continue
            '''
            ## ROC-AUC
            fpr, tpr, thresholds = metrics.roc_curve(y_true=labelList, y_score = transcriptPd, pos_label=1)
            aucValue = metrics.auc(fpr, tpr)
            roc_auc.append(aucValue)

            # ROCAUCPRC
            AUCPRCValue = average_precision_score(y_true=labelList, y_score=transcriptPd, pos_label=1)
            roc_auc_prc.append(AUCPRCValue)

            if args.Signal_OP:
                print(transcriptName,(np.around(np.exp(transcriptPd)-1, decimals=2)).tolist(),signalList, file=predictOP, sep="\t")
                print(transcriptName, transcriptPdpearsonCor[0], transcriptPdSpearmanCor[0], aucValue, AUCPRCValue, file=predictCorOP, sep="\t")
            # print(f"The M1 correlation of {transcriptName}:  pearson={transcriptPdpearsonCor[0]};  spearman={transcriptPdSpearmanCor[0]}; aucValue={aucValue}; AUCPRCValue={AUCPRCValue}")
        elif(mode == "midBase"):

            # 预测
            if (args.log):
                transcriptPd = np.exp(predictHead(seqTranscribed, window_size, signalList, model, mode="midBase"))-1
            else:
                transcriptPd = predictHead(seqTranscribed, window_size, signalList, model, mode="midBase")

            # transcriptPd = transcriptPd[0]

            # 通过平移sequence，测试模型泛化能力
            if(slide_len>0):
                if (args.log):
                    transcriptPd_slide = np.exp(predictHead(seqTranscribed_slide, window_size, signalList, model, mode="midBase"))-1
                else:
                    transcriptPd_slide = predictHead(seqTranscribed_slide, window_size, signalList, model, mode="midBase")

                transcriptPdpearsonCor_slide = scipy.stats.pearsonr(transcriptPd[slide_len:], transcriptPd_slide[0:-slide_len])[0]
                transcriptPdSpearmanCor_slide = scipy.stats.spearmanr(transcriptPd[slide_len:], transcriptPd_slide[0:-slide_len])[0]

                corPearson_slide.append(transcriptPdpearsonCor_slide)
                corSpearman_slide.append(transcriptPdSpearmanCor_slide)

                print("Slide test: ", f"; slide length = {slide_len}; overlap sequence length = {len(transcriptPd[slide_len:])}", "; pearsonCor:", transcriptPdpearsonCor_slide, "; SpearmanCor: ", transcriptPdSpearmanCor_slide)

            elif(slide_len<0):
                if (args.log):
                    transcriptPd_slide = np.exp(predictHead(seqTranscribed_slide, window_size, signalList, model))-1
                else:
                    transcriptPd_slide = predictHead(seqTranscribed_slide, window_size, signalList, model)

                transcriptPdpearsonCor_slide = scipy.stats.pearsonr(transcriptPd[0:-slide_len], transcriptPd_slide[slide_len:])[0]
                transcriptPdSpearmanCor_slide = scipy.stats.spearmanr(transcriptPd[0:-slide_len], transcriptPd_slide[slide_len:])[0]

                corPearson_slide.append(transcriptPdpearsonCor_slide)
                corSpearman_slide.append(transcriptPdSpearmanCor_slide)

                print("Slide test: ", f"; slide length = {slide_len}; overlap sequence length = {len(transcriptPd[slide_len:])}", "; pearsonCor:", transcriptPdpearsonCor_slide, "; SpearmanCor: ", transcriptPdSpearmanCor_slide)
            
            print(f"Sequence length: {len(transcriptPd)}; signalList length: {len(signalList)}", )
            # print signal and predicted signal
            transcriptPdpearsonCor = scipy.stats.pearsonr(transcriptPd, signalList) #np.exp()-1
            transcriptPdSpearmanCor = scipy.stats.spearmanr(transcriptPd, signalList)

            #print(f"The M1 spearmanCor of {transcriptName} is {subseqPadPdMergeCor[0]}")
            corPearson.append(transcriptPdpearsonCor[0])
            corSpearman.append(transcriptPdSpearmanCor[0])
            # print(f"The M1 correlation of {transcriptName}:  pearson={transcriptPdpearsonCor[0]};  spearman={transcriptPdSpearmanCor[0]}")
            #print(transcriptName,(np.around(np.exp(transcriptPd)-1, decimals=2)).tolist(),signalList, file=predictOP, sep="\t")

            '''
            signalPeakMark, peakCount = peakMark(signalList)
            peakCountList.append(peakCount)
            
            if(peakCount == 0):
                #roc_auc.append(0)
                #roc_auc_prc.append(0)
                continue
            '''
            ## ROC-AUC
            fpr, tpr, thresholds = metrics.roc_curve(y_true=labelList, y_score = transcriptPd, pos_label=1)
            aucValue = metrics.auc(fpr, tpr)
            roc_auc.append(aucValue)

            # ROCAUCPRC
            AUCPRCValue = average_precision_score(y_true=labelList, y_score=transcriptPd, pos_label=1)
            roc_auc_prc.append(AUCPRCValue)

            if args.Signal_OP:
                print(transcriptName,(np.around(np.exp(transcriptPd)-1, decimals=2)).tolist(),signalList, file=predictOP, sep="\t")
                print(transcriptName, transcriptPdpearsonCor[0], transcriptPdSpearmanCor[0], aucValue, AUCPRCValue, file=predictCorOP, sep="\t")
            # print(f"The M1 correlation of {transcriptName}:  pearson={transcriptPdpearsonCor[0]};  spearman={transcriptPdSpearmanCor[0]}; aucValue={aucValue}; AUCPRCValue={AUCPRCValue}")
        if transcriptName in  TPList[0]: # 第一条转录本输出图
            

            def moving_average(data, smooth_window):
                return np.convolve(data, np.ones(smooth_window)/smooth_window, mode='valid')

            # 示例
            '''
            # data = transcriptPd # np.random.randn(100)  # 替换为你的向量数据
            smooth_window = 3
            smoothed_data = moving_average(transcriptPd, smooth_window)
            print(len(transcriptPd), len(smoothed_data))
            smoothed_data = [0]*(smooth_window//2) + smoothed_data.tolist() + [0]*(smooth_window//2)
            '''

            # 绘制原始曲线和平滑后的曲线
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 8), sharex=True)

            ax = axes[0]
            ax.plot(range(1, len(signalList) + 1), np.log2(np.array(signalList)), '.', markersize=2, label='signal', color = point_colors(0))

            ax = axes[1]
            ax.plot(range(1, len(transcriptPd) + 1), transcriptPd, '.', markersize=2, label='predicted signal', color = point_colors(0))
            '''
            ax = axes[2]
            ax.plot(range(1, len(smoothed_data) + 1), smoothed_data, '.', markersize=2, label='smooth predicted signal', color = point_colors(0))
            '''
            ax = axes[2]
            ax.plot(range(1, len(labelList) + 1), labelList, '.', markersize=2, label='labelList', color = point_colors(0))

            # 隐藏右边框和上边框
            for ax in axes:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

            plt.savefig(f'{args.output}/{transcriptName}.png')
            plt.show()

    OP = open(f"{args.outputFolder}/EPRB_alleCLIP_modelTest.txt", "a+")
    #print(f"Pearson correlation: {np.nanmean(corPearson)}, spearman correlation: {np.nanmean(corSpearman)}, ROCAUC: {np.nanmean(roc_auc)}, ROCAUCPRC: {np.nanmean(roc_auc_prc)}")
    if(slide_len != 0):
        print(loss, protein, cell, np.nanmean(corPearson), np.nanmean(corSpearman), np.nanmean(roc_auc), np.nanmean(roc_auc_prc),np.nanmean(corPearson_slide), np.nanmean(corSpearman_slide), sep="\t", file=OP)
    else:
        print(loss, f"log={log}", protein, cell, np.nanmean(corPearson), np.nanmean(corSpearman), np.nanmean(roc_auc), np.nanmean(roc_auc_prc), sep="\t", file=OP)
    OP.close()
    print(loss, protein, cell, np.nanmean(corPearson), np.nanmean(corSpearman), np.nanmean(roc_auc), np.nanmean(roc_auc_prc), sep="\t")
    # ['ENSG00000104517.13']


# In[ ]:


# args.transcriptBed = f"/data/xliu/work/20231211_iDeepB/20240307_transcript_predict/TP_with_peak/TP_with_peak/QKI_HepG2_ENCSR570WLM_TP_with_peak_rmDup.txt"


# In[ ]:


TP_metric_data = {'RBPList': RBP_name, 'TPList': TP_name, 'corPearson': corPearson, 'corSpearman': corSpearman, 'roc_auc': roc_auc, 'roc_auc_prc': roc_auc_prc}

TP_metric_data = pd.DataFrame(TP_metric_data)
# 删除包含NaN的行
TP_metric_nan = TP_metric_data.dropna()
print("Transcript number: ", TP_metric_nan.shape)
TP_metric_nan[['corPearson', 'corSpearman', 'roc_auc', 'roc_auc_prc']].describe()

TP_metric_nan.to_csv(f"{args.outputFolder}/iDeepB_predict_transcript_head_statistics_all.txt", index = False, sep ="\t")
