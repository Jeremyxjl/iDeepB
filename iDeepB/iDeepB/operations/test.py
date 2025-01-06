
import torch
import numpy as np
import sys


sys.path.append('/data/xliu/work/') 
from iDeepB.iDeepB.utils.functions import onehot_encode

# for two head
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

        if mode == "Fragment":
            if remainder:
                for subseqOHTurn in subseqOH[:-1].split(1000,  0): # batch
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
                for subseqOHTurn in subseqOH.split(1000,  0): #预测转录本，数据太大，无法预
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


# for two head
def predict1Head(seqTranscribed, window_size, signalList, model, mode, codeModel = "OneHot"):  #mode "Fragment" midBase

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

        if mode == "Fragment":
            if remainder:
                for subseqOHTurn in subseqOH[:-1].split(1000,  0): # batch
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
                for subseqOHTurn in subseqOH.split(1000,  0): #预测转录本，数据太大，无法预
                    treat_predict = model(subseqOHTurn) #torch.Size([n, 101])
                    treat_predict = treat_predict.detach().cpu().numpy().tolist()

                    temp.extend(treat_predict)
                temp = np.array(temp).reshape(-1).tolist()   

        elif mode == "midBase":  #column_values = tensor_2d[:, 1]
            for subseqOHTurn in subseqOH.split(1000,  0): #预测转录本，数据太大，无法预 [0:-0] 非[1:-1]
                treat_predict = model(subseqOHTurn) #torch.Size([n, 101])
                treat_predict = treat_predict[:, window_size//2]
                treat_predict = treat_predict.detach().cpu().numpy().tolist()

                temp.extend(treat_predict)

            # for remainder length
            # 对于前半段 remainder_index == 0
            remainder_index = 0
            treat_predict_end = model(subseqOH[remainder_index].unsqueeze(0)) #torch.Size([n, 101])
            
            treat_predict_end = treat_predict_end.detach().cpu().numpy()
            
            temp = treat_predict_end[:, :window_size//2].reshape(-1).tolist() + temp

            # 对于后半段
            remainder_index = -1
            treat_predict_end = model(subseqOH[remainder_index].unsqueeze(0)) #torch.Size([n, 101])
  
            treat_predict_end = treat_predict_end.detach().cpu().numpy()

            temp = treat_predict_end[:, window_size//2:].reshape(-1).tolist() + temp
                

        transcriptPd = np.array(temp) #subseqPredicted.cpu().detach().numpy()
        # cor 1: subseq预测结果合并成一个向量，然后与实际信号计算相关性
        # transcriptPd = subseqPredictedNp.reshape(-1)
        print("Signal lenth of prediction and transcript length !", len(transcriptPd), len(signalList))
        if(len(transcriptPd) != len(signalList)):
            print("Error: signal lenth not equal transcript length !", len(transcriptPd), len(signalList))
            return False
    return transcriptPd
