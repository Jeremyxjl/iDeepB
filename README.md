# iDeepB

iDeepB: Base-resolution binding profile prediction of proteins on RNAs with deep learning

## Dependency
The detailed versions of the dependencies required by iDeepB can be found in the environment.yml file.

python=3.9.7 \
torch==1.11.0+cu113

## Preparation

iDeepB is implemented with Python3, so a Python3 (3.9.7) interpreter is required. At first, download the source code of iDeepB from Github:

```
git clone https://github.com/Jeremyxjl/iDeepB.git
```

Then, we recommend you to use a virtual environment, such as Anaconda, to install the dependencies of iDeepB:

```
conda create -n iDeepB python=3.9.7
conda activate iDeepB
```

## Usage:
### Train without control

```
python -u ideepB_train.py --lr 0.0004 --CUDA 2 --signalMax 0 --seqLength 101  --bsz 256 --encode OneHot --modelFramework Treat --epochs 100 -train ./data/QKI_HepG2_ENCSR570WLM/QKI_HepG2_ENCSR570WLM_chr3.h5 -validation ./data/QKI_HepG2_ENCSR570WLM/QKI_HepG2_ENCSR570WLM_chr9.h5 --task QKI_HepG2_ENCSR570WLM --lossM poissonLoss --output ideepB_trained_model/QKI_HepG2_ENCSR570WLM_c0 
```

The sample filtering can be achieved using the parameter combination --filter --signalMax 0, which removes samples with low maximum signal values.
```
python -u ideepB_train.py --lr 0.0004 --CUDA 2 --filter --signalMax 2 --seqLength 101  --bsz 256 --encode OneHot --modelFramework Treat --epochs 100 -train ./data/QKI_HepG2_ENCSR570WLM/QKI_HepG2_ENCSR570WLM_chr13.h5 -validation ./data/QKI_HepG2_ENCSR570WLM/QKI_HepG2_ENCSR570WLM_chr9.h5 --task QKI_HepG2_ENCSR570WLM --lossM poissonLoss --output ideepB_trained_model/QKI_HepG2_ENCSR570WLM_c0 
```

### Train with control
```
python -u ideepB_train.py --lr 0.0004 --CUDA 2 --signalMax 0 --seqLength 101  --bsz 256 --encode OneHot --modelFramework Control --epochs 100 -train ./data/QKI_HepG2_ENCSR570WLM/QKI_HepG2_ENCSR570WLM_chr13.h5 -validation ./data/QKI_HepG2_ENCSR570WLM/QKI_HepG2_ENCSR570WLM_chr9.h5 --task QKI_HepG2_ENCSR570WLM --lossM poissonLoss --output ideepB_trained_model/QKI_HepG2_ENCSR570WLM_c0 
```

### Predicting binding profile 
For a sequence of length 101 bp, the prediction code is shown below:
```
import numpy as np
import torch

from iDeepB.models.iDeepBModel import iDeepB
from iDeepB.utils.functions import onehot_encode
from iDeepB.plot.track import plot_single_track

class args:
    pass

args.head = "Treat"
args.CUDA = 3
args.task = "QKI_HepG2_ENCSR570WLM"
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(args.head == "Treat"): 
    print("Model: ", args.head)
    model = iDeepB(dropOut = 0, control = False).to(args.device)
elif(args.head == "Control"): 
    print("Model: ", args.head)
    model = iDeepB(dropOut = 0, control = True).to(args.device)

model.load_state_dict(torch.load(f'data/QKI_HepG2_ENCSR570WLM/model_params.pth'))

sequence = 'UUGGUUGUUUAUCUGAGAUUCAGAAUUAAGCAUUUUAUAUUUUAUUUGCUGCCUCUGGCCACCCUACUCUCUUCCUAACACUCUCUCCCUCUCCCAGUUUU'
vocab = list("AUGC")
seqsInt = onehot_encode([sequence], vocab, 4)
subseqOH = torch.from_numpy(np.asarray(seqsInt) ).to(args.device)

outputs = model(subseqOH.float()) 

outputs = np.array(outputs.detach().cpu().numpy().tolist()).reshape(-1).tolist()   
```

Read sequences from a FASTA file and predict the binding between protein and RNA sequences.
```

import os
import torch

from iDeepB.models.iDeepBModel import iDeepB
from iDeepB.utils.functions import read_fasta_file 
from iDeepB.operations.predictFunction import predict1HeadFromSeq, predict2HeadWFromSeq
from iDeepB.plot.track import plot_single_track

seqs, names = read_fasta_file("./data/QKI_HepG2_ENCSR570WLM/example.fa")

class args:
    pass

args.head = "Treat"
args.CUDA = 3
args.task = "QKI_HepG2_ENCSR570WLM"
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(args.head == "Treat"): 
    print("Model: ", args.head)
    model = iDeepB(dropOut = 0, control = False).to(args.device)
    predictHead = predict1HeadFromSeq
elif(args.head == "Control"): 
    print("Model: ", args.head)
    model = iDeepB(dropOut = 0, control = True).to(args.device)
    predictHead = predict2HeadWFromSeq

model.load_state_dict(torch.load(f'data/{args.task}/model_params.pth'))

for sequence in seqs[3:4]:
        prediction = predict1HeadFromSeq(sequence, 101, model, mode="Fragment", device = args.device)

        plot_single_track(prediction)
    
```


## Online service

We also provide online retrieval service [here](http://www.csbio.sjtu.edu.cn/bioinf/iDeepB/).
