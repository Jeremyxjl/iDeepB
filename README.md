# iDeepB: Base-resolution binding profile prediction of proteins on RNAs with deep learning

## Dependency
The detailed versions of the dependencies required by iDeepB can be found in the environment.yml file.

python=3.9.12 \
torch==1.11.0+cu113

## Usage:
### Train without control

```
python -u ideepB_train.py --lr 0.0004 --CUDA 2 --signalMax 0 --seqLength 101  --bsz 256 --encode OneHot --modelFramework Treat --epochs 100 -train ./data/QKI_HepG2_ENCSR570WLM/QKI_HepG2_ENCSR570WLM_chr3.h5 -validation ./data/QKI_HepG2_ENCSR570WLM/QKI_HepG2_ENCSR570WLM_chr9.h5 --task QKI_HepG2_ENCSR570WLM --lossM poissonLoss --output ideepB_trained_model/QKI_HepG2_ENCSR570WLM_c0 
```

The sample filtering can be achieved using the parameter combination --filter --signalMax 0, which removes samples with low maximum signal values.
```
python -u ideepB_train.py --lr 0.0004 --CUDA 2 --filter --signalMax 2 --seqLength 101  --bsz 256 --encode OneHot --modelFramework Treat --epochs 100 -train ./data/QKI_HepG2_ENCSR570WLM/QKI_HepG2_ENCSR570WLM_chr3.h5 -validation ./data/QKI_HepG2_ENCSR570WLM/QKI_HepG2_ENCSR570WLM_chr9.h5 --task QKI_HepG2_ENCSR570WLM --lossM poissonLoss --output ideepB_trained_model/QKI_HepG2_ENCSR570WLM_c0 
```

### Train with control
```
python -u ideepB_train.py --lr 0.0004 --CUDA 2 --signalMax 0 --seqLength 101  --bsz 256 --encode OneHot --modelFramework Control --epochs 100 -train ./data/QKI_HepG2_ENCSR570WLM/QKI_HepG2_ENCSR570WLM_chr3.h5 -validation ./data/QKI_HepG2_ENCSR570WLM/QKI_HepG2_ENCSR570WLM_chr9.h5 --task QKI_HepG2_ENCSR570WLM --lossM poissonLoss --output ideepB_trained_model/QKI_HepG2_ENCSR570WLM_c0 
```

### Predicting binding profile without control
```
python ideepB_predict.py --modelFramework Treat 
```

### Predicting binding profile with control
```
python ideepB_predict.py --modelFramework control 
```


## Reference
Contact: Xiaoyong Pan (xypan172436@gmail.com)
