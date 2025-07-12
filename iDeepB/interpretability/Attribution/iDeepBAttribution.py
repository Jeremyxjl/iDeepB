# https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from ...utils.functions import onehot_encode
import captum

# Define a list of supported and unsupported activations
SUPPORTED_ACTIVATIONS = [
    'relu', 'elu', 'sigmoid', 'tanh', 'softplus'
]

UNSUPPORTED_ACTIVATIONS = [
    'crelu', 'relu6', 'softsign'
]

# Utility function to get the activation function by name
def activation(type):
    if type not in SUPPORTED_ACTIVATIONS:
        warnings.warn(f'Activation function ({type}) not supported')
    return getattr(nn.functional, type)

# Base class for attribution methods
class AttributionMethod(object):
    def __init__(self, model):
        self.model = model
        # self.model.eval()
    
    def explain(self, input, target=None):
        pass

# Gradient-based attribution method
class GradientBasedMethod(AttributionMethod):
    def __init__(self, model):
        super(GradientBasedMethod, self).__init__(model)
    
    def explain(self, input, target=None):
        input.requires_grad = True
        output = self.model(input)
        if target is None:
            target = torch.ones_like(output)  # Default to a target of ones

        output.backward(gradient=target)
        return input.grad

# Saliency attribution method
class Saliency(GradientBasedMethod):
    def explain(self, input, target=None):
        gradient = super(Saliency, self).explain(input, target)
        return torch.abs(gradient)

# Gradient * Input attribution method
class GradientXInput(GradientBasedMethod):
    def explain(self, input, target=None):
        gradient = super(GradientXInput, self).explain(input, target)
        return gradient # gradient * input

# Integrated Gradients attribution method
class IntegratedGradients(GradientBasedMethod):
    def __init__(self, model, steps=100):
        super(IntegratedGradients, self).__init__(model)
        self.steps = steps

    def explain(self, input, target=None):
        # assert target is not None, "Target tensor must be provided for Integrated Gradients"
        device = input.device
        input = input.detach()
        input.requires_grad = False
        input = input.to(device)
        input.requires_grad = True
        
        baseline = torch.zeros_like(input)
        step_sizes = (input - baseline) / self.steps

        integrated_gradients = None
        for alpha in torch.linspace(0.0, 1.0, steps=self.steps):
            x_step = baseline + alpha * (input - baseline)

            device = x_step.device
            x_step = x_step.detach()
            x_step.requires_grad = False
            x_step = x_step.to(device)
            x_step.requires_grad = True
            
            output_step = self.model(x_step)
            output_step.backward(gradient=target, retain_graph=True)
            if integrated_gradients is None:
                integrated_gradients = input.grad.clone().detach()
            else:
                integrated_gradients += input.grad

        integrated_gradients *= step_sizes
        return integrated_gradients

# iDeepB
def iDeepB_integrated_gradients_fast(model, input_tensor, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Generate scaled inputs along the interpolation path
    scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
    scaled_inputs = torch.stack(scaled_inputs).requires_grad_() 

    # Initialize integrated gradients as zero
    integrated_grads = torch.zeros_like(input_tensor)

    # Compute gradients
    output = model(scaled_inputs)
    output_sum = torch.sum(output, dim=tuple(range(1, output.dim())))
    integrated_grads = torch.autograd.grad(output_sum, scaled_inputs, create_graph=True, grad_outputs=torch.ones_like(output_sum))[0]

    integrated_grads = integrated_grads / steps   #

    integrated_grads_local = integrated_grads.sum(axis = 0)

    # Scale the integrated gradients by the difference between input and baseline
    integrated_grads_global = (input_tensor - baseline) * integrated_grads_local
    
    return integrated_grads_global, integrated_grads_local


def iDeepB_integrated_gradients_fast_interpret_target(model, input_tensor, target, steps=50, baseline=None):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Generate scaled inputs along the interpolation path
    scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
    scaled_inputs = torch.stack(scaled_inputs).requires_grad_() 

    # Initialize integrated gradients as zero
    integrated_grads = torch.zeros_like(input_tensor)

    # Compute gradients
    output = model(scaled_inputs)

    if target is None:
        target = torch.ones_like(output)

    output = output * target
    output_sum = torch.sum(output, dim=tuple(range(1, output.dim())))
    integrated_grads = torch.autograd.grad(output_sum, scaled_inputs, create_graph=True, grad_outputs=torch.ones_like(output_sum))[0]

    # Accumulate the gradients
    integrated_grads = integrated_grads / steps

    integrated_grads_local = integrated_grads.mean(axis = 0)

    # Scale the integrated gradients by the difference between input and baseline
    integrated_grads_global = (input_tensor - baseline) * integrated_grads_local
    
    return integrated_grads_global, integrated_grads_local


def iDeepB_integrated_gradients_fast_interpret(model, input_tensor, topN, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Generate scaled inputs along the interpolation path
    scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
    scaled_inputs = torch.stack(scaled_inputs).requires_grad_() 

    # Initialize integrated gradients as zero
    integrated_grads = torch.zeros_like(input_tensor)

    # Compute gradients
    output = model(scaled_inputs)

    output_squeezed = model(input_tensor.unsqueeze(0)).squeeze(0)
    Top_n_value, Top_n_indices = output_squeezed.topk(topN, dim=0)
    Top_n_indice_vector = torch.zeros_like(output_squeezed)
    Top_n_indice_vector[Top_n_indices] = 1

    output = output * Top_n_indice_vector
    output_sum = torch.sum(output, dim=tuple(range(1, output.dim())))
    integrated_grads = torch.autograd.grad(output_sum, scaled_inputs, create_graph=True, grad_outputs=torch.ones_like(output_sum))[0]

    # Accumulate the gradients
    integrated_grads = integrated_grads / steps

    integrated_grads_local = integrated_grads.sum(axis = 0)

    # Scale the integrated gradients by the difference between input and baseline
    integrated_grads_global = (input_tensor - baseline) * integrated_grads_local
    
    return integrated_grads_global, integrated_grads_local


def attributionFromSeq(interpMethod, device, input, baseline, model, target=None, topN = None):
        
    if(interpMethod == "Top_IntegratedGradients"):

            ig = captum.attr.IntegratedGradients(model) 
            contrib_scores_integration = np.zeros(input.shape, dtype = float) 
            output_squeezed = model(input).squeeze(0)
            Top_n_value, Top_n_indices = output_squeezed.topk(topN, dim=0)
            Top_n_indice_vector = torch.zeros_like(output_squeezed)
            Top_n_indice_vector[Top_n_indices] = 1
            Top_n_indice_vector = Top_n_indice_vector.cpu().detach().numpy()

            for target_index in np.where(np.array(Top_n_indice_vector) > 0)[0]:
                attributions, delta = ig.attribute(input, baseline, target=int(target_index), return_convergence_delta=True)
                contrib_scores_integration = contrib_scores_integration + attributions.cpu().detach().numpy()

            return contrib_scores_integration*(input.cpu().detach().numpy())

    elif(interpMethod == "Top_Saliency"):
        ig = captum.attr.Saliency(model) 
        contrib_scores_integration = np.zeros(input.shape, dtype = float) # np.full_like(subseqOH.shape, 0)

        output_squeezed = model(input).squeeze(0)
        Top_n_value, Top_n_indices = output_squeezed.topk(topN, dim=0)
        Top_n_indice_vector = torch.zeros_like(output_squeezed)
        Top_n_indice_vector[Top_n_indices] = 1
        Top_n_indice_vector = Top_n_indice_vector.cpu().detach().numpy()

        for target_index in np.where(np.array(Top_n_indice_vector) > 0)[0]:
            attributions= ig.attribute(input, target=int(target_index))
            contrib_scores_integration = contrib_scores_integration + attributions.cpu().detach().numpy()

        return contrib_scores_integration*(input.cpu().detach().numpy())
    
    elif(interpMethod == "TopFast_Saliency"):

        output_squeezed = model(input).squeeze(0)
        Top_n_value, Top_n_indices = output_squeezed.topk(2, dim=0)
        Top_n_indice_vector = torch.zeros_like(output_squeezed)
        Top_n_indice_vector[Top_n_indices] = 1

        attribution_method = Saliency(model.to(device))
        attributions_local = attribution_method.explain(input.float().to(device), target = Top_n_indice_vector.unsqueeze(0))
        attributions_global = attributions_local * input

        return attributions_global.cpu().detach().numpy()
    
    elif(interpMethod == "Saliency"):

        attribution_method = Saliency(model.to(device))
        attributions_local = attribution_method.explain(input.float().to(device))
        attributions_global = attributions_local * input

        return attributions_global.cpu().detach().numpy()  #GradientXInput
    elif(interpMethod == "GradientXInput"):
        attribution_method = GradientXInput(model.to(device))
        attributions_local = attribution_method.explain(input.float().to(device))
        attributions_global = attributions_local * input

        return attributions_global.cpu().detach().numpy()  #GradientXInput

    elif(interpMethod == "IntegratedGradients"):

        attributions = iDeepB_integrated_gradients_fast(model = model.to(device), input_tensor = input.float().to(device).squeeze(axis =0 ), baseline = baseline.squeeze(axis =0 ))
        contrib_scores_integration = attributions[0].unsqueeze(axis =0 ).cpu().detach().numpy()

    elif(interpMethod == "TopFastTarget_IntegratedGradients"):
        attributions = iDeepB_integrated_gradients_fast_interpret_target(model = model.to(device), input_tensor = input.float().to(device).squeeze(axis =0 ), baseline = baseline.squeeze(axis =0 ), target= target)
        contrib_scores_integration = attributions[0].unsqueeze(axis =0 ).cpu().detach().numpy()

    elif(interpMethod == "TopFast_IntegratedGradients"):
        
        attributions = iDeepB_integrated_gradients_fast_interpret(model = model.to(device), input_tensor = input.float().to(device).squeeze(axis =0 ), baseline = baseline.squeeze(axis =0 ), topN = topN)
        contrib_scores_integration = attributions[0].unsqueeze(axis =0 ).cpu().detach().numpy()

    else:
        raise KeyError("method error")
    
    return contrib_scores_integration



def interpret1HeadFromSeq(seqTranscribed, window_size, model, method, codeModel = "OneHot", device = None, topN = None):  #mode "Fragment" midBase

    ## padding and extract subSeq or subSignal
    remainder = (len(seqTranscribed) % window_size)
    print(f"## remainder Length is: {remainder}")

    # 计算列表可以分成多少个子列表
    num_sublists = len(seqTranscribed) // window_size

    # 使用列表推导式创建子列表
    sublists = [seqTranscribed[i * window_size: (i + 1) * window_size] for i in range(num_sublists)]

    # 如果剩余元素不为零，将其加入最后一个子列表
    remainder = len(seqTranscribed) % window_size
    if remainder:
        print("remainder: ", remainder)
        sublists.append(seqTranscribed[-window_size:]) 

    if codeModel == "OneHot":
        vocab = list("AUGC")
        seqsInt = onehot_encode(sublists, vocab, 4)
    subseqOH = torch.from_numpy(np.asarray(seqsInt) ).to(device)

    baseline = torch.zeros(1, 101, 4).to(device)

    temp = []
    if(codeModel == "Embedding"):
        subseqOH = subseqOH.int()
    elif(codeModel == "OneHot"):
        subseqOH = subseqOH.float()

    if remainder:
        for subseqOHTurn in subseqOH[:-1].split(1,  0): # batch

            input = subseqOHTurn.to(device)
            treat_predict = attributionFromSeq(interpMethod= method, device=device, input = input, baseline=baseline, model=model, topN = topN)
            temp.extend(treat_predict)

        treat_predict_end = attributionFromSeq(interpMethod= method, device=device, input = subseqOH[-1:].to(device), baseline=baseline, model=model, topN = topN)

        # temp = temp + np.array(treat_predict_end).reshape(-1).tolist()[-remainder:]
        temp.extend(treat_predict_end[:, -remainder:, :])
        temp = np.concatenate(temp, axis=0)

        # 打印 temp 的形状
        print("temp shape: ", temp.shape, treat_predict_end[:, -remainder:, :].shape)
        
    else:
        for subseqOHTurn in subseqOH.split(1000,  0): #预测转录本，数据太大，无法预
            print("test: ", subseqOHTurn.shape, subseqOH.shape)
            treat_predict = attributionFromSeq(interpMethod= method, device=device, input = subseqOHTurn.to(device), baseline=baseline, model=model, topN = topN)

            print("test2: ", subseqOHTurn.shape, subseqOH.shape, treat_predict.shape)
            temp.extend(treat_predict)
        temp = np.concatenate(temp, axis=0)    
        print("temp shape: ", temp.shape,) 

    transcriptPd = temp 
    print("Signal lenth of prediction and transcript length !", len(transcriptPd), len(seqTranscribed))
    if(len(transcriptPd) != len(seqTranscribed)):
        print("Error: signal lenth not equal transcript length !", len(transcriptPd), len(seqTranscribed))
        return False
    return transcriptPd

# Example usage:
if __name__ == "__main__":
    # Create a PyTorch model (replace with your own model)
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 2)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = MyModel()

    # Create an attribution method and explain a sample input
    input = torch.randn(1, 10)
    attribution_method = GradientXInput(model)
    attribution = attribution_method.explain(input)

    print(attribution)

    # Example usage:
    # Create a PyTorch model (replace with your own model)
    class MyModel(torch.nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.fc1 = torch.nn.Linear(10, 5)
            self.fc2 = torch.nn.Linear(5, 2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = MyModel()

    # Create an attribution method and explain a sample input
    input = torch.randn(1, 10)*3
    attribution_method = IntegratedGradients(model)
    attribution = attribution_method.explain(input)

    print(attribution)
