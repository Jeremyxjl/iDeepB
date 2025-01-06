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
        # self.model.zero_grad()
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
        return gradient * input

# Integrated Gradients attribution method
class IntegratedGradients(GradientBasedMethod):
    def __init__(self, model, steps=100):
        super(IntegratedGradients, self).__init__(model)
        self.steps = steps

    def explain(self, input, target=None):
        # assert target is not None, "Target tensor must be provided for Integrated Gradients"
        # input.requires_grad = True
        device = input.device
        input = input.detach()
        input.requires_grad = False
        input = input.to(device)
        input.requires_grad = True

        # output = self.model(input)
        # self.model.zero_grad()
        
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

#############################################################################
# rbpnet igrads
# https://github.com/mhorlacher/igrads
import torch
import torch.nn.functional as F
import torch.autograd as autograd

class igrads_AttributionMethod(object):
    def __init__(self, model):
        self.model = model
        # self.model.eval()

    def explain(self, input, target_mask=None, postproc_fn=None, baseline=None, steps=50):
        pass

class igrads_IntegratedGradients(igrads_AttributionMethod):
    def __init__(self, model, steps=50):
        super(igrads_IntegratedGradients, self).__init__(model)
        self.steps = steps

    def _interpolate_inputs(self, inputs, baseline):
        alphas = torch.linspace(0.0, 1.0, steps=self.steps+1)
        delta = inputs - baseline
        interpolated_inputs = torch.stack([baseline + delta * alpha for alpha in alphas])
        return interpolated_inputs

    def _apply_fn(self, fn, inputs):
        if isinstance(inputs, torch.Tensor):
            return fn(inputs)
        elif isinstance(inputs, list):
            return [fn(x) for x in inputs]
        elif isinstance(inputs, dict):
            return {key: fn(value) for key, value in inputs.items()}
        else:
            raise ValueError('Unsupported type: {}'.format(type(inputs)))

    def _mask(self, x, mask):
        return x * mask

    def _apply_mask(self, x, mask):
        return self._apply_fn(lambda y: self._mask(y, mask), x)

    def _compute_gradients(self, inputs, target_mask=None, postproc_fn=None):
        
        # inputs.requires_grad = True
        # 先放置到CUDA，然后设置requires_grad = True，才不会报错
        device = inputs.device
        inputs = inputs.detach()
        inputs.requires_grad = False
        print(inputs.requires_grad,inputs.is_leaf, inputs.grad)
        inputs = inputs.to(device)
        inputs.requires_grad = True
        
        pred = self.model(inputs)

        if postproc_fn is not None:
            pred = self._apply_fn(postproc_fn, pred)

        if target_mask is not None:
            pred = self._apply_mask(pred, target_mask)

        print("pred shape:", pred.shape)
        pred_sum = pred.sum(dim=tuple(range(1, pred.dim())))
        # pred_sum = pred.sum(dim=tuple(range(1, pred.dim())))
        print("pred_sum shape:", pred_sum.shape, pred_sum)

        # gradients = autograd.grad(pred_sum, inputs, create_graph=True)[0]
        gradients = autograd.grad(pred_sum, inputs, create_graph=True, grad_outputs=torch.ones_like(pred_sum))[0]
        return gradients


    def _integral_approximation(self, gradients):
        grads = self._apply_fn(lambda x: (x[:-1] + x[1:]) / torch.tensor(2.0), gradients)
        integrated_gradients = grads.mean(dim=0)
        return integrated_gradients

    def explain(self, inputs, target_mask=None, postproc_fn=None, baseline=None):
        if baseline is None:
            baseline = torch.zeros_like(inputs)

        print("inputs shape:", inputs.shape)
        interpolated_inputs = self._interpolate_inputs(inputs, baseline)
        print("interpolated_inputs shape:", interpolated_inputs.shape)
        gradients = self._compute_gradients(interpolated_inputs, target_mask, postproc_fn)
        print("gradients:", gradients.shape)
        integrated_gradients = self._integral_approximation(gradients)
        print("integrated_gradients:", integrated_gradients.shape)
        integrated_gradients = self._apply_fn(lambda x: (inputs - baseline) * x, integrated_gradients)
        print("integrated_gradients:", integrated_gradients.shape)

        return integrated_gradients
    
class igrads_IntegratedGradients2(igrads_AttributionMethod):
    def __init__(self, model, steps=50):
        super(igrads_IntegratedGradients2, self).__init__(model)
        self.steps = steps

    def _interpolate_inputs(self, inputs, baseline):
        alphas = torch.linspace(0.0, 1.0, steps=self.steps+1)
        delta = inputs - baseline
        interpolated_inputs = torch.stack([baseline + delta * alpha for alpha in alphas])
        return interpolated_inputs

    def _apply_fn(self, fn, inputs):
        if isinstance(inputs, torch.Tensor):
            return fn(inputs)
        elif isinstance(inputs, list):
            return [fn(x) for x in inputs]
        elif isinstance(inputs, dict):
            return {key: fn(value) for key, value in inputs.items()}
        else:
            raise ValueError('Unsupported type: {}'.format(type(inputs)))

    def _mask(self, x, mask):
        return x * mask

    def _apply_mask(self, x, mask):
        return self._apply_fn(lambda y: self._mask(y, mask), x)

    def _compute_gradients(self, inputs, target_mask=None, postproc_fn=None):
        
        # inputs.requires_grad = True
        # 先放置到CUDA，然后设置requires_grad = True，才不会报错
        device = inputs.device
        inputs = inputs.detach()
        inputs.requires_grad = False
        print(inputs.requires_grad,inputs.is_leaf, inputs.grad)
        inputs = inputs.to(device)
        inputs.requires_grad = True
        
        pred = self.model(inputs)

        if postproc_fn is not None:
            pred = self._apply_fn(postproc_fn, pred)

        if target_mask is not None:
            pred = self._apply_mask(pred, target_mask)

        print("pred shape:", pred.shape)
        pred_sum = pred.sum(dim=tuple(range(1, pred.dim())))
        # pred_sum = pred.sum(dim=tuple(range(1, pred.dim())))
        print("pred_sum shape:", pred_sum.shape, pred_sum)

        # gradients = autograd.grad(pred_sum, inputs, create_graph=True)[0]
        # gradients = autograd.grad(pred_sum, inputs, create_graph=True, grad_outputs=torch.ones_like(pred_sum))[0]
        gradients = self._apply_fn(lambda x: torch.autograd.grad(outputs=x, inputs=inputs, grad_outputs=torch.ones_like(x), retain_graph=True)[0], pred_sum)

        return gradients


    def _integral_approximation(self, gradients):
        grads = self._apply_fn(lambda x: (x[:-1] + x[1:]) / torch.tensor(2.0), gradients)
        integrated_gradients = grads.mean(dim=0)
        return integrated_gradients

    def explain(self, inputs, target_mask=None, postproc_fn=None, baseline=None):
        if baseline is None:
            baseline = torch.zeros_like(inputs)

        print("inputs shape:", inputs.shape)
        interpolated_inputs = self._interpolate_inputs(inputs, baseline)
        print("interpolated_inputs shape:", interpolated_inputs.shape)
        gradients = self._compute_gradients(interpolated_inputs, target_mask, postproc_fn)
        print("gradients:", gradients.shape)
        integrated_gradients = self._integral_approximation(gradients)
        print("integrated_gradients:", integrated_gradients.shape)
        integrated_gradients = self._apply_fn(lambda x: (inputs - baseline) * x, integrated_gradients)
        print("integrated_gradients:", integrated_gradients.shape)

        return integrated_gradients
    
class circsite_IntegratedGradients():
    """
        Produces gradients generated with integrated gradients from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.train()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        self.model.register_backward_hook(hook_function)

    def generate_images_on_linear_path(self, input_image, steps):
        # Generate uniform numbers between 0 and steps
        step_list = np.arange(steps+1)/steps
        # Generate scaled xbar images
        xbar_list = [input_image*step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image):
        # Forward
        # input_image.requires_grad=True

        device = input_image.device
        input_image = input_image.detach()
        input_image.requires_grad = False
        input_image = input_image.to(device)
        input_image.requires_grad = True

        model_output = self.model(input_image)
        # Zero grads
        # self.model.zero_grad()
        gradients_as_arr = torch.autograd.grad(outputs=model_output, inputs=input_image)
        # print('gradients_as_arr: ', gradients_as_arr)
        return gradients_as_arr

    def generate_integrated_gradients(self, input_image, steps=50):
        # Generate xbar images
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        # Initialize an iamge composed of zeros
        integrated_grads = np.zeros(input_image.size())
        # print('input_image.size(): ', input_image.size())
        for i, xbar_image in enumerate(xbar_list):
            xbar_image = xbar_image.unsqueeze(0)
            # Generate gradients from xbar images
            single_integrated_grad = self.generate_gradients(xbar_image)
            # Add rescaled grads from xbar images
            # integrated_grads = integrated_grads + np.multiply(input_image.detach().cpu().numpy(), single_integrated_grad[0].detach().cpu().numpy()/steps)
            integrated_grads = integrated_grads + single_integrated_grad[0].detach().cpu().numpy()/steps

        return integrated_grads[0]
    
import torch

def integrated_gradients(model, input_tensor, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    # Generate scaled inputs along the interpolation path
    scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
    scaled_inputs = torch.stack(scaled_inputs)
    
    # Initialize integrated gradients as zero
    integrated_grads = torch.zeros_like(input_tensor)
    
    for i in range(steps):
        # Compute gradients for each scaled input
        scaled_input = scaled_inputs[i].unsqueeze(0)  # Add batch dimension
        scaled_input.requires_grad = True
        output = model(scaled_input).sum()
        
        # Compute gradients
        # model.zero_grad()
        grads = torch.autograd.grad(outputs=output, inputs=scaled_input)[0]
        
        # Accumulate the gradients
        integrated_grads += grads.squeeze(0) / steps
    
    # Scale the integrated gradients by the difference between input and baseline
    integrated_grads = (input_tensor - baseline) * integrated_grads
    
    return integrated_grads

def iDeepB_integrated_gradients(model, input_tensor, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    # Generate scaled inputs along the interpolation path
    scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
    scaled_inputs = torch.stack(scaled_inputs).requires_grad_() 
    
    # Initialize integrated gradients as zero
    integrated_grads = torch.zeros_like(input_tensor)
    
    for i in range(steps):
        # Compute gradients for each scaled input
        scaled_input = scaled_inputs[i].unsqueeze(0)  # Add batch dimension
        # scaled_input.requires_grad = True
        output = model(scaled_input).sum()
        
        # Compute gradients
        # model.zero_grad()
        grads = torch.autograd.grad(outputs=output, inputs=scaled_input)[0]
        
        # Accumulate the gradients
        integrated_grads += grads.squeeze(0) / steps
    
    # Scale the integrated gradients by the difference between input and baseline
    integrated_grads = (input_tensor - baseline) * integrated_grads
    
    return integrated_grads

def iDeepB_integrated_gradients_fast(model, input_tensor, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Generate scaled inputs along the interpolation path
    scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
    scaled_inputs = torch.stack(scaled_inputs).requires_grad_() 

    # Initialize integrated gradients as zero
    integrated_grads = torch.zeros_like(input_tensor)

    # Compute gradients
    # model.zero_grad()
    output = model(scaled_inputs)
    output_sum = torch.sum(output, dim=tuple(range(1, output.dim())))
    # grads = torch.autograd.grad(outputs=output_sum, inputs=scaled_inputs)[0]
    integrated_grads = torch.autograd.grad(output_sum, scaled_inputs, create_graph=True, grad_outputs=torch.ones_like(output_sum))[0]

    # Accumulate the gradients
    # integrated_grads += grads.squeeze(0) / steps
    integrated_grads = integrated_grads / steps

    integrated_grads = integrated_grads.sum(axis = 0)

    # Scale the integrated gradients by the difference between input and baseline
    integrated_grads = (input_tensor - baseline) * integrated_grads
    
    return integrated_grads

# Other attribution methods can be similarly implemented

# 公式：(input_tensor - baseline) * 单个梯度/steps

# 公式：(input_tensor - baseline) * 单个梯度/steps

def iDeepB_integrated_gradients_fast(model, input_tensor, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Generate scaled inputs along the interpolation path
    scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
    scaled_inputs = torch.stack(scaled_inputs).requires_grad_() 

    # Initialize integrated gradients as zero
    integrated_grads = torch.zeros_like(input_tensor)

    # Compute gradients
    # model.zero_grad()
    output = model(scaled_inputs)
    output_sum = torch.sum(output, dim=tuple(range(1, output.dim())))
    # grads = torch.autograd.grad(outputs=output_sum, inputs=scaled_inputs)[0]
    integrated_grads = torch.autograd.grad(output_sum, scaled_inputs, create_graph=True, grad_outputs=torch.ones_like(output_sum))[0]

    # Accumulate the gradients
    # integrated_grads += grads.squeeze(0) / steps
    integrated_grads = integrated_grads / steps

    integrated_grads_local = integrated_grads.sum(axis = 0)

    # Scale the integrated gradients by the difference between input and baseline
    integrated_grads_global = (input_tensor - baseline) * integrated_grads_local
    
    return integrated_grads_global, integrated_grads_local

'''
# seq_index = 1 # 2 有motif
# GradientXInput Saliency IntegratedGradients GradientBasedMethod
attribution_matrix = iDeepB_integrated_gradients_fast(model = model.to(args.device), input_tensor = subseqOH[seq_index].float().to(args.device))[0].cpu().detach().numpy()
print(attribution_matrix.shape)

bases = ['A', 'C', 'G', 'U']
plot_sequence_attribution(attribution_matrix, bases = bases, width=20, height=4)
'''


def iDeepB_integrated_gradients_fast_interpret(model, input_tensor, topN, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Generate scaled inputs along the interpolation path
    scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
    scaled_inputs = torch.stack(scaled_inputs).requires_grad_() 

    # Initialize integrated gradients as zero
    integrated_grads = torch.zeros_like(input_tensor)

    # Compute gradients
    # model.zero_grad()
    output = model(scaled_inputs)

    output_squeezed = model(input_tensor.unsqueeze(0)).squeeze(0)
    Top_n_value, Top_n_indices = output_squeezed.topk(topN, dim=0)
    Top_n_indice_vector = torch.zeros_like(output_squeezed)
    Top_n_indice_vector[Top_n_indices] = 1

    output = output * Top_n_indice_vector
    output_sum = torch.sum(output, dim=tuple(range(1, output.dim())))
    # grads = torch.autograd.grad(outputs=output_sum, inputs=scaled_inputs)[0]
    integrated_grads = torch.autograd.grad(output_sum, scaled_inputs, create_graph=True, grad_outputs=torch.ones_like(output_sum))[0]

    # Accumulate the gradients
    # integrated_grads += grads.squeeze(0) / steps
    integrated_grads = integrated_grads / steps

    integrated_grads_local = integrated_grads.sum(axis = 0)

    # Scale the integrated gradients by the difference between input and baseline
    integrated_grads_global = (input_tensor - baseline) * integrated_grads_local
    
    return integrated_grads_global, integrated_grads_local

def iDeepB_integrated_gradients_fast_interpret_predict(model, input_tensor, topN , baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Generate scaled inputs along the interpolation path
    scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
    scaled_inputs = torch.stack(scaled_inputs).requires_grad_() 

    # Initialize integrated gradients as zero
    integrated_grads = torch.zeros_like(input_tensor)

    # Compute gradients
    # model.zero_grad()
    output = model(scaled_inputs)

    output_squeezed = model(input_tensor.unsqueeze(0)).squeeze(0)
    Top_n_value, Top_n_indices = output_squeezed.topk(topN, dim=0)
    Top_n_indice_vector = torch.zeros_like(output_squeezed)
    Top_n_indice_vector[Top_n_indices] = 1

    output = output * Top_n_indice_vector
    output_sum = torch.sum(output, dim=tuple(range(1, output.dim())))
    # grads = torch.autograd.grad(outputs=output_sum, inputs=scaled_inputs)[0]
    integrated_grads = torch.autograd.grad(output_sum, scaled_inputs, create_graph=True, grad_outputs=torch.ones_like(output_sum))[0]

    # Accumulate the gradients
    # integrated_grads += grads.squeeze(0) / steps
    integrated_grads = integrated_grads / steps

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
    # model.zero_grad()
    output = model(scaled_inputs)

    if target is None:
        target = torch.ones_like(output)

    output = output * target
    output_sum = torch.sum(output, dim=tuple(range(1, output.dim())))
    # grads = torch.autograd.grad(outputs=output_sum, inputs=scaled_inputs)[0]
    integrated_grads = torch.autograd.grad(output_sum, scaled_inputs, create_graph=True, grad_outputs=torch.ones_like(output_sum))[0]

    # Accumulate the gradients
    # integrated_grads += grads.squeeze(0) / steps
    integrated_grads = integrated_grads / steps

    integrated_grads_local = integrated_grads.sum(axis = 0)

    # Scale the integrated gradients by the difference between input and baseline
    integrated_grads_global = (input_tensor - baseline) * integrated_grads_local
    
    return integrated_grads_global, integrated_grads_local


'''
# seq_index = 1 # 2 有motif
# GradientXInput Saliency IntegratedGradients GradientBasedMethod
attribution_matrix = iDeepB_integrated_gradients_fast(model = model.to(args.device), input_tensor = subseqOH[seq_index].float().to(args.device))[0].cpu().detach().numpy()
print(attribution_matrix.shape)

bases = ['A', 'C', 'G', 'U']
plot_sequence_attribution(attribution_matrix, bases = bases, width=20, height=4)
'''

def attributionFromSeq(interpMethod, device, input, baseline, model, target=None, topN = None):
        
    if(interpMethod == "Top_IntegratedGradientsSum"):
            # 参考：/data/xliu/work/20231211_iDeepB/20240401_trainning_gene/iDeepB_interpretability/EPRB.v12.regression.transcript.predict.v9.4.2.Top_CNNLSTM_Deeplift copy.ipynb
            # IntegratedGradients, Deeplift and GradientShap are designed to attribute the change between the input and baseline to a predictive class or a value that the neural network outputs.

            ig = IntegratedGradients(model) #GradientShap DeepLift DeepLiftShap IntegratedGradients
            # 初始化contrib_scores_integration和hypothetical_contribs_integration
            contrib_scores_integration = np.zeros(input.shape, dtype = float) # np.full_like(subseqOH.shape, 0)

            output_squeezed = model(input).squeeze(0)
            Top_n_value, Top_n_indices = output_squeezed.topk(topN, dim=0)
            Top_n_indice_vector = torch.zeros_like(output_squeezed)
            Top_n_indice_vector[Top_n_indices] = 1
            Top_n_indice_vector = Top_n_indice_vector.cpu().detach().numpy()
    
            for target_index in np.where(np.array(Top_n_indice_vector) > 0)[0]:
                attributions, delta = ig.attribute(input, baseline, target=int(target_index), return_convergence_delta=True)
                contrib_scores_integration = contrib_scores_integration + attributions.cpu().detach().numpy()

            contrib_scores_integration = np.sum(contrib_scores_integration,axis=2)[:,:,None]
            return contrib_scores_integration*(input.cpu().detach().numpy())
    elif(interpMethod == "Top_IntegratedGradients"):
            # 参考：/data/xliu/work/20231211_iDeepB/20240401_trainning_gene/iDeepB_interpretability/EPRB.v12.regression.transcript.predict.v9.4.2.Top_CNNLSTM_Deeplift copy.ipynb
            # IntegratedGradients, Deeplift and GradientShap are designed to attribute the change between the input and baseline to a predictive class or a value that the neural network outputs.

            ig = captum.attr.IntegratedGradients(model) #GradientShap DeepLift DeepLiftShap IntegratedGradients
            # 初始化contrib_scores_integration和hypothetical_contribs_integration
            contrib_scores_integration = np.zeros(input.shape, dtype = float) # np.full_like(subseqOH.shape, 0)
            
            output_squeezed = model(input).squeeze(0)
            Top_n_value, Top_n_indices = output_squeezed.topk(topN, dim=0)
            Top_n_indice_vector = torch.zeros_like(output_squeezed)
            Top_n_indice_vector[Top_n_indices] = 1
            Top_n_indice_vector = Top_n_indice_vector.cpu().detach().numpy()

            for target_index in np.where(np.array(Top_n_indice_vector) > 0)[0]:
                attributions, delta = ig.attribute(input, baseline, target=int(target_index), return_convergence_delta=True)
                contrib_scores_integration = contrib_scores_integration + attributions.cpu().detach().numpy()

            # contrib_scores_integration = np.sum(contrib_scores_integration,axis=2)[:,:,None]
            return contrib_scores_integration*(input.cpu().detach().numpy())

    elif(interpMethod == "Top_Saliency"):
        ig = captum.attr.Saliency(model) #GradientShap DeepLift DeepLiftShap IntegratedGradients
        # 初始化contrib_scores_integration和hypothetical_contribs_integration
        contrib_scores_integration = np.zeros(input.shape, dtype = float) # np.full_like(subseqOH.shape, 0)

        output_squeezed = model(input).squeeze(0)
        Top_n_value, Top_n_indices = output_squeezed.topk(topN, dim=0)
        Top_n_indice_vector = torch.zeros_like(output_squeezed)
        Top_n_indice_vector[Top_n_indices] = 1
        Top_n_indice_vector = Top_n_indice_vector.cpu().detach().numpy()

        for target_index in np.where(np.array(Top_n_indice_vector) > 0)[0]:
            attributions= ig.attribute(input, target=int(target_index))
            contrib_scores_integration = contrib_scores_integration + attributions.cpu().detach().numpy()

        # contrib_scores_integration = np.sum(contrib_scores_integration,axis=2)[:,:,None]
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

        return attributions_global.cpu().detach().numpy()

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
        # subseqOH = torch.from_numpy(np.asarray(seqsInt) ).to(device)  ####
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
            # temp.append(treat_predict)
            temp.extend(treat_predict)
        # temp = np.array(temp).reshape(-1).tolist() 
        temp = np.concatenate(temp, axis=0)    
        print("temp shape: ", temp.shape,) 

    transcriptPd = temp # np.array(temp) #subseqPredicted.cpu().detach().numpy()
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
