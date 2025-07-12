import torch.nn as nn
import torch.nn.functional as F
from torch import *
from einops import rearrange
import torch

import random
import numpy as np
import os
def seed_everything(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
seed_everything(1)  

class iDeepB(nn.Module):
    def __init__(self,dropOut, control):
        super(iDeepB, self).__init__()

        self.dropout = nn.Dropout( dropOut)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid() 
        self.control = control 
        
        self.Res1_conv1 = nn.Conv1d(in_channels=640,out_channels=128,kernel_size=5, padding='same')
        self.Res1_bn1 = nn.BatchNorm1d(128)
        self.Res1_dropout1 = nn.Dropout(dropOut)
        self.Res1_conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding='same')
        self.Res1_bn2 = nn.BatchNorm1d(64)
        self.Res1_conv3 = nn.Conv1d(640, 64, 1)
        self.Res1_dropout2 = nn.Dropout(dropOut)
    
        # Attention
        dim = 64*2
        hidden_dim = 128
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)
        self.softmax = nn.Softmax(dim=2) #归一化

        # LSTM
        self.lstm = nn.LSTM(64, 64, 2, bidirectional=True)
        
        # 
        self.fc1_T = nn.Linear(64 * 2, 32)
        self.fc2_T =nn.Linear(32, 8)
        self.fc3_T = nn.Linear(8, 1)

        self.output_head_T = nn.Sequential(
            self.fc1_T,
            self.leaky_relu,
            self.dropout,
            self.fc2_T,
            self.leaky_relu,
            self.dropout,
            self.fc3_T,
            self.relu
            )
              
        self.fc1_C = nn.Linear(64 * 2, 64)
        self.fc2_C = nn.Linear(64, 16)
        self.fc3_C = nn.Linear(16, 1)
        self.output_head_C = nn.Sequential(
            self.fc1_C,
            self.leaky_relu,
            self.dropout,
            self.fc2_C,
            self.leaky_relu,
            self.dropout,
            self.fc3_C,
            self.relu
            )
        
        self.fc1_w_n = nn.Linear(101, 50)
        self.fc2_w_n = nn.Linear(50, 16)
        self.fc3_w_n = nn.Linear(16, 1)
        self.output_head_w_n = nn.Sequential(
            self.fc1_w_n,
            self.leaky_relu,
            self.dropout,
            self.fc2_w_n,
            self.leaky_relu,
            self.dropout,
            self.fc3_w_n,
            self.relu
            )
        
        self.fc1_w_d = nn.Linear(64 * 2, 64)
        self.fc2_w_d = nn.Linear(64, 16)
        self.fc3_w_d = nn.Linear(16, 1)
        self.output_head_w_d = nn.Sequential(
            self.fc1_w_d,
            self.leaky_relu,
            self.dropout,
            self.fc2_w_d,
            self.leaky_relu,
            self.dropout,
            self.fc3_w_d,
            self.sigmoid
            )

        self.layer_norm_MH = nn.LayerNorm(128)
        self.layer_norm_FC = nn.LayerNorm(128)

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n') 

        # Resnet
        x_conv = self.Res1_dropout1(F.leaky_relu(self.Res1_bn1(self.Res1_conv1(x)))) 
        x_conv = F.leaky_relu(self.Res1_bn2(self.Res1_conv2(x_conv))) 
        x_conv = self.Res1_dropout2(x_conv + self.Res1_conv3(x)) #([64, 64, 101])
        
        # BiLSTM
        x1a = rearrange(x_conv, 'b d n -> n b d')
        hidden_state = torch.randn(2*2, x1a.shape[1] , 64, device=x.device)  
        cell_state = torch.randn(2*2, x1a.shape[1], 64,device=x.device) 
        x_LSTM, (_, _) = self.lstm(x1a, (hidden_state, cell_state))
        # x_LSTM = self.layer_norm_MH(x_LSTM)
        x_LSTM = rearrange(x_LSTM, 'n b d -> b d n')

        # Attention
        heads = 4
        qkv = self.to_qkv(x_LSTM).chunk(3, dim = 1) 
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x -> b h c x', h = heads), qkv)
        q = F.normalize(q, dim = -1)
        k = F.normalize(k, dim = -1)

        qk_t = einsum('b h d i, b h d j -> b h i j', q, k)
        qk_t = qk_t.softmax(dim = -1)
        x_att = einsum('b h i j, b h d j -> b h i d', qk_t, v)
        x_att = rearrange(x_att, 'b h x d -> x b (h d)')
        
        # x_att = self.layer_norm_FC(x_att)

        # FC
        if not self.control:
            output_T =self.output_head_T(x_att)
            output_T = rearrange(output_T, 'n b d -> b n d')
            output_T = torch.squeeze(output_T, 2)

            return output_T
            
        elif self.control:  
            output_T =self.output_head_T(x_att)
            output_T = rearrange(output_T, 'n b d -> b n d')
            output_T = torch.squeeze(output_T, 2)

            output_C =self.output_head_C(x_att)
            output_C = rearrange(output_C, 'n b d -> b n d')
            output_C = torch.squeeze(output_C, 2)
            
            x_w = rearrange(x_att, 'n b d -> b d n')
            x_w = self.output_head_w_n(x_w)
            x_w = x_w.squeeze(axis = 2)
            x_w = self.output_head_w_d(x_w)
            x_w = x_w.squeeze(axis = 1)
            
            return output_T, output_C, x_w