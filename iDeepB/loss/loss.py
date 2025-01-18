import torch
from torch.distributions.multinomial import Multinomial
from audtorch.metrics.functional import pearsonr
import torch.nn as nn
import math
from torcheval.metrics.functional import r2_score

class loss_fun(object):
    def __init__(self):
        self.eps = 1e-8
    #output: prediction ; 
    #target: label
    # multinomial negative log likelihood loss: 来自rbpnet
    # pytorch Multinomial的total_counts智能整数，不可以向量，k可能是错误的: bpnet losses.py: multinomial_nll

    def kl_divergence_loss(self, prediction, target):
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        log_prediction = torch.nn.functional.log_softmax(prediction, dim=1)
        log_target = torch.nn.functional.log_softmax(target, dim=1)
        loss = kl_loss(log_prediction, log_target)
        return loss
    
    def mse(self, prediction, target):
        loss = torch.nn.MSELoss()(prediction, target)
        return loss

    def SmoothL1Loss(self, prediction, target):
        loss = torch.nn.SmoothL1Loss()(prediction, target)
        return loss
    
    def multinomial_ng(self, prediction, target):
        total_counts = int(torch.sum(target).item())
        dist = Multinomial(total_count=total_counts, logits=prediction)
        loss = -torch.mean(dist.log_prob(target))
        return loss
        
    # bpnet losses
    def multinomialnll_batch(self, prediction, target):
        total_counts = int(torch.sum(target).item())
        dist = Multinomial(total_count=total_counts, logits =prediction)
        batchLen =  torch.tensor(target.shape[0], dtype=torch.float32)
        loss = -torch.sum(dist.log_prob(target))/batchLen
        return loss
    
    def mse_MN(self, prediction, target):  # prediction, target
        mse_loss = self.mse(prediction, target)
        poiss_loss = self.multinomialnll_batch(prediction, target)
        total_loss = (2 * mse_loss * poiss_loss) / (mse_loss + poiss_loss)
        return total_loss
    
    # bpnet losses
    # PoissonMultinomialNLL
    # 根据loss 数量级，Poisson + MultinomialNLL 感觉不合适
    

    def multinomialnll_seq(self, prediction, target):
        total_counts = int(torch.sum(target).item())
        dist = Multinomial(total_count=total_counts, logits =prediction)
        seqLen =  torch.tensor(target.shape[1], dtype=torch.float32)
        loss = -torch.sum(dist.log_prob(target))/seqLen
        return loss

    
    '''
    def pearson(self, prediction, target):
        loss = pearsonr(prediction, target).nanmean()
        return -loss
    
    def pearsonr_mse(self, target, prediction, alpha=1):
        pr_loss = self.pearson(target, prediction)
        mse_loss = self.mse(prediction, target)
        total_loss = pr_loss + alpha * mse_loss
        return total_loss
    '''
    
    @staticmethod
    def __log(t, eps=1e-20):
        return torch.log(t.clamp(min=eps)) # 小于某个数的数据设置为某个数等价于torch.log(prediction + self.eps))
    # https://pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html
    # Negative log likelihood loss with Poisson distribution of target.
    # target∼Poisson(input)loss(input,target)=input−target∗log(input)+log(target!)
    # if log: exp(input)−target∗input
    # if not log: input−target∗log(input+eps)

    # 等价于 nn.PoissonNLLLoss(）； if log_input is false : input−target∗log(input+eps)
    # def poisson_loss_test(self, prediction, target):
        # return (prediction - target * self.__log(prediction)).mean()
        # self.eps=1e-20
        # return (prediction - target * torch.log(prediction + self.eps)).mean() 算出来是nan；等价于 nn.PoissonNLLLoss(）
    def poissonLoss(self, prediction, target): # poisson_loss
        return (prediction - target * self.__log(prediction)).mean()
        # self.eps=1e-20
        # return (prediction - target * torch.log(prediction + self.eps)).mean() 算出来是nan；等价于 nn.PoissonNLLLoss(）
    
    # Poisson negative log-likelihood loss 
    def PoissonNLLLoss(self, prediction, target):
        loss = nn.PoissonNLLLoss(log_input=False, eps=1e-8)(prediction, target)
        return loss
    
    # code from gopher: https://github.com/shtoneyan/gopher/blob/main/gopher/losses.py
    '''
    def pearsonr_poisson(self, target, prediction): # 
        pr_loss = self.pearson(prediction, target)
        poiss_loss = self.poisson_loss_test(prediction, target)
        total_loss = (2 * pr_loss * poiss_loss) / (pr_loss + poiss_loss)
        return total_loss 
    '''

    def mse_poisson(self, prediction, target):  # prediction, target
            mse_loss = self.mse(prediction, target)
            poiss_loss = self.poisson_loss_test(prediction, target)
            total_loss = (2 * mse_loss * poiss_loss) / (mse_loss + poiss_loss)
            return total_loss
    
    def mse_poisson_alpha(self, prediction, target, alpha=1):  # prediction, target
            mse_loss = self.mse(prediction, target)
            poiss_loss = self.poisson_loss_test(prediction, target)
            total_loss = alpha * poiss_loss + mse_loss
            return total_loss
    
    def mse_poisson2(self, prediction, target):  # prediction, target
            mse_loss = self.mse(prediction, target)
            poiss_loss = self.PoissonNLLLoss(prediction, target)
            total_loss = (2 * mse_loss * poiss_loss) / (mse_loss + poiss_loss)
            return total_loss
    
    def mse_poisson_alpha2(self, prediction, target, alpha=1):  # prediction, target
            mse_loss = self.mse(prediction, target)
            poiss_loss = self.PoissonNLLLoss(prediction, target)
            total_loss = alpha * poiss_loss + mse_loss
            return total_loss
    
    
    '''
    def multinomialnll_mse(self, y_pred, y_true, alpha=1):
            mult_loss = self.multinomialnll_batch(y_pred, y_true)
            mse_loss = nn.MSELoss()(y_pred, y_true)
            total_loss = mult_loss + alpha * mse_loss
            return total_loss
    '''
    
    def multinomialnll_mse(self, y_pred, y_true, alpha=1):
        mult_loss = self.multinomialnll_batch(y_pred, y_true)
        mse_loss = nn.MSELoss()(y_pred, y_true)
        total_loss = alpha * mult_loss + mse_loss
        return total_loss
    
    def l1_loss(self, y_pred, y_true):
        return nn.L1Loss()(y_pred, y_true)

    '''
    def pearsonr_multinomialnll(self, target, prediction): 
        pr_loss = self.pearson(target, prediction)
        mult_loss = self.multinomialnll_seq(prediction, target)
        total_loss = (2 * pr_loss * mult_loss) / (pr_loss + mult_loss)
        return total_loss 
    '''
    '''
    def mse_multinomialnll(self, target, prediction, alpha=1):  # prediction, target
            mult_loss = self.multinomialnll_batch(prediction, target)
            mse_loss = self.mse(prediction, target)
            total_loss = (2 * mse_loss * mult_loss) / (mse_loss + mult_loss) # mult_loss + alpha * mse_loss
            return total_loss
    '''

    '''
    def poissonLammda(self, prediction, target):
        Mse_loss = torch.nn.MSELoss()
        loss = Mse_loss(torch.mean(prediction, axis=1), torch.mean(target, axis=1))
        return loss
    '''

    

    # loss的数量级
    # QKI
    # multinomialnll_batch: 25.311
    # multinomialnll_seq: 64.687
    # poisson: 0.30
    # mse: 3.451

    # SRSF1
    # multinomialnll_batch: 32.582
    # multinomialnll_seq: 82.132
    # poisson: 0.377
    # mse: 0.790

    
    
