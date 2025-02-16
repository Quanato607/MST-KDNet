#!/usr/bin/env python3
# encoding: utf-8
# Modified from https://github.com/Wangyixinxin/ACN
import torch
from torch.nn import functional as F
import numpy as np
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image 
import cv2
import torch.nn as nn
from models.evd import EVD
from models.slkd import DistillKL_logit_stand

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, consistency = 10, consistency_rampup = 20.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)
  
def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


def dice_loss(input, target):
    """soft dice loss"""
    eps = 1e-7
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

def gram_matrix(input):
    a, b, c, d, e = input.size()
    features = input.view(a * b, c * d * e)
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d * e)

def mix_matrix(input1, input2):
    a, b, c, d, e = input1.size()
    input1 = input1.view(a * b, c * d * e)
    input2 = input2.view(a * b, c * d * e)
    G = torch.mm(input1, input2.t())  # compute the gram product
    return G.div(a * b * c * d * e)

def get_style_loss(sf, sm):
    g_f = gram_matrix(sf)
    g_m = gram_matrix(sm)
    channels = sf.size(1)
    size     = sf.size(2)*sf.size(3) 
    sloss = torch.sum(torch.square(g_f-g_m)) / (4.0 * (channels ** 2) * (size ** 2))
    return sloss*0.0001

def get_GS_loss(Gs_f, Gs_m):
    Ms_f = []
    Ms_m = []
    for i in range(len(Gs_f)):
        Ms_f.append(mix_matrix(Gs_f[i], Gs_f[i-1]))
        Ms_m.append(mix_matrix(Gs_m[i], Gs_m[i-1]))
    channels = Gs_f[0].size(1)
    size     = Gs_f[0].size(2)*Gs_f[0].size(3)

    sloss = 0.0
    for M_f, M_m in zip(Ms_f, Ms_m):
        sloss += torch.sum(torch.square(M_f-M_m)) / (4.0 * (channels ** 2) * (size ** 2))
    
    return sloss*0.0001


def unet_Co_loss(config, batch_pred_full, content_full, batch_y, batch_pred_missing, content_missing, sf, sm, epoch):
    loss_dict = {}
    loss_dict['wt_dc_loss']  = dice_loss(batch_pred_full[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['tc_dc_loss'] = dice_loss(batch_pred_full[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_dc_loss']  = dice_loss(batch_pred_full[:, 2], batch_y[:, 2])  # enhance tumor
    
    loss_dict['wt_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['tc_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 2], batch_y[:, 2])  # enhance tumor

    ## Dice loss predictions
    loss_dict['loss_dc'] = loss_dict['wt_dc_loss'] + loss_dict['tc_dc_loss'] + loss_dict['et_dc_loss']
    loss_dict['loss_miss_dc'] = loss_dict['wt_miss_dc_loss'] + loss_dict['tc_miss_dc_loss'] + loss_dict['et_miss_dc_loss']
    
    ## Consistency loss
    loss_dict['wt_mse_loss']  = F.mse_loss(batch_pred_full[:, 0], batch_pred_missing[:, 0], reduction='mean') 
    loss_dict['tc_mse_loss'] = F.mse_loss(batch_pred_full[:, 1], batch_pred_missing[:, 1], reduction='mean') 
    loss_dict['et_mse_loss']  = F.mse_loss(batch_pred_full[:, 2], batch_pred_missing[:, 2], reduction='mean') 
    loss_dict['consistency_loss'] = loss_dict['wt_mse_loss'] + loss_dict['tc_mse_loss'] + loss_dict['et_mse_loss']
    
    ## Content loss
    loss_dict['content_loss'] = F.mse_loss(content_full, content_missing, reduction='mean')
    
    ## Style loss
    sloss = get_style_loss(sf, sm)
    
    
    ## Weights for each loss the lamba values
    weight_content = float(config['weight_content'])
    weight_missing = float(config['weight_mispath'])
    weight_full    = 1 - float(config['weight_mispath'])
    
    weight_consistency = get_current_consistency_weight(epoch)
    loss_dict['loss_Co'] = weight_full * loss_dict['loss_dc'] + weight_missing * loss_dict['loss_miss_dc'] + \
                            weight_consistency * loss_dict['consistency_loss'] + weight_content * loss_dict['content_loss']+sloss
    
    return loss_dict

def unet_Co_loss_divide(config, batch_pred_full, content_full, batch_y, batch_pred_missing, content_missing, sf, sm, epoch):
    loss_dict = {}
    loss_dict['net_dc_loss']  = dice_loss(batch_pred_full[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['snfh_dc_loss'] = dice_loss(batch_pred_full[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_dc_loss']  = dice_loss(batch_pred_full[:, 2], batch_y[:, 2])  # enhance tumor
    loss_dict['bg_dc_loss']  = dice_loss(batch_pred_full[:, 3], batch_y[:, 3])  # enhance tumor
    
    loss_dict['net_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['snfh_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 2], batch_y[:, 2])  # enhance tumor
    loss_dict['bg_miss_dc_loss']  = dice_loss(batch_pred_full[:, 3], batch_y[:, 3])  # enhance tumor

    ## Dice loss predictions
    loss_dict['loss_dc'] = loss_dict['net_dc_loss'] + loss_dict['snfh_dc_loss'] + loss_dict['et_dc_loss'] + loss_dict['bg_dc_loss']
    loss_dict['loss_miss_dc'] = loss_dict['net_miss_dc_loss'] + loss_dict['snfh_miss_dc_loss'] + loss_dict['et_miss_dc_loss'] + loss_dict['bg_miss_dc_loss']
    
    ## Consistency loss
    loss_dict['net_mse_loss']  = F.mse_loss(batch_pred_full[:, 0], batch_pred_missing[:, 0], reduction='mean') 
    loss_dict['snfh_mse_loss'] = F.mse_loss(batch_pred_full[:, 1], batch_pred_missing[:, 1], reduction='mean') 
    loss_dict['et_mse_loss']  = F.mse_loss(batch_pred_full[:, 2], batch_pred_missing[:, 2], reduction='mean') 
    loss_dict['bg_mse_loss']  = dice_loss(batch_pred_full[:, 3], batch_y[:, 3])  # enhance tumor
    loss_dict['consistency_loss'] = loss_dict['net_mse_loss'] + loss_dict['snfh_mse_loss'] + loss_dict['et_mse_loss'] + loss_dict['bg_mse_loss']
    
    ## Content loss
    loss_dict['content_loss'] = F.mse_loss(content_full, content_missing, reduction='mean')
    
    ## Style loss
    # sloss = get_style_loss(sf, sm)
    
    
    ## Weights for each loss the lamba values
    weight_content = float(config['weight_content'])
    weight_missing = float(config['weight_mispath'])
    weight_full    = 1 - float(config['weight_mispath'])
    
    weight_consistency = get_current_consistency_weight(epoch)
    loss_dict['loss_Co'] = weight_full * loss_dict['loss_dc'] + weight_missing * loss_dict['loss_miss_dc'] + \
                            weight_consistency * loss_dict['consistency_loss'] + weight_content * loss_dict['content_loss']
    
    return loss_dict

def get_losses(config):
    losses = {}
    losses['co_loss'] = unet_Co_loss
    losses['unetr_loss'] = Unetr_Loss()
    losses['evd_loss'] = EVD
    losses['slkd_loss'] = DistillKL_logit_stand()
    losses['gsm_loss'] = get_GS_loss
    return losses

def get_losses_divide(config):
    losses = {}
    losses['co_loss'] = unet_Co_loss_divide
    losses['unetr_loss'] = Unetr_Loss()
    losses['evd_loss'] = EVD
    losses['slkd_loss'] = DistillKL_logit_stand()
    losses['gsm_loss'] = get_GS_loss
    return losses


class Dice(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(Dice, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.Tensor(prediction)
        target = torch.Tensor(target)
        iflat = prediction.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()

        return ((2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)).numpy()

class Unetr_Loss(nn.Module):
    def __init__(self, weight_factor=[1.0,1.0,1.0,1.0]):
        super(Unetr_Loss, self).__init__()
        self.weight_factor = weight_factor  # 可以调整每个特征层的权重

    def forward(self, teacher_fs, student_fs):
        """
        计算教师和学生网络特征列表之间的L2损失

        参数:
        - teacher_fs: 教师网络的特征列表 (list of tensors)
        - student_fs: 学生网络的特征列表 (list of tensors)

        返回:
        - L2损失的总和
        """
        total_loss = 0.0
        for i, (t, s) in enumerate(zip(teacher_fs, student_fs)):
            # 确保teacher_fs和student_fs的形状相同
            assert t.shape == s.shape, f"Feature map shapes must match: {t.shape} vs {s.shape}"

            # 计算L2损失，使用F.mse_loss作为L2范数的实现
            loss = F.mse_loss(s, t, reduction='mean')
            total_loss += loss * self.weight_factor[i]

        return total_loss

if __name__ == '__main__':
    # 输入logit的大小为 (1, 4, 160, 192, 128)
    input_shape = (1, 4, 160, 192, 128)
    
    y_1 = torch.randn(input_shape).cuda() 
    y_2 = torch.randn(input_shape).cuda()  
    y_3 = torch.randn(input_shape).cuda() 

    

    s_1 = torch.randn(input_shape).cuda() 
    s_2 = torch.randn(input_shape).cuda()  
    s_3 = torch.randn(input_shape).cuda() 
    
    
