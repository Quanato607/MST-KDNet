#!/usr/bin/env python3
# encoding: utf-8
import yaml
from data import make_data_loaders
from models import build_model
from models.discriminator import get_style_discriminator
from solver import make_optimizer_double
from losses import get_losses, bce_loss, Dice
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import *
import nibabel as nib
import numpy as np

def load_old_model(model_full, model_missing, saved_model_path):
    print("Constructing model from saved file... ")
    checkpoint = torch.load(saved_model_path)
    model_full.load_state_dict(checkpoint["model_full"])
    model_missing.load_state_dict(checkpoint["model_missing"])

    return model_full, model_missing
        
def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()
         

## Main section
config = yaml.load(open('./config.yml'), Loader=yaml.FullLoader)
init_env('7')
loaders = make_data_loaders(config)
model_full, model_missing = build_model(inp_dim1 = 4, inp_dim2 = 1)
model_full    = model_full.cuda()
model_missing = model_missing.cuda()
d_style = get_style_discriminator(num_classes = 128).cuda()
task_name = 'brats2024_full'
log_dir = os.path.join(config['path_to_log'], task_name)
criteria = Dice() 

def evaluate_performance(model_full, model_missing, loaders):
    class_score_full  = np.array((0.,0.,0.))
    class_score_mono  = np.array((0.,0.,0.))
    hd95_score_full  = np.array((0.,0.,0.))
    hd95_score_mono  = np.array((0.,0.,0.))
    loader = loaders['eval']
    total = len(loader)
    with torch.no_grad():
        for batch_id, (batch_x, batch_y) in enumerate(loader):
            batch_x, batch_y = batch_x.cuda(non_blocking=True), batch_y.cuda(non_blocking=True)
            seg_f, style_f, content_f = model_full(batch_x[:,:])
            seg_m, style_m, content_m = model_missing(batch_x[:,0:1])
            metric_full, metric_miss, hd95_full, hd95_miss  = evaluate_sample(seg_f, seg_m, batch_y)
            class_score_full += metric_full
            class_score_mono += metric_miss
            hd95_score_full += hd95_full
            hd95_score_mono += hd95_miss

    class_score_full  /= total   
    class_score_mono /= total
    hd95_score_full  /= total
    hd95_score_mono /= total
    print(f' validation Dise score full modalities class>> dice_wt: {class_score_full[0]} dice_tc:{class_score_full[1]}  dice_et:{class_score_full[2]}')
    print(f' validation Dise score missing modality  class>> dice_wt: {class_score_full[0]} dice_tc:{class_score_full[1]}  dice_et:{class_score_full[2]}')
    print(f' validation HD95 score full modalities class>> hd95_wt: {hd95_score_full[0]} hd95_tc:{hd95_score_full[1]}  hd95_et:{hd95_score_full[2]}')
    print(f' validation HD95 score missing modality  class>> hd95_wt: {hd95_score_full[0]} hd95_tc:{hd95_score_full[1]}  hd95_et:{hd95_score_full[2]}')

saved_model_path = log_dir+'/model_best.pth'
model_full, model_missing = load_old_model(model_full, model_missing, saved_model_path)
evaluate_performance(model_full, model_missing, loaders)