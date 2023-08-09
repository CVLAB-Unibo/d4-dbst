# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os,sys,inspect

from numpy.core.fromnumeric import take
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
import time
import torch
import torch.optim as optim
from model.net import get_network, get_transfer, get_adaptive_network
from model.deeplab import Res_Deeplab
from model.bdl import Deeplab
from model.mrnet import DeeplabMulti as mrnet
from model.max_squares import DeeplabMulti as maxsq
# from model.classifier import ASPP_Classifier_V2
# from model.feature_extractor import resnet_feature_extractor
import torch.nn.functional as F
from utils.metrics import ScoreUpdater, Accuracy
import utils.utils as utils
from dataloader.cityscapes_test import CS_test
from torch.utils import data
from tqdm import tqdm
import cv2
import torch.nn.functional as F
from collections import namedtuple
from PIL import Image as pil
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
from collections import OrderedDict

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


# %%
device = torch.device('cuda')
num_classes = 19

data_dir = "/media/data_4t/aCardace/datasets"
ckpt_filename = "checkpoint.tar"
best_ckpt_filename = "model_best.tar"
model_dir_source = "gta2cs/net1_original_high_res"
model_dir_target = "gta2cs/net2_r50_wc_strong/ckpt/gta_src.pth"

model_dir_transfer = "gta2cs/transfer_net1_original_high_res2net2_r50_wc_strong_long"

# CAMBIA
model_dir_baseline = "gta2cs/DA/ltir/ResNet_GTA_50.2.pth"

json_path = os.path.join(model_dir_transfer, 'params.json')
params = utils.Params(json_path)
params.device = device


# %%
# val_dl = dataloader.fetch_dataloader(data_dir, txt_val, 'val', params)
mean=np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
std=np.array((1, 1, 1), dtype=np.float32)

file_path = "./splits/cityscapes/train_random.txt"
label_size = (1024, 512)

# file_path = "./splits/cityscapes/val.txt"
# label_size = (2048, 1024)

# CAMBIA (interpolation, per FADA cambia size e togli mean, std, rgb)
dataset = CS_test(root=data_dir, txt_file=file_path, use_depth=True, threshold=50, size=(1024, 512), label_size=label_size, mean=mean, std=std, rgb=False, interpolation="bicubic")
val_dl = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

dataset1 = CS_test(root=data_dir, txt_file=file_path, use_depth=True, threshold=50, size=(1024, 512), label_size=label_size, interpolation="lanczos")
val_dl1 = data.DataLoader(dataset1, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

gta_file_path = "splits/gta1_full/train.txt"
dataset_gta = CS_test(root=data_dir, txt_file=gta_file_path, use_depth=True, threshold=50, size=(1024, 512), label_size=label_size, mean=mean, std=std, rgb=False, interpolation="lanczos")
val_dl_gta = data.DataLoader(dataset_gta, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

model_source = Res_Deeplab(num_classes=1, use_sigmoid=True).to(device)
model_target = Res_Deeplab(num_classes=num_classes).to(device)

# CAMBIA
model_baseline = Deeplab(num_classes=num_classes).to(device)
# model_baseline = mrnet(num_classes=num_classes, use_se=True, train_bn=False, norm_style="gn", droprate=0.2).to(device)
# model_baseline = maxsq(num_classes=num_classes).to(device)
# model_baseline = Res_Deeplab(num_classes=num_classes, layers=23).to(device)

transfer = get_transfer().to(device)


# %%
ckpt_source_file_path = os.path.join(model_dir_source, 'ckpt', best_ckpt_filename)
saved_state_dict = torch.load(ckpt_source_file_path, map_location=device)["state_dict"]
model_source.load_state_dict(saved_state_dict)

# ckpt_target_file_path = os.path.join(model_dir_target, 'ckpt', best_ckpt_filename)
saved_state_dict = torch.load(os.path.join(model_dir_target), map_location=device)
model_target.load_state_dict(saved_state_dict)

ckpt_transfer_file_path = os.path.join(model_dir_transfer, 'ckpt', best_ckpt_filename)
saved_state_dict = torch.load(ckpt_transfer_file_path, map_location=device)["state_dict"]
transfer.load_state_dict(saved_state_dict)


#adaptive model 
source_encoder = torch.nn.Sequential(*(list(model_source.children())[:-2]))
target_encoder = torch.nn.Sequential(*(list(model_target.children())[:-1]))
target_decoder = list(model_target.children())[-1]
adaptive_model = get_adaptive_network(source_encoder, transfer, target_decoder)


# %%

# CAMBIA

# #maxsquare -> maxsq, bicubic,  _
# saved_state_dict = torch.load(model_dir_baseline2, map_location=device)["state_dict"]
# new_params = {'.'.join(k.split('.')[1:]) : v for k, v in saved_state_dict.items()}
# model_baseline2.load_state_dict(new_params)


# %%
# staff and things -> maxsq, bicubic, bicubic
# saved_state_dict = torch.load(model_dir_baseline, map_location=device)
# model_baseline.load_state_dict(saved_state_dict)


# %%
# ltir -> Deeplab, _ ,bicubic
saved_state_dict = torch.load(model_dir_baseline, map_location=device)
model_baseline.load_state_dict(saved_state_dict)


# %%
#bdl -> Deeplab, lanczos, lanczos
# saved_state_dict = torch.load(model_dir_baseline, map_location=device)
# model_baseline.load_state_dict(saved_state_dict)


# %%
#adaptsegnet -> maxsq, bicubic, _
# saved_state_dict = torch.load(model_dir_baseline, map_location=device)
# model_baseline.load_state_dict(saved_state_dict)


# %%
# # #mrnet -> mrnet, lanczos  ,lanczos
# saved_state_dict = torch.load(model_dir_baseline, map_location=device)
# new_params = {'.'.join(k.split('.')[1:]) : v for k, v in saved_state_dict.items()}
# model_baseline.load_state_dict(new_params)


# %%
# # fada, bicubic, lanczos
# def strip_prefix_if_present(state_dict, prefix):
#     keys = sorted(state_dict.keys())
#     if not all(key.startswith(prefix) for key in keys):
#         return state_dict
#     stripped_state_dict = OrderedDict()
#     for key, value in state_dict.items():
#         stripped_state_dict[key.replace(prefix, "")] = value
#     return stripped_state_dict

# def build_feature_extractor():
#     backbone = resnet_feature_extractor("resnet101", pretrained_weights="https://download.pytorch.org/models/resnet101-5d3b4d8f.pth", aux=False, pretrained_backbone=True, freeze_bn=False)
#     return backbone

# def build_classifier():
#     classifier = ASPP_Classifier_V2(2048, [6, 12, 18, 24], [6, 12, 18, 24], 19)
#     return classifier

# feature_extractor = build_feature_extractor()
# feature_extractor.to(device)

# classifier = build_classifier()
# classifier.to(device)
# checkpoint = torch.load(model_dir_baseline, map_location=torch.device('cuda'))
# feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
# feature_extractor.load_state_dict(feature_extractor_weights)
# classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
# classifier.load_state_dict(classifier_weights)

# model_baseline = torch.nn.Sequential(feature_extractor, classifier)


# %%
def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N

def update_mask_with_depth(mask, depth):

    depth_clone = depth.clone()
    eps = 1e-4
    b, h, w = mask.size()
    p_t = np.percentile(depth[-1].numpy(), 100)
    depth_clone[-1][depth_clone[-1]>p_t] = p_t+eps
    depth_clone[-1] = (depth_clone[-1] - depth_clone[-1].min())/(depth_clone[-1].max() - depth_clone[-1].min()) 

    depth_masks = []
    for i in range(depth.shape[0]-1):
        p = np.percentile(depth[i].numpy(), 100)
        depth_clone[i][depth_clone[i]>p] = p+eps   
        min_depth = depth_clone[i].min()
        max_depth = depth_clone[i].max()
        depth_clone[i] = (depth_clone[i] - depth_clone[i].min())/(depth_clone[i].max() - depth_clone[i].min()) 

        t = (p - min_depth) / (max_depth - min_depth)
        close_pixels = torch.le(depth_clone[i], t)
        depth_masks.append(close_pixels)
    depth_masks = torch.stack(depth_masks, dim=0)
    source_masks = mask[:-1]
    updated_source_masks = source_masks * depth_masks
    return torch.cat([updated_source_masks, mask[-1].unsqueeze(0)], dim=0), depth_clone

def cross_check_mask(mask, normalized_depth):
    sorted_indexes = torch.argsort(normalized_depth, dim=0, descending=False)
    
    original_mask = mask.clone()
    # _, h, w = original_mask.size() 
    # original_mask = torch.cat([original_mask, torch.full((1,h,w), 1, dtype=torch.int)], dim=0)
    masks = []

    for k in range(original_mask.shape[0]):
        masks.append([])

    for i in range(sorted_indexes.shape[0]):
        for j in range(original_mask.shape[0]):
            closest = sorted_indexes[i] == j

            mask_j = original_mask[j] & closest
            masks[j].append(mask_j)
            _, mask_j = torch.broadcast_tensors(original_mask, mask_j)
            mask_j = ~mask_j
            mask_j[j] = True
            mask_j[-1] = True
            original_mask = original_mask & mask_j

    final_masks = []
    for k in range(mask.shape[0]):
        mask_k = torch.stack(masks[k], dim=0).numpy()
        mask_k = np.any(mask_k, axis=0)
        final_masks.append(torch.tensor(mask_k))

    final_masks = torch.stack(final_masks, dim=0)
    return final_masks.long()

def oneMix(mask, source, target):
    stackedMask0, _ = torch.broadcast_tensors(mask, source)
    return stackedMask0*source+(1-stackedMask0)*target


# %%

def generate(adaptive_model, model_baseline, val_dl, val_dl1, params):
    # set model to evaluation mode
        adaptive_model.eval()
        model_baseline.eval()
        # plt.figure(figsize=(30,20))

        valid_labels = range(19)
        x_num = 500

        scorer = ScoreUpdater(valid_labels, params.num_classes, x_num, None)
        acc = Accuracy(params.num_classes+1, ignore_index=19)
        scorer.reset()
        acc.reset()     

        classes = np.arange(num_classes)
        inverted_w = np.load("weights/inverted_weights_gta1.npy")
        baseline_weights = inverted_w/inverted_w.max()
        model_weights = 1-(inverted_w/inverted_w.max())
        iter_transfer = iter(val_dl1)
        iter_gta = iter(val_dl_gta)
        # print(model_weights)

        with torch.no_grad():
            for i, (xb, yb, depth, name) in enumerate(tqdm(val_dl, position=0, leave=True)):
                stack_masks = []
                depth[depth > 60] = 60
                xb = xb.to(device)
                yb = yb.to(device)

                # print("xb", xb.shape)
                # print("yb", yb.shape)

                xb_transfer, _, _ , _ = next(iter_transfer)
                xb_transfer = xb_transfer.to(device)

                # print("xb transfer", xb_transfer.shape)

                try:
                    xb_gta, yb_gta, depth_gta , _ = next(iter_gta)
                except StopIteration:
                    iter_gta = iter(val_dl_gta)
                    xb_gta, yb_gta, depth_gta , _ = next(iter_gta)
                xb_gta = xb_gta.to(device)
                depth_gta[depth_gta > 60] = 60

                # print("xb_gta", xb_gta.shape)
                # print("yb_gta", yb_gta.shape)
                # print("depth_gta", depth_gta.shape)

                outs = []
                for h in range(0, xb.shape[0], 2):
                    
                    start_i = h
                    end_i = start_i+2

                    out_size = (params.load_size[1], params.load_size[0])
                    # out_size = (params.label_size[1], params.label_size[0])

                    z = adaptive_model(xb_transfer[start_i:end_i])
                    z = F.interpolate(z, size=out_size, mode='bilinear', align_corners=True)
                    z = F.softmax(z/6, dim=1)
                    
                    prediction_z = torch.argmax(z, dim=1)
                    mask_z = torch.zeros(prediction_z.size()).to(params.device)
                    for label in classes:
                        mask_z[prediction_z.eq(label)] = model_weights[label]

                    z *= mask_z.unsqueeze(dim=1)
        
                    model_target_out = model_baseline(xb[start_i:end_i])
                    model_target_out = F.interpolate(model_target_out, size=out_size, mode='bilinear', align_corners=True)
                    model_target_out = F.softmax(model_target_out/6, dim=1)

                    prediction_baseline = torch.argmax(model_target_out, dim=1)
                    mask_baseline = torch.zeros(prediction_baseline.size()).to(params.device)
                    for label in classes:
                        mask_baseline[prediction_baseline.eq(label)] = baseline_weights[label]
                    model_target_out *= mask_baseline.unsqueeze(dim=1)

                    out = z + model_target_out
                    out = F.softmax(out, dim=1)
                    outs.append(out)

                outs = torch.cat(outs, dim=0)
                # print("outs", outs.size())

                # for b in range(outs.shape[0]):
                #     output = outs[b].cpu().numpy()
                #     output = output.transpose(1, 2, 0)
                #     label, prob = np.argmax(output, axis=2), np.max(output, axis=2)

                #     scorer.update(label.flatten(), yb[b].cpu().numpy().flatten(), i)
                #     acc.add(outs[b].unsqueeze(0), yb[b].unsqueeze(0))

                out = outs.cpu()
                xb = xb.cpu()
                yb = yb.cpu()
                depth = depth.cpu()
                prediction = torch.argmax(out, dim=1).cpu()

                xb = torch.cat([xb_gta.cpu(), xb], dim=0)
                yb = torch.cat([yb_gta.cpu(), yb], dim=0)
                depth = torch.cat([depth_gta.cpu(), depth], dim=0)
                prediction = torch.cat([yb_gta.cpu(), prediction], dim=0)

                # print("xb post", xb.shape)
                # print("yb post", yb.shape)
                # print("depth post", depth.shape)
                # print("prediction", prediction.shape)

                take_classes = torch.unique(yb_gta).numpy()
                take_classes = np.delete(take_classes, np.where(take_classes == 19))
                n = len(take_classes)
                n = int((n + (n % 2)) / 2)
                rand_classes = np.random.choice(take_classes, size=n, replace=False)
                rand_classes = torch.tensor(rand_classes)
                for k in range(prediction.shape[0]):
                    stack_masks.append(generate_class_mask(prediction[k].cpu(), rand_classes))
                mask = torch.stack(stack_masks, dim=0)

                # mask, normalized_depth = update_mask_with_depth(mask, depth)
                # mask = cross_check_mask(mask, normalized_depth)
                
                a_xb = xb[-1].cpu()
                a_prediction = prediction[-1].cpu()
                a_yb = yb[-1].cpu()
                a_depth = depth[-1].cpu()

                for j in range(prediction.shape[0]-1):        
                    a_xb = oneMix(mask[j], xb[j].cpu(), a_xb)
                    a_prediction = oneMix(mask[j], prediction[j].cpu(), a_prediction)
                    a_yb = oneMix(mask[j], yb[j].cpu(), a_yb)
                    a_depth = oneMix(mask[j], depth[j].cpu(), a_depth)

                # #### VISUALIZATION
                # xb = torch.cat([xb, a_xb.unsqueeze(0)], dim=0)
                # yb = torch.cat([yb, a_yb.unsqueeze(0)], dim=0)
                # depth = torch.cat([depth, a_depth.unsqueeze(0)], dim=0)
                # prediction = torch.cat([prediction, a_prediction.unsqueeze(0)], dim=0)

                # figure = val_dl.dataset.get_predictions_plot_sem(xb, prediction, yb, depth)
                # image = pil.fromarray(figure)
                # image.save(f"./test/{i}.png")
                # if i>10:
                #     break

                #### VISUALIZATION

                output = np.asarray(a_prediction.cpu().numpy(), dtype=np.uint8)
                output = pil.fromarray(output)
                # name = name[0].replace('gtFine', 'dbst/gtFine_bdl', 1)
                image_name = "/".join(name[0].split('/')[-3:])
                name = "/media/data_4t/aCardace/datasets/CityScapes/dacs/gtFine_ltir"
                # name = name.replace(image_name, f'{i}.png')
                name = f"{name}/{i}.png"
                os.makedirs(os.path.dirname(name), exist_ok=True)
                output.save(name)

                img = np.asarray(a_xb.cpu())
                img = val_dl.dataset.re_normalize(img)
                img = pil.fromarray(img)
                name = name.replace('gtFine_ltir', "train_ltir")
                os.makedirs(os.path.dirname(name), exist_ok=True)
                img.save(name)
                
                if i == 7437:
                    break

        # scorer.print_score()
        # acc = acc.value()
        # print('acc:', acc)


# %%
generate(adaptive_model, model_baseline, val_dl, val_dl1, params)


# %%


