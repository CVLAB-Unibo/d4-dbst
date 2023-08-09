# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
os.chdir('..')
import numpy as np
import time
import torch
import torch.optim as optim
from model.net import get_network, get_transfer, get_adaptive_network
from model.deeplab import Res_Deeplab
from model.bdl import Deeplab
from model.mrnet import DeeplabMulti as mrnet
from model.max_squares import DeeplabMulti as maxsq
from model.classifier import ASPP_Classifier_V2
from model.feature_extractor import resnet_feature_extractor
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
device = torch.device('cuda:0')
num_classes = 19
root = "/media/data_4t/aCardace"

data_dir = os.path.join(root, "datasets/")
txt_val = os.path.join(root, "atdt/splits/cityscapes/val.txt")
ckpt_filename = "checkpoint.tar"
best_ckpt_filename = "model_best.tar"
model_dir_source = os.path.join(root, "atdt/synthia2cs/net1_full_synthia_cityscapes_alternated")
model_dir_target = os.path.join(root, "atdt/synthia2cs/net2_r50_wc7_augmentation")

model_dir_transfer = os.path.join(root, "atdt/synthia2cs/net1_full_net2_full_aug_wc7_transfer_manual_subset")
# model_dir_baseline = os.path.join(root, "atdt/gta2cs/DA/mrnet/stage2.pth")
# model_dir_baseline = os.path.join(root, "atdt/gta2cs/DA/max_square/GTA5_to_Cityscapes_MaxSquare.pth")
# model_dir_baseline = os.path.join(root, "atdt/gta2cs/DA/adaptsegnet/GTA5_multi.pth")
# model_dir_baseline = os.path.join(root, "atdt/gta2cs/DA/bdl/gta5_ssl.pth")
# model_dir_baseline = os.path.join(root, "atdt/gta2cs/DA/ltir/ResNet_GTA_50.2.pth")
# model_dir_baseline = os.path.join(root, "atdt/gta2cs/DA/stuff_and_things/BestGTA5_post_SSL.pth")
model_dir_baseline = os.path.join(root, "atdt/synthia2cs/DA/bdl/syn_2_city_deeplab.pth")
# model_dir_baseline = os.path.join(root, "atdt/gta2cs/DA/fada/g2c_sd.pth")
# model_dir_baseline0 = "/media/data_4t/aCardace/atdt/gta2cs/target_only_TA_augmented_bdl/ckpt/checkpoint.tar"

json_path = os.path.join(model_dir_transfer, 'params.json')
params = utils.Params(json_path)
params.device = device


# %%
# val_dl = dataloader.fetch_dataloader(data_dir, txt_val, 'val', params)
mean=np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
std=np.array((1, 1, 1), dtype=np.float32)

dataset = CS_test(root=data_dir, txt_file="/media/data_4t/aCardace/atdt/splits/cityscapes/val.txt", use_depth=True, threshold=50, size=(1024, 512), label_size=(2048, 1024), mean=mean, std=std, rgb=False, interpolation="bicubic")
val_dl = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

dataset1 = CS_test(root=data_dir, txt_file="/media/data_4t/aCardace/atdt/splits/cityscapes/val.txt", use_depth=True, threshold=50, size=(1024, 512), label_size=(2048, 1024), interpolation="lanczos")
val_dl1 = data.DataLoader(dataset1, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

model_source = Res_Deeplab(num_classes=1, use_sigmoid=True).to(device)
model_target = Res_Deeplab(num_classes=19).to(device)
# model_baseline = mrnet(num_classes=19, use_se=True, train_bn=False, norm_style="gn", droprate=0.2).to(device)
# model_baseline = maxsq(num_classes=19).to(device)
# model_baseline = maxsq(num_classes=19).to(device)
# model_baseline = Deeplab(num_classes=19).to(device)
# model_baseline2 = Deeplab(num_classes=19).to(device)

model_baseline = Deeplab(num_classes=19).to(device)
# model_baseline2 = Deeplab(num_classes=19).to(device)
# model_baseline0 = Res_Deeplab(num_classes=19, layers=23).to(device)

transfer = get_transfer().to(device)


# %%
ckpt_source_file_path = os.path.join(model_dir_source, 'ckpt', best_ckpt_filename)
saved_state_dict = torch.load(ckpt_source_file_path, map_location=device)["state_dict"]
model_source.load_state_dict(saved_state_dict)

ckpt_target_file_path = os.path.join(model_dir_target, 'ckpt', ckpt_filename)
saved_state_dict = torch.load(os.path.join(ckpt_target_file_path), map_location=device)["state_dict"]
model_target.load_state_dict(saved_state_dict)

ckpt_transfer_file_path = os.path.join(model_dir_transfer, 'ckpt', best_ckpt_filename)
saved_state_dict = torch.load(ckpt_transfer_file_path, map_location=device)["state_dict"]
transfer.load_state_dict(saved_state_dict)

#adaptive model 
source_encoder = torch.nn.Sequential(*(list(model_source.children())[:-2]))
target_encoder = torch.nn.Sequential(*(list(model_target.children())[:-1]))
target_decoder = list(model_target.children())[-1]
adaptive_model = get_adaptive_network(source_encoder, transfer, target_decoder)

# ckpt = torch.load("/content/drive/My Drive/projects/atdt/gta2cs/fn_transfer_decoder_transfer_43.1/ckpt/checkpoint.tar")
# adaptive_model.load_state_dict(ckpt["state_dict"])
# saved_state_dict = torch.load(model_dir_baseline0, map_location=device)
# model_baseline0.load_state_dict(saved_state_dict["state_dict"])


# %%
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
# saved_state_dict = torch.load(model_dir_baseline, map_location=device)
# model_baseline.load_state_dict(saved_state_dict)


# %%
#bdl -> Deeplab, lanczos, lanczos
saved_state_dict = torch.load(model_dir_baseline, map_location=device)
model_baseline.load_state_dict(saved_state_dict)


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
def eval(adpative_model, model_baseline, val_dl, val_dl1, params):
    # set model to evaluation mode
        adpative_model.eval()
        model_baseline.eval()
        valid_labels = range(19)
        x_num = 500

        scorer = ScoreUpdater(valid_labels, params.num_classes, x_num, None)
        acc = Accuracy(params.num_classes+1, ignore_index=19)
        scorer.reset()
        acc.reset()
        classes = np.arange(num_classes)
        inverted_w = np.load("/media/data_4t/aCardace/atdt/weights/inverted_weights_synthia.npy")
        
        baseline_weights = inverted_w/inverted_w.max()
        model_weights = 1-(inverted_w/inverted_w.max())
        iter_transfer = iter(val_dl1)

        with torch.no_grad():
            for i, (xb, yb, _, name) in enumerate(tqdm(val_dl)):

                xb = xb.to(device)
                yb = yb.to(device)

                xb_transfer, _, _, _ =  next(iter_transfer)
                xb_transfer = xb_transfer.to(device)
      
                z = adpative_model(xb_transfer)
                z = F.interpolate(z, size=(params.label_size[1],params.label_size[0]), mode='bilinear', align_corners=True)
                z = F.softmax(z/6, dim=1)
                
                # rescale transfer with softmax based on TA weights
                prediction_z = torch.argmax(z, dim=1)
                mask_z = torch.zeros(prediction_z.size()).to(params.device)
                for label in classes:
                    mask_z[prediction_z.eq(label)] = model_weights[label]

                z *= mask_z.unsqueeze(dim=1)
    
                model_target_out = model_baseline(xb)
                model_target_out = F.interpolate(model_target_out, size=(params.label_size[1],params.label_size[0]), mode='bilinear', align_corners=True)
                model_target_out = F.softmax(model_target_out/6, dim=1)

                prediction_baseline = torch.argmax(model_target_out, dim=1)
                mask_baseline = torch.zeros(prediction_baseline.size()).to(params.device)
                for label in classes:
                    mask_baseline[prediction_baseline.eq(label)] = baseline_weights[label]
                model_target_out *= mask_baseline.unsqueeze(dim=1)

                out = z + model_target_out
                out = F.softmax(out, dim=1)

                output = out[0].cpu().numpy()
                output = output.transpose(1, 2, 0)
                label, prob = np.argmax(output, axis=2), np.max(output, axis=2)

                scorer.update(label.flatten(), yb[0].cpu().numpy().flatten(), i)
                acc.add(out, yb)
        scorer.print_score()
        acc = acc.value()
        print('acc:', acc)
        

# %%
eval(adaptive_model, model_baseline, val_dl, val_dl1, params)


# %%


np.load("/media/data_4t/aCardace/atdt/weights/inverted_weights_synthia.npy")

# %%
