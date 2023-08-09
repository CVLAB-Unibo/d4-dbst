"""Train the model"""

import argparse
import copy
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from evaluate import evaluate
import dataloader.dataloader as dataloader
import utils.utils as utils
from model.losses import get_loss_fn
from model.metrics import get_metrics
from model.net import get_network, get_transfer, get_adaptive_network, get_adaptive_network_combined
from collections import OrderedDict
from model.deeplab import Res_Deeplab

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/content/drive/My Drive/atdt',
                    help="Directory containing the dataset")

parser.add_argument('--model_dir_source', default='experiments/depth_resnet50',
                    help="Directory containing params.json")
parser.add_argument('--model_dir_target', default='experiments/segmentation_resnet50',
                    help="Directory containing params.json")
parser.add_argument('--model_dir_transfer', default='experiments/transfer_baseline',
                    help="Directory containing params.json")

parser.add_argument('--checkpoint_dir_source', default="experiments/depth_resnet50/ckpt",
                    help="Directory containing source model weights")
parser.add_argument('--checkpoint_dir_target', default="experiments/segmentation_resnet50/ckpt",
                    help="Directory containing weights target model weights")
parser.add_argument('--checkpoint_dir_transfer', default="experiments/transfer_baseline/ckpt",
                    help="Directory containing weights target model weights")

parser.add_argument('--txt_val', default='/content/drive/My Drive/atdt/input_list_val_cityscapes.txt',
                    help="Txt file containing path to validation images")

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()

    json_path = os.path.join(args.model_dir_source, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params_source = utils.Params(json_path)

    json_path = os.path.join(args.model_dir_target, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params_target = utils.Params(json_path)

    json_path = os.path.join(args.model_dir_transfer, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params_transfer = utils.Params(json_path)

    # ckpt_filename = "checkpoint.tar"
    best_ckpt_filename = "model_best.tar"
    # writer = SummaryWriter(args.tensorboard_dir)

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params_transfer.device = device

    # Set the random seed for reproducible experiments
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    params_transfer.encoding = params_transfer.encoding
    val_dl = dataloader.fetch_dataloader(
        args.data_dir, args.txt_val, 'val', params_transfer)

    # logging.info("- done.")

    # Define the model and optimizer
       # Define the model and optimizer
    model_source = Res_Deeplab(num_classes=1, use_sigmoid=True).to(params_transfer.device)
    model_target = Res_Deeplab(num_classes=19).to(params_transfer.device)
    transfer = get_transfer().to(params_transfer.device)

    ckpt_source_file_path = os.path.join(args.checkpoint_dir_source, best_ckpt_filename)
    if os.path.exists(ckpt_source_file_path):
        model_source = utils.load_checkpoint(model_source, ckpt_dir=args.checkpoint_dir_source, filename=best_ckpt_filename, is_best=True)[0]
        print("=> loaded source model checkpoint form {}".format(ckpt_source_file_path))
    else:
        print("=> Initializing source model from scratch")

    
    ckpt_target_file_path = os.path.join(args.checkpoint_dir_target, best_ckpt_filename)
    if os.path.exists(ckpt_target_file_path):
        model_target = utils.load_checkpoint(model_target, ckpt_dir=args.checkpoint_dir_target, filename=best_ckpt_filename, is_best=True)[0]
        print("=> loaded taregt model checkpoint form {}".format(ckpt_target_file_path))
    else:
        print("=> Initializing target model from scratch")
    
    ckpt_transfer_file_path = os.path.join(args.checkpoint_dir_transfer, best_ckpt_filename)
    if os.path.exists(ckpt_transfer_file_path):
        model_transfer = utils.load_checkpoint(transfer, ckpt_dir=args.checkpoint_dir_transfer, filename=best_ckpt_filename, is_best=True)[0]
        print("=> loaded transfer checkpoint form {}".format(ckpt_transfer_file_path))
    else:
        print("=> Initializing from scratch")
    
    metrics = OrderedDict({})
    for metric in params_transfer.metrics:
        metrics[metric] = get_metrics(metric, params_transfer)

    #construct graph adaptation model
    source_encoder = torch.nn.Sequential(*(list(model_source.children())[:-2]))
    target_encoder = torch.nn.Sequential(*(list(model_target.children())[:-1]))
    target_decoder = list(model_target.children())[-1]
    adpative_model = get_adaptive_network(source_encoder, transfer, target_decoder)
    # adpative_model = get_adaptive_network_combined(source_encoder, transfer, target_encoder, target_decoder)

     # Evaluate
    val_metrics = evaluate(adpative_model, val_dl, metrics=metrics, params=params_transfer)        

    best_json_path = os.path.join(args.model_dir_transfer, "logs/evaluation.json")
    for val_metric_name, val_metric_results in val_metrics.items():
        print("{}: {}".format(val_metric_name, val_metric_results))
    utils.save_dict_to_json(val_metrics, best_json_path)