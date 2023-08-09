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
import dataloader.dataloader_DBST as dataloader
import utils.utils as utils
from model.losses import get_loss_fn
from model.metrics import get_metrics
from collections import OrderedDict
# from model.deeplab import Res_Deeplab
from model.deeplab_bn import Res_Deeplab
import torch.nn.functional as F
import torch.nn as nn
import time 

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/content/drive/My Drive/atdt',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir_target', default='experiments/segmentation_resnet50',
                    help="Directory containing params.json")
parser.add_argument('--checkpoint_dir_target', default="experiments/segmentation_resnet50/ckpt",
                    help="Directory containing weights target model weights")
parser.add_argument('--tensorboard_dir', default="experiments/transfer_baseline/tensorboard",
                    help="Directory for Tensorboard data")
parser.add_argument('--txt_train', default='/content/drive/My Drive/atdt/input_list_train_carla.txt',
                    help="Txt file containing path to training images")
parser.add_argument('--txt_val_target', default='/content/drive/My Drive/atdt/input_list_val_cityscapes.txt',
                    help="Txt file containing path to validation images target dataset")
parser.add_argument('--train_augmented_dir', default='', help="Directory containing augmented train images")

RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def inference(model, batch):
    model.eval()
    with torch.no_grad():
        y_pred = model(batch.to(device))
    return y_pred

def train_epoch(model, loss_fn, dataset_dl, opt=None, lr_scheduler=None, metrics=None, params=None):
    running_loss = utils.RunningAverage()
    num_batches = len(dataset_dl)

    if metrics is not None:
        for metric_name, metric in metrics.items():
            metric.reset()

    for (xb, yb) in tqdm(dataset_dl):
        xb = xb.to(params.device)
        yb = yb.to(params.device)
        
        out = model(xb)
        out = F.interpolate(out, size=(params.load_size[1],params.load_size[0]), mode='bilinear', align_corners=True)

        loss_b = loss_fn(out, yb)

        if opt is not None:
            opt.zero_grad()
            loss_b.backward()
            opt.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss.update(loss_b.item())

    if metrics is not None:
        metrics_results = OrderedDict({})
        for metric_name, metric in metrics.items():
            metrics_results[metric_name] = metric.value()
        return running_loss(), metrics_results
    else:
        return running_loss(), None


def train_and_evaluate(model, train_dl, val_dl_target, opt, loss_fn, metrics, params,
                       lr_scheduler, checkpoint_dir, ckpt_filename, log_dir, writer):
    ckpt_file_path = os.path.join(checkpoint_dir, ckpt_filename)
    best_value = -float('inf')
    # early_stopping = utils.EarlyStopping(patience=10, verbose=True)
    start_epoch = 0

    if os.path.exists(ckpt_file_path):
        model, opt, lr_scheduler, start_epoch, best_value = utils.load_checkpoint(model, opt, lr_scheduler,
                                                                start_epoch, False, best_value, checkpoint_dir, ckpt_filename)
        print("=> loaded model checkpoint form {} (epoch {})".format(
            ckpt_file_path, start_epoch))
    else:
        saved_state_dict = torch.utils.model_zoo.load_url(RESTORE_FROM)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = str(i).split('.')
            if not i_parts[1]=='fc' and not i_parts[1]=='layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)
        print("=> Initializing model from imagenet")

    model.to(params.device)
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    iter_val_target = iter(val_dl_target)
    batch_sample_target1, batch_gt_target1 = next(iter_val_target)
    batch_sample_target2, batch_gt_target2 = next(iter_val_target)
    batch_sample_target = torch.cat([batch_sample_target1, batch_sample_target2], dim=0)
    batch_gt_target = torch.cat([batch_gt_target1, batch_gt_target2], dim=0)
    
    for epoch in range(start_epoch, params.num_epochs):
        # Run one epoch

        iter_val_train = iter(train_dl)
        batch_sample_target_train1, batch_gt_target_train1 = next(iter_val_train)
        batch_sample_target_train2, batch_gt_target_train2 = next(iter_val_train)
        batch_sample_target_train = torch.cat([batch_sample_target_train1, batch_sample_target_train2], dim=0)
        batch_gt_target_train = torch.cat([batch_gt_target_train1, batch_gt_target_train2], dim=0)   
        
        current_lr = get_lr(opt)

        writer.add_scalar('Learning_rate', current_lr, epoch)

        predictions = inference(model, batch_sample_target_train)
        predictions = F.interpolate(predictions, size=(params.label_size[1],params.label_size[0]), mode='bilinear', align_corners=True)
        plot = train_dl.dataset.get_predictions_plot(
            batch_sample_target_train, predictions.cpu(), batch_gt_target_train)
        writer.add_image('Predictions_train', plot, epoch, dataformats='HWC')

        predictions = inference(model, batch_sample_target)
        predictions = F.interpolate(predictions, size=(params.label_size[1],params.label_size[0]), mode='bilinear', align_corners=True)
        plot = train_dl.dataset.get_predictions_plot(
            batch_sample_target, predictions.cpu(), batch_gt_target)
        writer.add_image('Predictions_val', plot, epoch, dataformats='HWC')

        logging.info('Epoch {}/{}, current lr={}'.format(epoch, params.num_epochs-1, current_lr))

        model.train()
        train_loss, _ = train_epoch(model, loss_fn, train_dl, opt, lr_scheduler, params=params)
    
        model.eval()
        # Evaluate for one epoch on validation set
        val_metrics_target = evaluate(
            model, val_dl_target, metrics=metrics, params=params)

        writer.add_scalars('Loss', {
            'Training': train_loss,
        }, epoch)

        for (val_metric_name, val_metric_results) in val_metrics_target.items():
            writer.add_scalars(val_metric_name, {
                'Validation': val_metric_results[0],
            }, epoch)

        current_value = list(val_metrics_target.values())[0][0]
        is_best = current_value >= best_value

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_value = current_value
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                log_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics_target, best_json_path)

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': opt.state_dict(),
                               'scheduler_dict': lr_scheduler.state_dict(),
                               'best_value': best_value},
                              is_best=is_best,
                              ckpt_dir=checkpoint_dir,
                              filename=ckpt_filename)

        logging.info("\ntrain loss: %.3f" %
                     train_loss)
        
        for (val_metric_name_t, val_metric_results_t) in val_metrics_target.items():
            logging.info("val %s: %.3f" % (val_metric_name_t, val_metric_results_t[0]))
        logging.info("-"*20)

        # early_stopping(val_loss_source)
        # if early_stopping.early_stop:
        #     logging.info("Early stopping")
        #     break

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()

    json_path = os.path.join(args.model_dir_target, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    ckpt_filename = "checkpoint.tar"
    best_ckpt_filename = "model_best.tar"
    writer = SummaryWriter(args.tensorboard_dir)

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params.device = device

    # Set the random seed for reproducible experiments
    seed = 6
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Set the logger
    log_dir = os.path.join(args.model_dir_target, "logs")
    if not os.path.exists(log_dir):
        print("Making log directory {}".format(log_dir))
        os.mkdir(log_dir)
    utils.set_logger(os.path.join(log_dir, "train.log"))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    train_dl = dataloader.fetch_dataloader(
        args.data_dir, args.train_augmented_dir, args.txt_train, 'train', params)
    val_dl_target = dataloader.fetch_dataloader(
        args.data_dir, args.train_augmented_dir, args.txt_val_target, 'val', params)

    logging.info("- done.")

    # Define the model and optimizer
    model = Res_Deeplab(num_classes=params.num_classes, layers=23)

    opt = optim.SGD(model.optim_parameters(params.learning_rate), lr=params.learning_rate, weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=params.learning_rate, steps_per_epoch=len(train_dl), epochs=params.num_epochs, div_factor=20)

    # fetch loss function and metrics
    loss_fn = get_loss_fn(params)
    metrics = OrderedDict({})
    for metric in params.metrics:
        metrics[metric] = get_metrics(metric, params)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))

    train_and_evaluate(model, train_dl, val_dl_target, opt, loss_fn, metrics,
                       params, lr_scheduler, args.checkpoint_dir_target, ckpt_filename, log_dir, writer)