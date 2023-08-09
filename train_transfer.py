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
from model.net import get_network, get_transfer, get_adaptive_network
from collections import OrderedDict
from model.deeplab import Res_Deeplab
import torch.nn.functional as F

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

parser.add_argument('--tensorboard_dir', default="experiments/transfer_baseline/tensorboard",
                    help="Directory for Tensorboard data")
parser.add_argument('--txt_train', default='/content/drive/My Drive/atdt/input_list_train_carla.txt',
                    help="Txt file containing path to training images")          
parser.add_argument('--txt_val_source', default='/content/drive/My Drive/atdt/input_list_val_carla.txt',
                    help="Txt file containing path to validation images source dataset")
parser.add_argument('--txt_val_target', default='/content/drive/My Drive/atdt/input_list_val_cityscapes.txt',
                    help="Txt file containing path to validation images target dataset")

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def inference(model, batch):
    model.eval()
    with torch.no_grad():
        y_pred = model(batch.to(device))
    return y_pred

def train_epoch(source_encoder, target_encoder, transfer, loss_fn, dataset_dl, opt=None, lr_scheduler=None, metrics=None, params=None):
    running_loss = utils.RunningAverage()
    num_batches = len(dataset_dl)

    if metrics is not None:
        for metric_name, metric in metrics.items():
            metric.reset()

    for (xb, _) in tqdm(dataset_dl):
        xb = xb.to(params.device)

        output_source_encoder = source_encoder(xb)
        output_target_encoder = target_encoder(xb)
        output_transfer = transfer(output_source_encoder)
        loss_b = loss_fn(output_transfer, output_target_encoder)

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

def train_and_evaluate(model_source, model_target, transfer, train_dl, val_dl_source, val_dl_target, opt, loss_fn, metrics, params,
                       lr_scheduler, checkpoint_dir, ckpt_filename, log_dir, writer):

    ckpt_file_path = os.path.join(checkpoint_dir, ckpt_filename)
    best_value = -float('inf')
    early_stopping = utils.EarlyStopping(patience=10, verbose=True)
    start_epoch = 0

    batch_sample_source, batch_gt_source = next(iter(val_dl_source))
    batch_sample_target, batch_gt_target = next(iter(val_dl_target))

    if os.path.exists(ckpt_file_path):
        transfer, opt, lr_scheduler, start_epoch, best_value = utils.load_checkpoint(transfer, opt, lr_scheduler,
                                                                start_epoch, False, best_value, checkpoint_dir, ckpt_filename)
        print("=> loaded transfer checkpoint form {} (epoch {})".format(
            ckpt_file_path, start_epoch))
    else:
        print("=> Initializing transfer from scratch")

    source_encoder = torch.nn.Sequential(*(list(model_source.children())[:-2]))
    target_encoder = torch.nn.Sequential(*(list(model_target.children())[:-1]))
    target_decoder = list(model_target.children())[-1]

    adpative_model = get_adaptive_network(source_encoder, transfer, target_decoder)

    for epoch in range(start_epoch, params.num_epochs):
        # Run one epoch
        current_lr = get_lr(opt)
        logging.info('Epoch {}/{}, current lr={}'.format(epoch, params.num_epochs-1, current_lr))

        writer.add_scalar('Learning_rate', current_lr, epoch)

        predictions = inference(adpative_model, batch_sample_source)
        predictions = F.interpolate(predictions, size=(params.label_size[1],params.label_size[0]), mode='bilinear', align_corners=True)
        plot = train_dl.dataset.get_predictions_plot(
            batch_sample_source, predictions.cpu(), batch_gt_source)
        writer.add_image('Predictions_source', plot, epoch, dataformats='HWC')

        predictions = inference(adpative_model, batch_sample_target)
        predictions = F.interpolate(predictions, size=(params.label_size[1],params.label_size[0]), mode='bilinear', align_corners=True)
        plot = train_dl.dataset.get_predictions_plot(
            batch_sample_target, predictions.cpu(), batch_gt_target)
        writer.add_image('Predictions_target', plot, epoch, dataformats='HWC')

        predictions = inference(model_target, batch_sample_target)
        predictions = F.interpolate(predictions, size=(params.label_size[1],params.label_size[0]), mode='bilinear', align_corners=True)
        plot = train_dl.dataset.get_predictions_plot(
            batch_sample_target, predictions.cpu(), batch_gt_target)
        writer.add_image('Predictions_target_net2', plot, epoch, dataformats='HWC')

        transfer.train()
        train_loss, train_metrics = train_epoch(
            source_encoder, target_encoder, transfer, loss_fn, train_dl, opt, lr_scheduler, params=params)
    
        transfer.eval()
        val_loss_source, _ = train_epoch(
            source_encoder, target_encoder, transfer, loss_fn, val_dl_source, params=params)
    
        # Evaluate for one epoch on validation set
        val_metrics_source = evaluate(
            adpative_model, val_dl_source, metrics=metrics, params=params)

        val_metrics_target = evaluate(
            adpative_model, val_dl_target, metrics=metrics, params=params)

        writer.add_scalars('Loss', {
            'Training': train_loss,
        }, epoch)

        for (val_metric_name_s, val_metric_results_s), (val_metric_name_t, val_metric_results_t) in zip(val_metrics_source.items(), val_metrics_target.items()):
            writer.add_scalars(val_metric_name_s, {
                'Validation_source': val_metric_results_s[0],
                'Validation_target': val_metric_results_t[0],
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
            utils.save_dict_to_json(val_metrics_source, best_json_path)
            utils.save_dict_to_json(val_metrics_target, best_json_path)

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': transfer.state_dict(),
                               'optim_dict': opt.state_dict(),
                               'scheduler_dict': lr_scheduler.state_dict(),
                               'best_value': best_value},
                              is_best=is_best,
                              ckpt_dir=checkpoint_dir,
                              filename=ckpt_filename)

        logging.info("\ntrain loss: %.3f" %
                     train_loss)
        
        for (val_metric_name_s, val_metric_results_s), (val_metric_name_t, val_metric_results_t) in zip(val_metrics_source.items(), val_metrics_target.items()):
            logging.info("source %s: %.3f, target %s: %.3f" % (val_metric_name_s, val_metric_results_s[0], val_metric_name_t, val_metric_results_t[0]))
        logging.info("-"*20)

        # early_stopping(val_loss_source)
        # if early_stopping.early_stop:
        #     logging.info("Early stopping")
        #     break

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

    ckpt_filename = "checkpoint.tar"
    best_ckpt_filename = "model_best.tar"
    writer = SummaryWriter(args.tensorboard_dir)

    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params_transfer.device = device

    # Set the random seed for reproducible experiments
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Set the logger
    log_dir = os.path.join(args.model_dir_transfer, "logs")
    if not os.path.exists(log_dir):
        print("Making log directory {}".format(log_dir))
        os.mkdir(log_dir)
    utils.set_logger(os.path.join(log_dir, "train.log"))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    params_transfer.encoding = "cs"
    # fetch dataloaders
    train_dl = dataloader.fetch_dataloader(
        args.data_dir, args.txt_train, 'train', params_transfer, use_data_augmentation=True)

    val_dl_source = dataloader.fetch_dataloader(
        args.data_dir, args.txt_val_source, 'val', params_transfer)
    
    params_transfer.encoding = "cs"
    val_dl_target = dataloader.fetch_dataloader(
        args.data_dir, args.txt_val_target, 'val', params_transfer)

    logging.info("- done.")

    # Define the model and optimizer
    model_source = Res_Deeplab(num_classes=1, use_sigmoid=True, layers=6).to(params_transfer.device)
    model_target = Res_Deeplab(num_classes=params_transfer.num_classes, layers=6).to(params_transfer.device)

    #load source and target model before training and extract backbones
    ckpt_source_file_path = os.path.join(args.checkpoint_dir_source, best_ckpt_filename)
    print(ckpt_source_file_path)
    if os.path.exists(ckpt_source_file_path):
        model_source = utils.load_checkpoint(model_source, ckpt_dir=args.checkpoint_dir_source, filename=best_ckpt_filename, is_best=True)[0]
        print("=> loaded source model checkpoint form {}".format(ckpt_source_file_path))
    else:
        print("=> Initializing source model from scratch")

    ckpt_target_file_path = os.path.join(args.checkpoint_dir_target, best_ckpt_filename)
    print(ckpt_target_file_path)
    if os.path.exists(ckpt_target_file_path):
        model_target = utils.load_checkpoint(model_target, ckpt_dir=args.checkpoint_dir_target, filename=best_ckpt_filename, is_best=True)[0]
        print("=> loaded target model checkpoint form {}".format(ckpt_target_file_path))
    else:
        print("=> Initializing target model from scratch")

    transfer = get_transfer().to(params_transfer.device)
    
    for p in model_source.parameters():
        p.requires_grad = False
    model_source.eval()

    for p in model_target.parameters():
        p.requires_grad = False    
    model_target.eval()

    opt = optim.AdamW(transfer.parameters(), lr=params_transfer.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=params_transfer.learning_rate, steps_per_epoch=len(train_dl), epochs=params_transfer.num_epochs, div_factor=20)

    # fetch loss function and metrics
    loss_fn = get_loss_fn(params_transfer)
    # num_classes+1 for background.
    metrics = OrderedDict({})
    for metric in params_transfer.metrics:
        metrics[metric] = get_metrics(metric, params_transfer)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params_transfer.num_epochs))

    train_and_evaluate(model_source, model_target, transfer, train_dl, val_dl_source, val_dl_target, opt, loss_fn, metrics,
                       params_transfer, lr_scheduler, args.checkpoint_dir_transfer, ckpt_filename, log_dir, writer)