"""Train the model"""

import argparse
import copy
from json import encoder
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import dataloader.dataloader as dataloader
import utils.utils as utils
from evaluate import evaluate
from model.losses import get_loss_fn
from model.metrics import get_metrics
from model.net import get_transfer, get_adaptive_network
from model.deeplab import Res_Deeplab
from collections import OrderedDict
from torch.utils import data, model_zoo
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/content/drive/My Drive/atdt',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/baseline',
                    help="Directory containing params.json")
parser.add_argument('--checkpoint_dir', default="experiments/baseline/ckpt",
                    help="Directory containing weights to reload before \
                    training")
parser.add_argument('--tensorboard_dir', default="experiments/baseline/tensorboard",
                    help="Directory for Tensorboard data")
parser.add_argument('--txt_train', default='/content/drive/My Drive/atdt/input_list_train_mixed_carla_cityscapes.txt',
                    help="Txt file containing path to training images")
parser.add_argument('--txt_val1', default='/content/drive/My Drive/atdt/input_list_val_carla.txt',
                    help="Txt file containing path to validation images")
parser.add_argument('--txt_val2', default='/content/drive/My Drive/atdt/input_list_val_carla.txt',
                    help="Txt file containing path to validation images")

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def inference(model, batch):
    model.eval()
    with torch.no_grad():
        y_pred = model(batch.to(device))
        y_pred = y_pred
    return y_pred

def train_epoch(encoder, transfer, aspp, loss_fn, dataset_dl, opt=None, lr_scheduler=None, metrics=None, params=None):
    running_loss = utils.RunningAverage()
    num_batches = len(dataset_dl)

    if metrics is not None:
        for metric_name, metric in metrics.items():
            metric.reset()

    for (xb, yb) in tqdm(dataset_dl):
        xb = xb.to(params.device)
        yb = yb.to(params.device)

        with torch.no_grad():
            features = encoder(xb)
            features = transfer(features)

        output = aspp(features)
        output = F.interpolate(output, size=(params.load_size[1],params.load_size[0]), mode='bilinear', align_corners=True)
        loss_b = loss_fn(output, yb)

        if opt is not None:
            opt.zero_grad()
            loss_b.backward()
            opt.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss.update(loss_b.item())

        if metrics is not None:
            for metric_name, metric in metrics.items():
                metric.add(output.detach(), yb)

    if metrics is not None:
        metrics_results = OrderedDict({})
        for metric_name, metric in metrics.items():
            metrics_results[metric_name] = metric.value()
        return running_loss(), metrics_results
    else:
        return running_loss(), None


def train_and_evaluate(encoder, transfer, aspp, train_dl, val_dl1, val_dl2, opt, loss_fn, metrics, lr_scheduler, 
                        params, checkpoint_dir, ckpt_filename, log_dir, writer):

    adpative_model = get_adaptive_network(encoder, transfer, aspp)

    ckpt_file_path = os.path.join(checkpoint_dir, ckpt_filename)
    best_value = -float('inf')
    early_stopping = utils.EarlyStopping(patience=10, verbose=True)
    start_epoch = 0


    batch_sample_train, batch_gt_train = next(iter(train_dl))
    batch_sample_val1, batch_gt_val1 = next(iter(val_dl1))
    batch_sample_val2, batch_gt_val2 = next(iter(val_dl2))

    if os.path.exists(ckpt_file_path):
        adpative_model, opt, lr_scheduler, start_epoch, best_value = utils.load_checkpoint(adpative_model, opt, lr_scheduler,
                                start_epoch, False, best_value, checkpoint_dir, ckpt_filename)
        # adpative_model = utils.load_checkpoint(adpative_model, start_epoch=start_epoch, ckpt_dir=checkpoint_dir)[0]                                
        print("=> loaded checkpoint form {} (epoch {})".format(
            ckpt_file_path, start_epoch))
    else:
        print("=> Initializing from scratch")

    adpative_model.eval()

    for epoch in range(start_epoch, params.num_epochs):

        # Run one epoch
        current_lr = get_lr(opt)
        logging.info('Epoch {}/{}, current lr={}'.format(epoch,
                                                         params.num_epochs-1, current_lr))

        # if epoch % 5 == 0 or epoch==params.num_epochs-1: 
        predictions = inference(adpative_model, batch_sample_train)
        output = F.interpolate(predictions, size=(params.label_size[1],params.label_size[0]), mode='bilinear', align_corners=True)
        plot = train_dl.dataset.get_predictions_plot(
            batch_sample_train, output.cpu(), batch_gt_train)
        writer.add_image('Predictions_train', plot,
                            epoch, dataformats='HWC')

        predictions = inference(adpative_model, batch_sample_val1)
        output = F.interpolate(predictions, size=(params.label_size[1],params.label_size[0]), mode='bilinear', align_corners=True)
        plot = train_dl.dataset.get_predictions_plot(
            batch_sample_val1, output.cpu(), batch_gt_val1)
        writer.add_image('Predictions_val_source', plot, epoch, dataformats='HWC')

        predictions = inference(adpative_model, batch_sample_val2)
        output = F.interpolate(predictions, size=(params.label_size[1],params.label_size[0]), mode='bilinear', align_corners=True)
        plot = train_dl.dataset.get_predictions_plot(
            batch_sample_val2, output.cpu(), batch_gt_val2)
        writer.add_image('Predictions_val_target', plot, epoch, dataformats='HWC')

        train_loss, train_metrics = train_epoch(
            encoder, transfer, aspp, loss_fn, train_dl, opt, lr_scheduler=lr_scheduler, metrics=metrics, params=params)
    
        # Evaluate for one epoch on validation set
        val_metrics1 = evaluate(
            adpative_model, val_dl1, metrics=metrics, params=params)
        val_metrics2 = evaluate(
            adpative_model, val_dl2, metrics=metrics, params=params)
        
        writer.add_scalar('Learning_rate', current_lr, epoch)

        writer.add_scalars('Loss', {
            'Training': train_loss,
            # 'Validation_source': val_loss1,
            # 'Validation_target': val_loss2
        }, epoch)

        for (train_metric_name, train_metric_results), (val_metric_name1, val_metric_results1), (val_metric_name2, val_metric_results2) in \
                    zip(train_metrics.items(), val_metrics1.items(), val_metrics2.items()):
            writer.add_scalars(train_metric_name, {
                'Training': train_metric_results[0],
                'Validation_source': val_metric_results1[0],
                'Validation_target': val_metric_results2[0],
            }, epoch)

        current_value = list(val_metrics1.values())[0][0]
        is_best = current_value >= best_value

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_value = current_value
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                log_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics1, best_json_path)
            utils.save_dict_to_json(val_metrics2, best_json_path)

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': adpative_model.state_dict(),
                               'optim_dict': opt.state_dict(),
                               'scheduler_dict': lr_scheduler.state_dict(),
                               'best_value': best_value},
                              is_best=is_best,
                              ckpt_dir=checkpoint_dir,
                              filename=ckpt_filename)

        # logging.info("\ntrain loss: %.3f, val loss source: %.3f, val loss target: %.3f" %
                    #  (train_loss, val_loss1, val_loss2))
        logging.info("\ntrain loss: %.3f" % (train_loss))                    
        for (train_metric_name, train_metric_results), (val_metric_name1, val_metric_results1), (val_metric_name2, val_metric_results2) in \
                        zip(train_metrics.items(), val_metrics1.items(), val_metrics2.items()):
            logging.info("train %s: %.3f, val source %s: %.3f, val target %s: %.3f" % (
                train_metric_name, train_metric_results[0], val_metric_name1, val_metric_results1[0], val_metric_name2, val_metric_results2[0]))

        logging.info("-"*20)

        #early_stopping(current_value)
        #if early_stopping.early_stop:
        #    logging.info("Early stopping")
        #    break


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    ckpt_filename = "checkpoint.tar"
    writer = SummaryWriter(args.tensorboard_dir)

    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params.device = device

    # Set the random seed for reproducible experiments
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    # Set the logger
    log_dir = os.path.join(args.model_dir, "logs")
    if not os.path.exists(log_dir):
        print("Making log directory {}".format(log_dir))
        os.mkdir(log_dir)
    utils.set_logger(os.path.join(log_dir, "train.log"))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    train_dl = dataloader.fetch_dataloader(
        args.data_dir, args.txt_train, "train", params)
    val_dl1 = dataloader.fetch_dataloader(
        args.data_dir, args.txt_val1, "val", params)
    val_dl2 = dataloader.fetch_dataloader(
        args.data_dir, args.txt_val2, "val", params)

    logging.info("- done.")

    # Define the model and optimizer
    # model = get_network(params).to(params.device)
    model = Res_Deeplab(num_classes=params.num_classes).to(params.device)
    # saved_state_dict = torch.utils.model_zoo.load_url(model_urls['resnet50'])
    saved_state_dict = torch.load("/home/ldeluigi/dev/atdt-da2/synthia2cs/net1_original_high_res_gt/ckpt/model_best.tar")["state_dict"]
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = str(i).split('.')
        if not i_parts[0]=='layer5':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
    model.load_state_dict(new_params)

    transfer = get_transfer().to(params.device)
    saved_state_dict = torch.load("/home/ldeluigi/dev/atdt-da2/synthia2cs/transfer_net1_gt_V1_2_net2/ckpt/model_best.tar")["state_dict"]
    transfer.load_state_dict(saved_state_dict)

    encoder = torch.nn.Sequential(*(list(model.children())[:-1]))
    aspp = list(model.children())[-1]

    opt = optim.SGD(aspp.parameters(), lr=params.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=params.learning_rate, steps_per_epoch=len(train_dl), epochs=params.num_epochs, div_factor=20)

    # fetch loss function and metrics
    loss_fn = get_loss_fn(params)

    metrics = OrderedDict({})
    for metric in params.metrics:
        metrics[metric] = get_metrics(metric, params)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(encoder, transfer, aspp, train_dl, val_dl1, val_dl2, opt, loss_fn, metrics, lr_scheduler,
                       params, args.checkpoint_dir, ckpt_filename, log_dir, writer)