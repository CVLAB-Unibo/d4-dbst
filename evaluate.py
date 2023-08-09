"""Evaluates the model"""

import argparse
import os
import random
import numpy as np
import torch
import utils.utils as utils
from model.net import get_network
import PIL.Image as pil

from tqdm import tqdm
import dataloader.dataloader as dataloader
from model.losses import get_loss_fn
from model.metrics import get_metrics
from collections import namedtuple
from model.deeplab import Res_Deeplab
# from model.deeplab_multi_mrnet import DeeplabMulti
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/baseline',
                    help="Directory containing params.json")
parser.add_argument('--checkpoint_dir', default="experiments/baseline/checkpoints",
                    help="Directory containing weights to reload before \
                    training")
parser.add_argument('--txt_val', default='/content/drive/My Drive/atdt/input_list_val_carla.txt',
                    help="Txt file containing path to validation images")

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id','color'])

encoding = [
        CityscapesClass('unlabeled',        0,      19,      (0, 0, 0)),
        CityscapesClass('ambiguous',        1,      19,      (0, 0, 0)),
        CityscapesClass('sky',              2,      10,      (70, 130, 180)),
        CityscapesClass('road',             3,       0,      (128, 64, 128)),
        CityscapesClass('sidewalk',         4,       1,      (244, 35, 232)),
        CityscapesClass('railtrack',        5,      19,      (0, 0, 0)),
        CityscapesClass('terrain',          6,       9,      (152, 251, 152)),
        CityscapesClass('tree',             7,       8,      (107, 142, 35)),
        CityscapesClass('vegetation',       8,       8,      (107, 142, 35)),
        CityscapesClass('building',         9,       2,      (70, 70, 70)),
        CityscapesClass('infrastructure',   10,      5,      (153, 153, 153)),
        CityscapesClass('fence',            11,      4,      (190, 153, 153)),
        CityscapesClass('billboard',        12,      19,     (0, 0, 0)),
        CityscapesClass('trafficlight',     13,      6,      (250, 170, 30)),
        CityscapesClass('trafficsign',      14,      7,      (220, 220, 0)),
        CityscapesClass('mobilebarrier',    15,      3,      (102, 102, 156)),
        CityscapesClass('firehydrant',      16,      19,     (0, 0, 0)),
        CityscapesClass('chair',            17,      19,     (0, 0, 0)),
        CityscapesClass('trash',            18,      19,     (0, 0, 0)),
        CityscapesClass('trashcan',         19,      19,     (0, 0, 0)),
        CityscapesClass('person',           20,      11,     (220, 20, 60)),
        CityscapesClass('animal',           21,      12,     (255, 0, 0)),
        CityscapesClass('bicycle',          22,      18,     (119, 11, 32)),
        CityscapesClass('motorcycle',       23,      17,     (0, 0, 230)),
        CityscapesClass('car',              24,      13,     (0, 0, 142)),
        CityscapesClass('van',              25,      13,     (0, 0, 142)),
        CityscapesClass('bus',              26,      15,     (0, 60, 100)),
        CityscapesClass('truck',            27,      14,     (0, 0, 70)),
        CityscapesClass('trailer',          28,      19,     (0, 0, 0)),
        CityscapesClass('train',            29,      16,     (0, 80, 100)),
        CityscapesClass('plane',            30,      19,     (0, 0, 0)),
        CityscapesClass('boat',             31,      19,     (0, 0, 0)),
    ]
palette = []
colors = {cs_class.train_id: cs_class.color for cs_class in encoding}
for train_id, color in sorted(colors.items(), key=lambda item: item[0]):
    R, G, B = color
    palette.extend((R, G, B))

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def evaluate(model, dataset_dl, metrics=None, params=None):

    # set model to evaluation mode
    model.eval()

    metrics_results = {}

    if metrics is not None:
        for metric_name, metric in metrics.items():
            metric.reset()

    with torch.no_grad():
        for i, (xb, yb) in enumerate(tqdm(dataset_dl)):
            xb = xb.to(params.device)
            yb = yb.to(params.device)

            output = model(xb)
            output = F.interpolate(output, size=(params.label_size[1],params.label_size[0]), mode='bilinear', align_corners=True)
            
            # if i<10:
            #     prediction = output[0].cpu().numpy().argmax(0)
            #     image = pil.fromarray(np.uint8(prediction)).convert('P')
            #     image.putpalette(palette)
            #     image.save('/home/adricarda/projects/atdt/samples/{}.png'.format(i))

            if metrics is not None:
                for metric_name, metric in metrics.items():
                    metric.add(output, yb)

    if metrics is not None:
        for metric_name, metric in metrics.items():
            metrics_results[metric_name] = metric.value()
    
    return metrics_results

if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params.device = device

    # Set the random seed for reproducible experiments
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # fetch dataloaders
    val_dl = dataloader.fetch_dataloader(
        args.data_dir, args.txt_val, 'val', params)

    # Define the model
    
    if params.task =='depth':
        model = Res_Deeplab(num_classes=1, use_sigmoid=True).to(params.device)    
    else:
        model = Res_Deeplab(num_classes=params.num_classes,  layers=6).to(device)
    model = utils.load_checkpoint(model, is_best=True, ckpt_dir=args.checkpoint_dir)[0]

    # fetch loss function and metrics
    loss_fn = get_loss_fn(params)
    metrics = {}
    for metric in params.metrics:
        metrics[metric] = get_metrics(metric, params)

    # Reload weights from the saved file

    model = utils.load_checkpoint(model, ckpt_dir=args.checkpoint_dir, is_best=True)[0]                                

    # Evaluate
    val_metrics = evaluate(
        model, val_dl, metrics=metrics, params=params)

    if not os.path.isdir(os.path.join(args.model_dir, "logs")):
        os.mkdir(os.path.join(args.model_dir, "logs"))

    best_json_path = os.path.join(args.model_dir, "logs/evaluation.json")

    for val_metric_name, val_metric_results in val_metrics.items():
        print("{}: {}".format(val_metric_name, val_metric_results))
    utils.save_dict_to_json(val_metrics, best_json_path)