import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.CNN_with_Swin_E_CNN_D import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, default='checkpoint/CNN_with_Swin_E_CNN_D', help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iteration number to train')
parser.add_argument('--max_epochs', type=int,
                    default=250, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.05,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="use gradient checkpointing")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='evaluation only')
parser.add_argument('--throughput', action='store_true', help='test throughput only')
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs.",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='data caching strategy')

args = parser.parse_args()

if args.dataset == "Synapse" and "train_npz" not in args.root_path:
    args.root_path = os.path.join(args.root_path, "train_npz")

config = get_config(args)

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net.load_from(config)

    # Handle checkpoint loading
    start_epoch = 0
    optimizer_state = None
    scheduler_state = None

    if args.resume is not None and os.path.isfile(args.resume):
        print(f"🔄 Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)

        if 'model' in checkpoint:
            net.load_state_dict(checkpoint['model'])
            print("✅ Model state loaded.")
        else:
            net.load_state_dict(checkpoint)
            print("✅ Model weights loaded (no optimizer/scheduler).")

        optimizer_state = checkpoint.get('optimizer', None)
        scheduler_state = checkpoint.get('scheduler', None)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"▶️ Resuming from epoch: {start_epoch}")
    else:
        print("🚀 Starting training from scratch.")

    # Call the trainer
    trainer = {'Synapse': trainer_synapse}
    trainer[dataset_name](args, net, args.output_dir, start_epoch, optimizer_state, scheduler_state)
