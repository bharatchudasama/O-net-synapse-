import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume

# === ✅ GLOBAL SEED for workers ===
GLOBAL_WORKER_SEED = 1234  # will be overridden by args.seed dynamically

def worker_init_fn(worker_id):
    seed = GLOBAL_WORKER_SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)

def trainer_synapse(args, model, snapshot_path, start_epoch=0, optimizer_state=None, scheduler_state=None):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    global GLOBAL_WORKER_SEED
    GLOBAL_WORKER_SEED = args.seed

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])
    )
    print("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(num_classes)

    # === 🧠 Create optimizer and load state if resuming
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        logging.info("🔁 Optimizer state loaded from checkpoint.")

    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = start_epoch * len(trainloader)
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)

    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} total iterations.")

    best_performance = 0.0
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            valid_labels = label_batch.clone()
            valid_labels[valid_labels >= num_classes] = 0

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, valid_labels.long())
            loss_dice = dice_loss(outputs, valid_labels, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Adjust learning rate manually (simple decay)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info(f"iteration {iter_num} : loss : {loss.item():.6f}, loss_ce: {loss_ce.item():.6f}")
            # ✅ Save model every 1000 iterations
            if iter_num % 10000 == 0:
                iter_save_path = os.path.join(snapshot_path, f'iter_{iter_num}.pth')
                torch.save({
                    'iteration': iter_num,
                    'epoch': epoch_num,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, iter_save_path)
                logging.info(f"💾 Saved checkpoint at iteration {iter_num} to {iter_save_path}")

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)

                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = valid_labels[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # Save model periodically
        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save({
                'epoch': epoch_num,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_mode_path)
            logging.info(f"✅ Saved model to {save_mode_path}")

        # Final save at the end
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save({
                'epoch': epoch_num,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_mode_path)
            logging.info(f"✅ Saved final model to {save_mode_path}")
            iterator.close()
            break

    writer.close()
    return "🎉 Training Finished!"
