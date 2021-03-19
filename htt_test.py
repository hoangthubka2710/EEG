import argparse
import random
import time
from abc import ABC

from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
import torch.nn as nn
import warnings
# from models.resnet import resnet18

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from os.path import join, isfile, isdir
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

import cv2
import pickle
from scipy.stats import pearsonr, spearmanr
import shutil


from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils import get_current_time


device = 'cuda'

#-------------------------------------------DATA-------------------------------------------------------#
current_path = os.getcwd()
DEAP_folder = '/data/EEG_SIGNAL/DEAP'

deapfiles = glob.glob(join(DEAP_folder, 'data_preprocessed_python', '*.dat'))
deapfiles = sorted(deapfiles)
print(len(deapfiles))

#-------------------------------------------DATA LOADER---------------------------------------------------#


def get_pearson_corrmap(array_encode):
    # array_encode[0] is the order of the subject
    # array_encode[1] is the order of the trial

    f = open(deapfiles[array_encode[0]], 'rb')

    x = pickle.load(f, encoding='latin1')
    no_channels = 32

    channel_patch = []
    for i in range(0, no_channels):
        chdata = x['data'][array_encode[1]][i]
        chdata = chdata[128*3:-1]
        channel_patch.append(chdata)

    corr_mat = np.zeros((32, 32))
    for i in range(0, no_channels):
        for j in range(0, no_channels):
            pearson_corr, _ = pearsonr(channel_patch[i], channel_patch[j])
            corr_mat[i, j] = pearson_corr

    label = x['labels'][array_encode[1]]

    return corr_mat, label


class Dataloader_img(Dataset):
    def __init__(self, list_IDs):
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        # Empty array for image
        X = np.empty((32, 32))
        # Label
        y = np.empty((1))

        corr_mat, label = get_pearson_corrmap(self.list_IDs[idx])
        corr_mat = (corr_mat+1)/2   # normalize
        valence = 1 if label[0] > 5.0 else 0
        arousal = 1 if label[1] > 5.0 else 0
        corr_mat = np.expand_dims(corr_mat, axis=0)
        final_img = np.concatenate([corr_mat, corr_mat, corr_mat], axis=0)

        # corr_mat = np.expand_dims(corr_mat,axis=0)
        # size image: 32x32x1
        valence = np.asarray(valence)
        # 0->1

        return {'image': final_img, 'labels': valence}


def main():
    list_best = []
    list_acc = []

    # Initialize Tensorboard Log Writer
    writer = SummaryWriter(log_dir=None)

    # Initialize checkpoint directory
    dir_to_save = f'checkpoints/{get_current_time()}'
    os.makedirs(dir_to_save, exist_ok=True)


    for idx in range(len(deapfiles)):
        is_best = 0
        lr = 1e-5
        BATCH_SIZE = 32
        num_epochs = 200
        num_classes = 1

        ignored_value = idx       # leave one subject out
        deap_no_trial = 40
        deap_no_channels = 32
        deap_no_subject = 32
        deapval_subject_array = [ignored_value]
        deap_trial_array = np.arange(0, deap_no_trial)
        deaptrain_subject_array = np.arange(0, deap_no_subject)
        deaptrain_subject_array = np.delete(
            deaptrain_subject_array, ignored_value)

        deaptrain_genarray = np.array(
            [[i, j] for i in deaptrain_subject_array for j in deap_trial_array])
        deapval_genarray = np.array(
            [[i, j] for i in deapval_subject_array for j in deap_trial_array])

        print("Number of Testing object: {}".format(ignored_value))

        train_dataset = Dataloader_img(deaptrain_genarray)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  num_workers=16,
                                  shuffle=True, pin_memory=True)

        val_dataset = Dataloader_img(deapval_genarray)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=40,
                                num_workers=8,
                                shuffle=False, pin_memory=True)

        model = models.resnet101(num_classes=num_classes, pretrained=True)
        # weight = model.conv1.weight.clone()

        # # import pdb; pdb.set_trace()
        # model.conv1 = nn.Conv2d(1,64,kernel_size=7, stride=2, padding=3, bias=False)
        # with torch.no_grad():
        #     model.conv1.weight = nn.Parameter(weight[:,0])

        model.to(device)
        # model.load_state_dict(torch.load('./checkpoints/valence_wave.pth'))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        start_time = time.time()

        optimizer.zero_grad()
        optimizer.step()

        for epoch in range(0, num_epochs):
            print('Epoch [{}/{}]'.format(epoch, num_epochs))
            # print('Current learning rate: ', optimizer.param_groups[0]['lr'])
            model.train()
            batch_loss = 0
            s_time = time.time()

            # Training
            for batch_idx, sample in enumerate(tqdm(train_loader)):
                warnings.filterwarnings("ignore")
                # import pdb; pdb.set_trace()a
                image = sample['image'].to(device).float()
                labels = sample['labels'].to(device).float()
                labels = labels.unsqueeze(1)

                # import pdb;pdb.set_trace()
                # FORWARD AND BACK PROP
                out = model(image)
                loss = criterion(out, labels)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
            e_time = time.time() - s_time

            train_loss = batch_loss / len(train_loader)  # (batch_idx + 1)
            print('Time: {:.1f}s '.format(e_time), end='')
            print(f' Train loss: {train_loss}')

            # Write tensorboard log

            writer.add_scalar(f'Loss/train_{idx}', train_loss, epoch)
            # writer.add_scalar(f'Accuracy/train_{idx}', np.random.random(), n_iter)

            # Validation
            model.eval()

            val_batch_loss = 0
            with torch.no_grad():
                result = []
                ground = []
                for batch_idx, sample in enumerate(val_loader):
                    image = sample['image'].to(device).float()
                    labels = sample['labels'].to(device).float()
                    labels = labels.unsqueeze(1)

                    out = model(image)

                    loss = criterion(out, labels)

                    val_batch_loss += loss.item()
                    out = out.cpu().detach().numpy()
                    label = labels.cpu().detach().numpy()
                    result.append(out)
                    ground.append(label)

            val_batch_loss /= len(val_loader)
            # val_batch_loss = val_batch_loss / len(val_loader)
            # import pdb;pdb.set_trace()

            result = np.array(result)
            result = np.squeeze(result, axis=0)
            result = np.squeeze(result, axis=-1)
            result = [1 if i > 0.5 else 0 for i in result]

            ground = np.array(ground)
            ground = np.squeeze(ground, axis=0)

            a = 0
            for i in range(0, 40):
                if result[i] == ground[i]:
                    a += 1
            acc = a/40
            print('Eval acc of test object number {}: {}'.format(ignored_value, acc))

            # Write tensorboard log
            writer.add_scalar(f'Loss/val_{idx}', val_batch_loss, epoch)
            writer.add_scalar(f'Accuracy/val_{idx}', acc, epoch)

            # Save best
            if (is_best < acc):
                is_best = acc
                save_checkpoint({'epoch': epoch + 1,
                                    'state_dict': model.state_dict(),
                                    'Loss': batch_loss,
                                    'optimizer': optimizer.state_dict()},
                                is_best=is_best,
                                prefix="acc_eval_object={}_acc={}_e={}".format(
                                    ignored_value, acc, epoch),
                                dir_to_save=dir_to_save)

            # import pdb;pdb.set_trace()

            elapsed = (time.time() - start_time) / 60
            print(f'Time elapsed: {elapsed:.2f} min')

        print("Best_acc_eval of testing object {}: {} [{}/{}]".format(
            ignored_value, is_best, epoch, num_epochs))
        list_acc.append(acc)
        list_best.append(is_best)

        elapsed = (time.time() - start_time) / 60
        print(f'Total Training Time: {elapsed:.2f} min')
        # torch.save(model.state_dict(), f'./checkpoints/spec_model.pth')
        del model
        torch.cuda.empty_cache()

    print("List of best acc each object: {}".format(list_best))
    sum_best = np.mean(list_best)
    print("List of acc - 32 objects {}: {}".format(idx, list_acc))
    sum_acc = np.mean(list_acc)
    print("Avg Acc of 32 objects: {}".format(sum_acc))

# =======================================================

# get_current_time


def save_checkpoint(state, is_best, prefix, dir_to_save):
    if is_best:
        filename = f'{dir_to_save}/{prefix}_checkpoint.pth.tar'
        torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(
    #         filename, './checkpoints/%s_model_best.pth.tar' % prefix)


# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

#
# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


if __name__ == '__main__':
    main()
