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
    list_acc = []
    for idx in range(len(deapfiles)):
        lr = 1e-4
        BATCH_SIZE = 32
        num_epochs = 1
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


        train_dataset = Dataloader_img(deaptrain_genarray)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  num_workers=8,
                                  shuffle=True)

        val_dataset = Dataloader_img(deapval_genarray)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=40,
                                num_workers=8,
                                shuffle=False)

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
            print('Current learning rate: ', optimizer.param_groups[0]['lr'])
            model.train()
            cost_list = 0
            s_time = time.time()

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

                cost_list += loss.item()
            e_time = time.time()-s_time
            print('-T: {:.1f}s '.format(e_time), end='')
            print(f' Cost: {cost_list / (batch_idx + 1)}')

            model.eval()
            with torch.no_grad():
                result = []
                ground = []
                for batch_idx, sample in enumerate(val_loader):
                    image = sample['image'].to(device).float()
                    labels = sample['labels'].to(device).float()
                    out = model(image)

                    out = out.cpu().detach().numpy()
                    label = labels.cpu().detach().numpy()
                    result.append(out)
                    ground.append(label)

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
            acc = a/40.0
             

            list_acc.append(acc)

            # import pdb;pdb.set_trace()

            elapsed = (time.time() - start_time) / 60
            print(f'Time elapsed: {elapsed:.2f} min')

        elapsed = (time.time() - start_time) / 60
        print(f'Total Training Time: {elapsed:.2f} min')
        # torch.save(model.state_dict(), f'./checkpoints/spec_model.pth')
        del model
        torch.cuda.empty_cache()

    print("List of acc: {}".format(list_acc))
    sum_acc = np.mean(list_acc)
    print("Avg Acc: {}".format(sum_acc))


if __name__ == '__main__':
    main()
