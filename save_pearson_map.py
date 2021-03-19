import os
from os.path import isfile, isdir, join

import glob
from PIL import Image
import PIL.ImageOps
import numpy as np
import cv2
from numpy.random import randint, rand
from tqdm import tqdm
import multiprocessing
import time
from numpy import shape, ones, uint8, float32, reshape, ndarray
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

import pickle
from scipy.stats import pearsonr, spearmanr

#-------------------------------------------DATA-------------------------------------------------------#
current_path = os.getcwd()
DEAP_folder = '/home/tp/EEGdataset/DEAP'
SEED_folder = '/home/tp/EEGdataset/SEED/eeg-seed'

deapfiles = glob.glob(join(DEAP_folder,'data_preprocessed_python','*.dat'))
deapfiles = sorted(deapfiles)
print(len(deapfiles))

seedfiles = glob.glob(join(SEED_folder,'Preprocessed_EEG','*.mat'))
seedfiles = sorted(seedfiles)
print(len(seedfiles))


deap_no_trial = 40
deap_no_channels = 32
deap_no_subject = 32
deap_trial_array = np.arange(0,deap_no_trial)
deap_subject_array = np.arange(0,deap_no_subject)
# print(deap_trial_array)
# print(deap_subject_array)

deap_genarray = np.array([[i,j] for i in deap_subject_array for j in deap_trial_array])
# print(deap_genarray.shape)
# for i in range(0,deap_genarray.shape[0]):



for i in range(0,deap_no_subject):
    for j in range(0,deap_no_trial):
        file_name = deapfiles[i]
        x = open(file_name,'rb')
        x = pickle.load(x,encoding='latin1')

        no_channels = 32
        channel_patch = []
        for cha in range(0,no_channels):
            chdata = x['data'][j][cha]
            chdata = chdata[128*3:-1]
            channel_patch.append(chdata)

        corr_mat = np.zeros((32,32))
        for fx in range(0,no_channels):
            for fy in range(0,no_channels):
                pearson_corr,_ = pearsonr(channel_patch[fx],channel_patch[fy])
                corr_mat[fx,fy] = pearson_corr

        label = x['labels'][j]
        corr_mat = (corr_mat+1)/2
        corr_mat = corr_mat*255
        corr_mat = np.uint8(corr_mat)
        corr_mat = np.expand_dims(corr_mat,axis=-1)

        name_subject = str(i+1)
        name_trial = str(j+1)
        name_valence = str(label[0])
        name_arousal = str(label[1])

        save_name = name_subject + "_" + name_trial + "_" + name_valence + "_" + name_arousal + "_" + "pearson"
        cv2.imwrite(join(DEAP_folder,'data_image_pearson',save_name+'.png'),corr_mat)



name = "25_33_1.94_6.06_pearson.png"
name_valence = name.split("_")[2]
name_arousal = name.split("_")[3]
print(name_valence, name_arousal)
