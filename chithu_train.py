from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda, Input
from keras.models import Model
from keras.activations import sigmoid
from keras import backend as K
import tensorflow as tf
import os
from os.path import isfile, isdir, join
import keras
from keras import applications as ka

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

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import SGD, Adam, Adamax
from keras.losses import binary_crossentropy, Huber
import keras.backend as K
from keras.metrics import Precision

from classification_models.classification_models.models.senet import SEResNeXt50
from classification_models.classification_models.models.resnext import ResNeXt50
from classification_models.classification_models.models.resnet import ResNet50, ResNet34, ResNet18

import math
from albumentations import (Compose, HorizontalFlip, VerticalFlip, ShiftScaleRotate, Rotate, ElasticTransform,
                            GridDistortion, OpticalDistortion, RandomRotate90)
import pickle
from scipy.stats import pearsonr, spearmanr


#-------------------------------------------DATA-------------------------------------------------------#
current_path = os.getcwd()
DEAP_folder = '/home/tp/EEGdataset/DEAP/data_image_pearson'
deapfiles = glob.glob(join(DEAP_folder,'*.png'))
deapfiles = sorted(deapfiles)
print(len(deapfiles))

#---------------------------------------DATA LOADER-------------------------------------------------------#
def get_pearson_corrmap(path):
    img = cv2.imread(path,0)
    img = np.float32(img/255.)
    # img = np.expand_dims(img,axis=-1)
    path = path.split("/")[-1]
    name_valence = path.split("_")[2]
    name_arousal = path.split("_")[3]
    label = [float(name_valence), float(name_arousal)]

    return img, label

class DataGenerator_deap(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_IDs, batch_size=2, shuffle=True):
        # Initialization
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y_combined = self.__data_generation(list_IDs_temp)

        return X, y_combined

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, 32, 32))
        y = np.empty((self.batch_size,1))
        z = np.empty((self.batch_size,1))

        # Generate data
        for i in range(len(list_IDs_temp)):
            # print(list_IDs_temp[i])
            name_subject = list_IDs_temp[i][0]+1
            name_trial = list_IDs_temp[i][1]+1
            findname = str(name_subject)+"_"+str(name_trial)+"_"
            # print(findname)
            filenames = [x.replace((DEAP_folder+'/'),'') for x in deapfiles]
            filepath = [n for n in filenames if n.startswith(findname)]
            # print(filepath)
            filepath = join(DEAP_folder,filepath[0])
            # print(filepath)
            

            corr_mat, label = get_pearson_corrmap(filepath)
            # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))
            # ax1.imshow(corr_mat)
            # ax2.imshow(corr_mat)
            # plt.show()
            # print(label)
            valence = 1 if label[0] > 5. else 0
            arousal = 1 if label[1] > 5. else 0
            # print(valence, arousal)
            X[i,] = corr_mat
            y[i,] = valence
            z[i,] = arousal 

        return X, [y,z]
#-------------------------------------------MODEL-------------------------------------------------------#

# BATCH_SIZE = 40
# EPOCHS = 200

# for ig in range(0,32):
#     ignored_value = ig
#     deap_no_trial = 40
#     deap_no_channels = 32
#     deap_no_subject = 32
#     deapval_subject_array = [ignored_value]
#     deap_trial_array = np.arange(0,deap_no_trial)
#     deaptrain_subject_array = np.arange(0,deap_no_subject)
#     deaptrain_subject_array = np.delete(deaptrain_subject_array,ignored_value)

#     deaptrain_genarray = np.array([[i,j] for i in deaptrain_subject_array for j in deap_trial_array])
#     deapval_genarray = np.array([[i,j] for i in deapval_subject_array for j in deap_trial_array])
    


#     train_generator = DataGenerator_deap(list_IDs=deaptrain_genarray,batch_size = BATCH_SIZE,shuffle=True)
#     val_generator = DataGenerator_deap(list_IDs=deapval_genarray,batch_size = BATCH_SIZE,shuffle=False)


#     model_resnet = ResNet18(input_shape=(32,32,3),weights='imagenet',include_top=False)
#     x_in = Input(shape=(32,32,1))
#     x = Conv2D(3,(3,3),padding='same',name='transform')(x_in)
#     x = model_resnet(x)

#     x1 = GlobalAveragePooling2D(name='pool_valence')(x)
#     x1 = Dense(1,name='fc_valence')(x1)
#     x1 = Activation('sigmoid',name='valence')(x1)

#     x2 = GlobalAveragePooling2D(name='pool_arousal')(x)
#     x2 = Dense(1,name='fc_arousal')(x2)
#     x2 = Activation('sigmoid',name='arousal')(x2)

#     model_final = Model(inputs=x_in,outputs=[x1,x2],name='eegemo')

#     model_final.compile(optimizer=Adam(learning_rate=5e-4),
#                         loss={"valence":'binary_crossentropy',
#                             "arousal":'binary_crossentropy'},
#                         metrics={"valence":'acc',
#                                 "arousal":'acc'})

#     mcp_savebest1 = ModelCheckpoint(join('/home/tp/EEGdataset/DEAP', 'weights','best_valence_ignore_'+str(ig+1)+'.h5'), 
#                                                             save_best_only=True,verbose=1, monitor='val_valence_acc', mode='max')
#     mcp_savebest2 = ModelCheckpoint(join('/home/tp/EEGdataset/DEAP', 'weights','best_arousal_ignore_'+str(ig+1)+'.h5'),
#                                                             save_best_only=True,verbose=1, monitor='val_arousal_acc', mode='max')

#     train = model_final.fit(train_generator,steps_per_epoch = len(deaptrain_genarray)//BATCH_SIZE,
#                                 epochs=EPOCHS,verbose=1,callbacks=[mcp_savebest1,mcp_savebest2],
#                                 validation_data=val_generator,
#                                 validation_steps = len(deapval_genarray)//BATCH_SIZE)


#-------------------------------------------EVALUATE-------------------------------------------------------#
# valence
savepath = '/home/tp/EEGdataset/DEAP/weights'

all_vala = 0
all_arou = 0
for ig in range(0,32):
    ignored_value = ig
    deap_no_trial = 40
    deap_no_channels = 32
    deap_no_subject = 32
    deapval_subject_array = [ignored_value]
    deap_trial_array = np.arange(0,deap_no_trial)

    deapval_genarray = np.array([[i,j] for i in deapval_subject_array for j in deap_trial_array])



    model_resnet = ResNet18(input_shape=(32,32,3),weights=None,include_top=False)
    x_in = Input(shape=(32,32,1))
    x = Conv2D(3,(3,3),padding='same',name='transform')(x_in)
    x = model_resnet(x)

    x1 = GlobalAveragePooling2D(name='pool_valence')(x)
    x1 = Dense(1,name='fc_valence')(x1)
    x1 = Activation('sigmoid',name='valence')(x1)

    x2 = GlobalAveragePooling2D(name='pool_arousal')(x)
    x2 = Dense(1,name='fc_arousal')(x2)
    x2 = Activation('sigmoid',name='arousal')(x2)

    model_final = Model(inputs=x_in,outputs=[x1,x2],name='eegemo')
    model_final.load_weights(join(savepath,'best_arousal_ignore_'+str(ig+1)+'.h5'))



    name_subject = ignored_value+1
    total_vala = 0
    total_arou = 0
    for name_trial in tqdm(range(0,40),total=40):
        findname = str(name_subject)+"_"+str(name_trial+1)+"_"
        # print(findname)
        filenames = [x.replace((DEAP_folder+'/'),'') for x in deapfiles]
        filepath = [n for n in filenames if n.startswith(findname)]
        # print(filepath)
        filepath = join(DEAP_folder,filepath[0])
        # print(filepath)
        
        corr_mat, label = get_pearson_corrmap(filepath)
        corr_mat = np.expand_dims(corr_mat,axis=0)
        thres = 0.55

        vala, arou = model_final.predict(corr_mat)
        vala = 1 if vala[0][0] > thres else 0
        arou = 1 if arou[0][0] > thres else 0

        real_vala = 1 if label[0] > 5. else 0
        real_arou = 1 if label[1] > 5. else 0
        
        if vala == real_vala:
            total_vala += 1
        if arou == real_arou:
            total_arou += 1
    
    print(total_vala/40,total_arou/40)
    all_arou = all_arou + total_arou/40
    all_vala = all_vala + total_vala/40

print("average arousal:",all_arou/32)
print("average valence:",all_vala/32)


