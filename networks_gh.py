import numpy as np
import sys
from scipy.ndimage import zoom
import tensorflow as tf
#from os import listdir
#from os.path import isfile, join
#import pickle
#import os

#from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from numpy.core.umath_tests import inner1d

from keras.models import Model,load_model
from keras.layers import Dropout,Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, add, AtrousConvolution2D
from keras.layers import Reshape, Conv2D,MaxPooling2D,TimeDistributed,Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Activation, Dense, ZeroPadding3D
#from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import losses

from scipy.ndimage import zoom
from scipy import signal
from skimage import morphology
from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
import math
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage as ndi
from skimage import feature
import pickle,random

#import torch
#from geomloss import SamplesLoss
#use_cuda = torch.cuda.is_available()
#dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
#tf.enable_eager_execution()

def tensor_to_array(tensor1):
    return tensor1.numpy()

smooth = 0;
def dice_coef(y_true, y_pred):
    smooth=0;
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef_loss2(y_true, y_pred):
    return (tf.nn.softmax_cross_entropy_with_logits(logits =y_pred, labels=y_true) + 1e-8)
    #return (tf.contrib.losses.sparse_softmax_cross_entropy(y_pred, y_true, weight=1.0, scope=None) + 1e-8)

def dice_coef_loss3(y_true, y_pred,weights):
    error =tf.nn.softmax_cross_entropy_with_logits(logits =y_pred, labels=y_true)
    cost = tf.reduce_mean(error * weights)
    return (cost + 1e-8)

def tversky_coef(y_true, y_pred, alpha, beta, smooth=1):    

    y_true_f = K.flatten(y_true)
    y_true_f_r = K.flatten(1. - y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f_r = K.flatten(1. - y_pred)

    weights = 1.

    intersection = K.sum(y_pred_f * y_true_f *  weights)

    fp = K.sum(y_pred_f * y_true_f_r)
    fn = K.sum(y_pred_f_r * y_true_f *  weights)

    return (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth)

def tversky_loss(alpha, beta, weights=False):
    def tversky(y_true, y_pred):
        return -tversky_coef(y_true, y_pred, alpha, beta, weights)
    return tversky

tversky = tversky_loss(alpha=0.3, beta=0.7, weights=False)
    

def box_overlap(y_true, y_pred):
    y_trueBoxed = tf.identity(y_true)
    y_predBoxed = tf.identity(y_pred)
    uniq=tf.unique(y_true);
    for i in range(1,len(uniq)):
        lNdx=tf.where(y_true==i);
        y_trueBoxed[min(lNdx[0]):max(lNdx[0]),min(lNdx[1]):max(lNdx[1])]=i;

    uniq=tf.unique(y_pred);
    for i in range(1,len(uniq)):
        lNdx=tf.where(y_pred==i);
        y_predBoxed[min(lNdx[0]):max(lNdx[0]),min(lNdx[1]):max(lNdx[1])]=i;    

    return -dice_coef(y_trueBoxed, y_predBoxed)


def deepReduction(inputs,xyDim,zDim,n_channels,n_classes):
    reshapedInput=Reshape((zDim*xyDim*xyDim, n_channels,1))(inputs);
    conv0 = Conv2D(1, (1,3),strides=(1, 1), activation='relu')(reshapedInput)
    pool01 = MaxPooling2D(pool_size=(1, 4),strides=(1,2))(conv0)
    inputs2=Reshape((zDim,xyDim,xyDim, 23))(pool01);
    inputs2 =BatchNormalization() (inputs2)
    return inputs2

def get_unet2(xyDim,zDim,n_channels,n_classes,deepRed,multiHead):
    inputs = Input((zDim,xyDim, xyDim, n_channels))
    
    if deepRed:
        inputs=deepReduction(inputs,xyDim,zDim,n_channels,n_classes);
        
    conv1 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv1)

    conv2 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv2)

    conv3 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv3)

    conv4 = Conv3D(xyDim, (3, 3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(xyDim, (3, 3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv4)

    conv5 = Conv3D(xyDim*2, (3, 3,3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(xyDim*2, (3, 3,3), activation='relu', padding='same')(conv5)

    up60=UpSampling3D((2, 2,2))(conv5)
    up6 = concatenate([up60, conv4], axis=4)
    conv6 = Conv3D(int(xyDim), (3, 3,3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(int(xyDim), (3, 3,3), activation='relu', padding='same')(conv6)

    up70=UpSampling3D((2, 2,2))(conv6)
    up7 = concatenate([up70, conv3], axis=4)
    conv7 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(conv7)

    up80=UpSampling3D((2, 2,2))(conv7)
    up8 = concatenate([up80, conv2], axis=4)
    conv8 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(conv8)

    up90=UpSampling3D((2, 2,2))(conv8)
    up9 = concatenate([up90, conv1], axis=4)
    conv9 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(conv9)

    BN =BatchNormalization() (conv9)
    segPred = Conv3D(n_classes, (1, 1,1),activation='linear',padding='valid',name='seg')(BN)
    if multiHead:
        conv10 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(BN)
        conv10 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(conv10)
        BN1 =BatchNormalization() (conv10)
        boxPred = Conv3D(n_classes, (1, 1,1),activation='linear',padding='valid',name='box')(BN1)    
        
        model = Model(inputs=[inputs], outputs=[segPred,boxPred])    
        model.compile(optimizer=Adam(lr=1e-3),  loss={'seg':dice_coef_loss2,'box':dice_coef_loss2},loss_weights=[1., 1.])
       
    else:
        model = Model(inputs=[inputs], outputs=[segPred])    
        model.compile(optimizer=Adam(lr=1e-3),  loss=dice_coef_loss2)
    return model


def get_unet3(xyDim,zDim,n_channels,n_classes,deepRed,multiHead):
    inputs = Input((zDim,xyDim, xyDim, n_channels))
    
    if deepRed:
        inputs2=deepReduction(inputs,xyDim,zDim,n_channels,n_classes);
        conv1 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(inputs2)
    else:
        conv1 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(inputs)
    
    conv1 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv1)

    conv2 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv2)

    conv3 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv3)

    conv4 = Conv3D(xyDim, (3, 3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(xyDim, (3, 3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv4)

    conv5 = Conv3D(xyDim*2, (3, 3,3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(xyDim*2, (3, 3,3), activation='relu', padding='same')(conv5)

    up60=UpSampling3D((2, 2,2))(conv5)
    up6 = concatenate([up60, conv4], axis=4)
    conv6 = Conv3D(int(xyDim), (3, 3,3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(int(xyDim), (3, 3,3), activation='relu', padding='same')(conv6)

    up70=UpSampling3D((2, 2,2))(conv6)
    up7 = concatenate([up70, conv3], axis=4)
    conv7 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(conv7)

    up80=UpSampling3D((2, 2,2))(conv7)
    up8 = concatenate([up80, conv2], axis=4)
    conv8 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(conv8)

    up90=UpSampling3D((2, 2,2))(conv8)
    up9 = concatenate([up90, conv1], axis=4)
    conv9 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(conv9)

    BN =BatchNormalization() (conv9)
    Pred = Conv3D(n_classes, (1, 1,1),activation='linear',padding='valid')(BN)
    
    model = Model(inputs=[inputs], outputs=[Pred])
    model.compile(optimizer=Adam(lr=1e-4),  loss=dice_coef_loss2)

    return model


def get_unetCnnRnn(xyDim,zDim,n_channels,n_classes,deepRed,multiHead):
    inputs = Input((zDim,xyDim, xyDim, n_channels))
    
    if deepRed:
        inputs2=deepReduction(inputs,xyDim,zDim,n_channels,n_classes);
        conv1 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(inputs2)
    else:
        conv1 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(inputs)
    
    conv1 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv1)

    conv2 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv2)

    conv3 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv3)

    conv4 = Conv3D(xyDim, (3, 3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(xyDim, (3, 3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv4)

    conv5 = Conv3D(xyDim*2, (3, 3,3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(xyDim*2, (3, 3,3), activation='relu', padding='same')(conv5)

    up60=UpSampling3D((2, 2,2))(conv5)
    up6 = concatenate([up60, conv4], axis=4)
    conv6 = Conv3D(int(xyDim), (3, 3,3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(int(xyDim), (3, 3,3), activation='relu', padding='same')(conv6)

    up70=UpSampling3D((2, 2,2))(conv6)
    up7 = concatenate([up70, conv3], axis=4)
    conv7 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(conv7)

    up80=UpSampling3D((2, 2,2))(conv7)
    up8 = concatenate([up80, conv2], axis=4)
    conv8 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(conv8)

    up90=UpSampling3D((2, 2,2))(conv8)
    up9 = concatenate([up90, conv1], axis=4)
    conv9 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(conv9)

    BN =BatchNormalization() (conv9)
    Pred = Conv3D(1, (1, 1,1),activation='linear',padding='valid')(BN)
    
    return Pred

def get_unet_testrnn(xyDim,zDim,n_channels,n_classes,deepRed,multiHead):

    mod = Sequential()
    inputs = Input((n_channels,zDim,xyDim, xyDim,1))
    mod.add(inputs)
    conv1 = TimeDistributed(Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same'))(inputs)
    mod.add(conv1)
    
    conv1 = TimeDistributed(Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same'))(conv1)
    pool1 = TimeDistributed(MaxPooling3D(pool_size=(2, 2,2),strides=2))(conv1)

    conv2 = TimeDistributed(Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same'))(pool1)
    conv2 = TimeDistributed(Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same'))(conv2)
    pool2 = TimeDistributed(MaxPooling3D(pool_size=(2, 2,2),strides=2))(conv2)

    conv3 = TimeDistributed(Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same'))(pool2)
    conv3 = TimeDistributed(Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same'))(conv3)
    pool3 = TimeDistributed(MaxPooling3D(pool_size=(2, 2,2),strides=2))(conv3)

    conv4 = TimeDistributed(Conv3D(xyDim, (3, 3,3), activation='relu', padding='same'))(pool3)
    conv4 = TimeDistributed(Conv3D(xyDim, (3, 3,3), activation='relu', padding='same'))(conv4)
    pool4 = TimeDistributed(MaxPooling3D(pool_size=(2, 2,2),strides=2))(conv4)

    conv5 = TimeDistributed(Conv3D(xyDim*2, (3, 3,3), activation='relu', padding='same'))(pool4)
    conv5 = TimeDistributed(Conv3D(xyDim*2, (3, 3,3), activation='relu', padding='same'))(conv5)

    up60=TimeDistributed(UpSampling3D((2, 2,2)))(conv5)
    up6 = concatenate([up60, conv4], axis=5)
    conv6 = TimeDistributed(Conv3D(int(xyDim), (3, 3,3), activation='relu', padding='same'))(up6)
    conv6 = TimeDistributed(Conv3D(int(xyDim), (3, 3,3), activation='relu', padding='same'))(conv6)

    up70=TimeDistributed(UpSampling3D((2, 2,2)))(conv6)
    up7 = concatenate([up70, conv3], axis=5)
    conv7 = TimeDistributed(Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same'))(up7)
    conv7 = TimeDistributed(Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same'))(conv7)

    up80=TimeDistributed(UpSampling3D((2, 2,2)))(conv7)
    up8 = concatenate([up80, conv2], axis=5)
    conv8 = TimeDistributed(Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same'))(up8)
    conv8 = TimeDistributed(Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same'))(conv8)

    up90=TimeDistributed(UpSampling3D((2, 2,2)))(conv8)
    up9 = concatenate([up90, conv1], axis=5)
    conv9 = TimeDistributed(Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same'))(up9)
    conv9 = TimeDistributed(Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same'))(conv9)

    BN =TimeDistributed(BatchNormalization()) (conv9)
    Pred = TimeDistributed(Conv3D(n_classes, (1, 1,1),activation='linear',padding='valid'))(BN)
    mod.add(LSTM(output_dim=64, return_sequences=True))
    
    model = Model(inputs=[inputs], outputs=[Pred])
    model.compile(optimizer=Adam(lr=1e-4),  loss=dice_coef_loss2)
    return model


def get_unet_l(xyDim,zDim,n_channels,n_classes,deepRed):
    inputs = Input((zDim,xyDim, xyDim, n_channels))
    
    if deepRed:
        inputs=deepReduction(inputs,xyDim,zDim,n_channels,n_classes);
        
    conv1 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv1)
    pool1 = Dropout(0.2)(pool1)

    conv2 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv2)
    pool2 = Dropout(0.2)(pool2)

    conv3 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv3)
    pool3 = Dropout(0.2)(pool3)

    conv4 = Conv3D(xyDim, (3, 3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(xyDim, (3, 3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv4)
    pool4 = Dropout(0.2)(pool4)
    
    conv5 = Conv3D(xyDim*2, (3, 3,3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(xyDim*2, (3, 3,3), activation='relu', padding='same')(conv5)

    up60=UpSampling3D((2, 2,2))(conv5)
    up6 = concatenate([up60, conv4], axis=4)
    conv6 = Conv3D(int(xyDim), (3, 3,3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(int(xyDim), (3, 3,3), activation='relu', padding='same')(conv6)

    up70=UpSampling3D((2, 2,2))(conv6)
    up7 = concatenate([up70, conv3], axis=4)
    conv7 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(int(xyDim/2), (3, 3,3), activation='relu', padding='same')(conv7)

    up80=UpSampling3D((2, 2,2))(conv7)
    up8 = concatenate([up80, conv2], axis=4)
    conv8 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(int(xyDim/4), (3, 3,3), activation='relu', padding='same')(conv8)

    up90=UpSampling3D((2, 2,2))(conv8)
    up9 = concatenate([up90, conv1], axis=4)
    conv9 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(conv9)

    BN =BatchNormalization() (conv9)
    pred = Conv3D(n_classes, (1, 1,1),activation='linear',padding='valid')(BN)

    model = Model(inputs=[inputs], outputs=[pred])
    model.compile(optimizer=Adam(lr=1e-3,decay=0.001),  loss=dice_coef_loss2)
    
    return model

def get_rbunet(xyDim,zDim,n_channels,n_classes,deepRed,multiHead):
    inputs = Input((zDim,xyDim, xyDim, n_channels))
    
    if deepRed:
        inputs2=deepReduction(inputs,xyDim,zDim,n_channels,n_classes);
        conv1 = Conv3D(n_channels, (3, 3,3), activation='relu', padding='same')(inputs2)
    else:
        conv1 = Conv3D(n_channels, (3, 3,3), activation='relu', padding='same')(inputs)
    
    conv1 = Conv3D(n_channels, (3, 3,3), activation='relu', padding='same')(conv1)
    conv1 = add([conv1, inputs]);
    pool1 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv1)
    pool1 = Dropout(0.2)(pool1)

    conv2 = Conv3D(n_channels, (3, 3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(n_channels, (3, 3,3), activation='relu', padding='same')(conv2)
    conv2 = add([conv2, pool1]);
    pool2 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv2)
    pool2 = Dropout(0.2)(pool2)

    conv3 = Conv3D(n_channels, (3, 3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(n_channels, (3, 3,3), activation='relu', padding='same')(conv3)
    conv3 = add([conv3, pool2]);
    pool3 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv3)
    pool3 = Dropout(0.2)(pool3)

    conv4 = Conv3D(n_channels, (3, 3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(n_channels, (3, 3,3), activation='relu', padding='same')(conv4)
    conv4 = add([conv4, pool3]);
    pool4 = MaxPooling3D(pool_size=(2, 2,2),strides=2)(conv4)
    pool4 = Dropout(0.2)(pool4)
    
    conv5 = Conv3D(n_channels, (3, 3,3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv3D(n_channels, (3, 3,3), activation='relu', padding='same')(conv5)
    conv5 = add([conv5, pool4]);

    up60=UpSampling3D((2, 2,2))(conv5)
    up6 = concatenate([up60, conv4], axis=4)
    conv6 = Conv3D(n_channels*2, (3, 3,3), activation='relu', padding='same')(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv3D(n_channels*2, (3, 3,3), activation='relu', padding='same')(conv6)
    conv6 = add([conv6, up6]);

    up70=UpSampling3D((2, 2,2))(conv6)
    up7 = concatenate([up70, conv3], axis=4)
    conv7 = Conv3D(n_channels*3, (3, 3,3), activation='relu', padding='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv3D(n_channels*3, (3, 3,3), activation='relu', padding='same')(conv7)
    conv7 = add([conv7, up7]);

    up80=UpSampling3D((2, 2,2))(conv7)
    up8 = concatenate([up80, conv2], axis=4)
    conv8 = Conv3D(n_channels*4, (3, 3,3), activation='relu', padding='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv3D(n_channels*4, (3, 3,3), activation='relu', padding='same')(conv8)
    conv8 = add([conv8, up8]);

    up90=UpSampling3D((2, 2,2))(conv8)
    up9 = concatenate([up90, conv1], axis=4)
    conv9 = Conv3D(n_channels*5, (3, 3,3), activation='relu', padding='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv3D(n_channels*5, (3, 3,3), activation='relu', padding='same')(conv9)
    conv9 = add([conv9, up9]);

    BN =BatchNormalization() (conv9)
    
    segPred = Conv3D(n_classes, (1, 1,1),activation='linear',padding='valid',name='seg')(BN)
    if multiHead:
        conv10 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(BN)
        conv10 = Conv3D(int(xyDim/8), (3, 3,3), activation='relu', padding='same')(conv10)
        BN1 =BatchNormalization() (conv10)
        boxPred = Conv3D(n_classes, (1, 1,1),activation='linear',padding='valid',name='box')(BN1)    
        
        model = Model(inputs=[inputs], outputs=[segPred,boxPred])            
        model.compile(optimizer=Adam(lr=1e-3,decay=0.001),  loss={'seg':dice_coef_loss2,'box':dice_coef_loss2},loss_weights=[1., 1.])
       
    else:
        model = Model(inputs=[inputs], outputs=[segPred])    
        model.compile(optimizer=Adam(lr=1e-3,decay=0.001),  loss=dice_coef_loss2)
        
    return model

def get_meshNet(xyDim,zDim,n_channels,n_classes,deepRed,multiHead):

    #-- number of layers
    n_layers = 8
    #    xyDim=64;zDim=64;
    #-- input 4th dimension
    inputt = [n_channels, 21, 21, 21, 21, 21, 21, 21]
    #-- output 4th dimension
    output = [21, 21, 21, 21, 21, 21, 21, n_classes]
    #-- kernel size for layers from 1 to 8
    
    kZ = [3, 3, 3, 3, 3, 3, 3, 1];kY = kZ;kX = kZ;    
    #-- default convolution step
    #    dZ = 1;dY = dZ;dX = dZ
    
    #-- default padding
    padZ = [1, 1, 2, 4, 8, 16, 1, 0];padY = padZ;padX = padZ
    
    #-- dilation value for layers from 1 to 8
    dilZ = [1, 1, 2, 4, 8, 16, 1, 1];dilY = dilZ;dilX = dilZ
    
    #-- building net architecture
    #    local net = nn.Sequential()
    net = Sequential()
    
    #    model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
    #    net = Input((zDim,xyDim, xyDim, n_channels))
    for i in range(n_layers):
      if i != n_layers:
        net.add(Conv3D(output[i], (kZ[i], kY[i], kX[i]),input_shape=(zDim,xyDim, xyDim, inputt[i]),dilation_rate=dilZ[i]));
        net.add(ZeroPadding3D(padding=(padZ[i], padY[i], padX[i]), dim_ordering='default'));
        net.add(Activation('relu'));
        
        net.add(BatchNormalization())
      else:
        net.add(Conv3D(output[i], (kZ[i], kY[i], kX[i]),input_shape=(zDim,xyDim, xyDim, inputt[i]),dilation_rate=dilZ[i]));
        net.add(ZeroPadding3D(padding=(padZ[i], padY[i], padX[i]), dim_ordering='default',name='seg'))

    net.compile(optimizer=Adam(lr=1e-4),  loss=dice_coef_loss2)
    return net

def get_denseNet103(xyDim,zDim,n_channels,n_classes,deepRed,multiHead):

#    from models.dense_net_3d import DenseNet3D
#    model_params['sequence_length'] = train_params['sequence_length'];
#    model_params['crop_size']= train_params['crop_size'];
#    model = DenseNet3D(data_provider=data_provider, **model_params)
#    if args.train:
#       print("Data provider train videos: ", data_provider.train.num_examples)
#    

    import densenet_gh
    image_dim = (zDim,xyDim,xyDim,n_channels)

#    net = densenet_gh.DenseNetFCN103(input_shape=image_dim, nb_dense_block=5, growth_rate=12, nb_layers_per_block=4,
#                reduction=0.5, dropout_rate=0.2, weight_decay=1e-8, init_conv_filters=48,
#                classes=n_classes)
    
    net = densenet_gh.DenseNetFCN103(input_shape=image_dim, nb_dense_block=5, growth_rate=12, nb_layers_per_block=4,
                reduction=0.5, dropout_rate=0.2, weight_decay=1e-8, init_conv_filters=48,
                classes=n_classes)
    net.compile(optimizer=Adam(lr=1e-4),  loss=dice_coef_loss2)
    return net

def get_denseNet(xyDim,zDim,n_channels,n_classes,deepRed,multiHead):
    
#    from models.dense_net_3d import DenseNet3D
#    model_params['sequence_length'] = train_params['sequence_length'];
#    model_params['crop_size']= train_params['crop_size'];
#    model = DenseNet3D(data_provider=data_provider, **model_params)
#    if args.train:
#       print("Data provider train videos: ", data_provider.train.num_examples)
#    

    import densenet_gh
    image_dim = (zDim,xyDim,xyDim,n_channels)

    net = densenet_gh.DenseNetFCN(input_shape=image_dim, nb_dense_block=5, growth_rate=12, nb_layers_per_block=4,
                reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, init_conv_filters=24,
                classes=n_classes)
    net.compile(optimizer=Adam(lr=1e-4),  loss=dice_coef_loss2)
    
#    net,x = densenet_gh.DenseNetFCN(input_shape=image_dim, nb_dense_block=5, growth_rate=12, nb_layers_per_block=4,
#                reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, init_conv_filters=24,
#                classes=n_classes)
#    net.compile(optimizer=Adam(lr=1e-4),  loss=dice_coef_loss2x(x=x))
    
    return net

def get_denseNetT(xyDim,zDim,n_channels,n_classes,deepRed,multiHead):
    
#    from models.dense_net_3d import DenseNet3D
#    model_params['sequence_length'] = train_params['sequence_length'];
#    model_params['crop_size']= train_params['crop_size'];
#    model = DenseNet3D(data_provider=data_provider, **model_params)
#    if args.train:
#       print("Data provider train videos: ", data_provider.train.num_examples)
#    
#    model.train_all_epochs(train_params)

    import densenet_gh
    image_dim = (zDim,xyDim,xyDim,n_channels)

    net = densenet_gh.DenseNetT(input_shape=image_dim, nb_dense_block=5, growth_rate=12, nb_layers_per_block=4,
                reduction=0.5, dropout_rate=0.2, weight_decay=1e-8, init_conv_filters=48,
                classes=n_classes)
    net.compile(optimizer=Adam(lr=1e-4),  loss=dice_coef_loss2)
    return net

def generateAugmentation(image_batch,labels): 
    
    modifiedBatch = np.zeros(np.shape(image_batch));
    modifiedLabels = np.zeros(np.shape(labels));
    
    tform = AffineTransform(translation = (-20, 20))
    
    for jj in range(image_batch.shape[3]):
        
        for tt in range(image_batch.shape[4]):
            
            sliceOriR = image_batch[0,:,:,jj,tt];
            sliceModR = warp(sliceOriR, tform.inverse);
            
            sliceOriL = image_batch[1,:,:,jj,tt];
            sliceModL = warp(sliceOriL, tform.inverse);
            
            modifiedBatch[0,:,:,jj,tt] =  sliceModR;
            modifiedBatch[1,:,:,jj,tt] =  sliceModL;
            
        for tl in range(labels.shape[4]):
            
            labelOriR = labels[0,:,:,jj,tl];
            labelModR = warp(labelOriR, tform.inverse);
            
            labelOriL = labels[1,:,:,jj,tl];
            labelModL = warp(labelOriL, tform.inverse);
        
            modifiedLabels[0,:,:,jj,tl] =  labelModR;
            modifiedLabels[1,:,:,jj,tl] =  labelModL;
            
    return modifiedBatch,modifiedLabels
    #generateAugmentation(image_batch,labels): 
        
def augmentation(Data,Labels,scaleRange,steps):    
    scaleVec=np.arange(scaleRange[0],scaleRange[1],steps);
    
    DataAug=np.zeros((len(scaleVec)*Data.shape[0],Data.shape[1],Data.shape[2],Data.shape[3],Data.shape[4]));
    labAug=np.zeros((len(scaleVec)*Labels.shape[0],Labels.shape[1],Labels.shape[2],Labels.shape[3],Labels.shape[4]));

    for s in range(Data.shape[0]):
        DataS=Data[s,:,:,:,:];
        labelS=Labels[s,:,:,:,:];
        for r in range(len(scaleVec)):
            DataS1=zoom(DataS,(scaleVec[r],scaleVec[r],scaleVec[r],1),order=0);
            labelS1=zoom(labelS,(scaleVec[r],scaleVec[r],scaleVec[r],1),order=0);
            
            S1=np.size(DataS1,1);
            S=np.size(DataS,1);
            diffSize=S1-S;
            if diffSize<0:
                n=int(abs(diffSize)/2);
                DataAug[s*len(scaleVec)+r,n:n+S1,n:n+S1,n:n+S1,:]=DataS1;
                labAug[s*len(scaleVec)+r,n:n+S1,n:n+S1,n:n+S1,:]=labelS1;
            elif diffSize>=0:
                n=int(abs(diffSize)/2);
                DataAug[s*len(scaleVec)+r,:,:,:,:]=DataS1[n:n+S,n:n+S,n:n+S,:];  
                labAug[s*len(scaleVec)+r,:,:,:,:]=labelS1[n:n+S,n:n+S,n:n+S,:];  
           
    
    return DataAug,labAug
    #def augmentation(Data,Labels,scaleRange,steps): 
        
def augment_sample(image, label,labelb):
        scaleRange=[0.6,1.4]; steps=0.1;

        scaleVec=np.arange(scaleRange[0],scaleRange[1],steps);
        DataS=image;labelS=label;labelSb=labelb;

        r=random.sample(range(len(scaleVec)), 1)[0];   

        DataAug=np.zeros((DataS.shape[0],DataS.shape[1],DataS.shape[2],DataS.shape[3]));
        labAug=np.zeros((DataS.shape[0],labelS.shape[1],labelS.shape[2],labelS.shape[3]));
        labAugb=np.zeros((DataS.shape[0],labelSb.shape[1],labelSb.shape[2],labelSb.shape[3]));
            
        DataS1=zoom(DataS,(scaleVec[r],scaleVec[r],scaleVec[r],1),order=0);
        labelS1=zoom(labelS,(scaleVec[r],scaleVec[r],scaleVec[r],1),order=0);
        labelS1b=zoom(labelSb,(scaleVec[r],scaleVec[r],scaleVec[r],1),order=0); 
        
        S1=np.size(DataS1,1);
        S=np.size(DataS,1);
        
        diffSize=S1-S;
        if diffSize<0:
            n=int(abs(diffSize)/2);
            DataAug=np.pad(DataS1, ((n, abs(diffSize)-n), (n, abs(diffSize)-n),(n, abs(diffSize)-n),(0,0)), 'edge');
            labAug=np.pad(labelS1, ((n, abs(diffSize)-n), (n, abs(diffSize)-n),(n, abs(diffSize)-n),(0,0)), 'edge');
            labAugb=np.pad(labelS1b, ((n, abs(diffSize)-n), (n, abs(diffSize)-n),(n, abs(diffSize)-n),(0,0)), 'edge');
        elif diffSize>=0:
            n=int(abs(diffSize)/2);
            DataAug[:,:,:,:]=DataS1[n:n+S,n:n+S,n:n+S,:];  
            labAug[:,:,:,:]=labelS1[n:n+S,n:n+S,n:n+S,:];      
            labAugb[:,:,:,:]=labelS1b[n:n+S,n:n+S,n:n+S,:];      
            
        return(DataAug, labAug,labAugb)                       

def augment_sample_segment(image, label):
        scaleRange=[0.5,1.5]; steps=0.1;
        scaleVec=np.arange(scaleRange[0],scaleRange[1],steps);
        DataS=image;labelS=label;
        r=random.sample(range(len(scaleVec)), 1)[0];   

        DataAug=np.zeros((DataS.shape[0],DataS.shape[1],DataS.shape[2],DataS.shape[3]));
        labAug=np.zeros((DataS.shape[0],labelS.shape[1],labelS.shape[2],labelS.shape[3]));
            
        DataS1=zoom(DataS,(scaleVec[r],scaleVec[r],scaleVec[r],1),order=0);
        labelS1=zoom(labelS,(scaleVec[r],scaleVec[r],scaleVec[r],1),order=0);
        
        S1=np.size(DataS1,1);
        S=np.size(DataS,1);
        #print(S,S1)
        
        diffSize=S1-S;
        if diffSize<0:
            n=int(abs(diffSize)/2);
            DataAug=np.pad(DataS1, ((n, abs(diffSize)-n), (n, abs(diffSize)-n),(n, abs(diffSize)-n),(0,0)), 'edge');
            labAug=np.pad(labelS1, ((n, abs(diffSize)-n), (n, abs(diffSize)-n),(n, abs(diffSize)-n),(0,0)), 'edge');

        elif diffSize>=0:
            n=int(abs(diffSize)/2);
            DataAug[:,:,:,:]=DataS1[n:n+S,n:n+S,n:n+S,:];  
            labAug[:,:,:,:]=labelS1[n:n+S,n:n+S,n:n+S,:];         
            
        return(DataAug, labAug)                       
        
    
def calculatedPerfMeasures(y_true, y_pred):    

    y_true[y_true!=0]=1;y_pred[y_pred!=0]=1;
    from sklearn.metrics import precision_recall_fscore_support
    y_true_f = y_true.flatten();
    y_pred_f = y_pred.flatten()
    prec,rec,f1,ss=precision_recall_fscore_support(y_true_f, y_pred_f,pos_label=1,average='binary');
    vee=np.count_nonzero(y_true_f)-np.count_nonzero(y_pred_f);
    return [f1,prec,rec,vee,vee*1.25*1.25*3*0.001]
    
def IoU3D(boxPred0,boxTest0,labelsPred0):
    rbp=boxPred0[0:6];
    rbt=boxTest0[0:6];
    lbp=boxPred0[6:];
    lbt=boxTest0[6:];    
    
    intersRight=(min([rbt[0]+int(rbt[3]/2),rbp[0]+int(rbp[3]/2)])-max([rbt[0]-int(rbt[3]/2),rbp[0]-int(rbp[3]/2)]))*\
    (min([rbt[1]+int(rbt[4]/2),rbp[1]+int(rbp[4]/2)])-max([rbt[1]-int(rbt[4]/2),rbp[1]-int(rbp[4]/2)]))*\
    (min([rbt[2]+int(rbt[5]/2),rbp[2]+int(rbp[5]/2)])-max([rbt[2]-int(rbt[5]/2),rbp[2]-int(rbp[5]/2)]))
    
    unionRight=(rbt[3]*rbt[4]*rbt[5]+rbp[3]*rbp[4]*rbp[5])-intersRight;
    
    IoUright=intersRight/unionRight;

    intersLeft=(min([lbt[0]+int(lbt[3]/2),lbp[0]+int(lbp[3]/2)])-max([lbt[0]-int(lbt[3]/2),lbp[0]-int(lbp[3]/2)]))*\
    (min([lbt[1]+int(lbt[4]/2),lbp[1]+int(lbp[4]/2)])-max([lbt[1]-int(lbt[4]/2),lbp[1]-int(lbp[4]/2)]))*\
    (min([lbt[2]+int(lbt[5]/2),lbp[2]+int(lbp[5]/2)])-max([lbt[2]-int(lbt[5]/2),lbp[2]-int(lbp[5]/2)]))
    
    unionLeft=(lbt[3]*lbt[4]*lbt[5]+lbp[3]*lbp[4]*lbp[5])-intersLeft;
    
    IoUright=intersRight/unionRight;
    IoUleft=intersLeft/unionLeft; 
    averageIoU=np.mean([IoUright,IoUleft]);
    
    labelsPredRight=np.copy(labelsPred0);
    labelsPredRight[labelsPredRight!=1]=0;labelsPredRight[int(rbt[0]-(rbt[3]/2)):int(rbt[0]+(rbt[3]/2)),int(rbt[1]-int(rbt[4]/2)):int(rbt[1]+(rbt[4]/2)),int(rbt[2]-(rbt[5]/2)):int(rbt[2]+(rbt[5]/2))]=0;
    missedVoxelsRight=np.count_nonzero(labelsPredRight);

    labelsPredLeft=np.copy(labelsPred0);
    labelsPredLeft[labelsPredLeft!=2]=0;labelsPredLeft[int(lbt[0]-(lbt[3]/2)):int(lbt[0]+(lbt[3]/2)),int(lbt[1]-int(lbt[4]/2)):int(lbt[1]+(lbt[4]/2)),int(lbt[2]-(lbt[5]/2)):int(lbt[2]+(lbt[5]/2))]=0;
    missedVoxelsLeft=np.count_nonzero(labelsPredLeft);
    
    return averageIoU,[missedVoxelsRight,missedVoxelsLeft]    
    

def get_unetAAD(xDim,yDim,n_channels,n_classes):
    inputs = Input((xDim, yDim, n_channels))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(lr=1e-6,beta_1=0.5),  loss=losses.mean_squared_error)
    
    return model    
    
    
