#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# params = parameters
# trainMode = training mode (1) or not (0)
# testMode = testing mode (1) or not (0)
# netNdx = network of choice ('tNet', 'rbUnet','Unet','meshNet','denseNet')
# testSetNdx = test group 1 or 2 from selectTrainAndTestSubjects_gh
# pcUsedNdx = PCA used (1) or not (0)
# tpUsedNdx = time sampling points
# deepReduction = deep reduction (1) or not (0)
# selectedEpoch = [num] for testing stage
# fileNumModel = part of address to folder that will contain trained model per defined epoch

# detectSeg3DkerasDR_gh
# train detection model (trainMode=0; testMode=1)
# test detection model (trainMode=1; testMode=0)

import detectSeg3DkerasDR_ha_use
#for netNdx in ['tNet', 'Unet','rbUnet','meshNet','denseNet']:
for netNdx in ['rbUnet']: # network of choice
    for testSetNdx in [1,2]:
        for pcUsedNdx in [1]: #PCA used (1) or not (0)
            for tpUsedNdx in [50]:#[50,100]:            
                params={};params['networkToUse']=netNdx; #'tNet', 'rbUnet','Unet','meshNet','denseNet'
                params['TestSetNum']=testSetNdx;params['fileNumModel']='';params['selectedEpoch']='';
                params['PcUsed']=pcUsedNdx;params['tDim']=tpUsedNdx;params['tpUsed']=tpUsedNdx;params['deepReduction']=0;
                trainMode=1;testMode=0;
                performanceMeasures,avgPerf=detectSeg3DkerasDR_gh.detectSeg3DkerasDR(trainMode,testMode,params);


# detectSeg3DkerasDR_gh
# test detection model
"""
######## test single
import detectSeg3DkerasDR_gh
params={};params['networkToUse']='rbUnet'; # 'tNet', 'Unet','rbUnet','meshNet','denseNet
params['TestSetNum']=1;params['fileNumModel']=500;params['selectedEpoch']=46000;
params['PcUsed']=0;params['tDim']=50;params['tpUsed']=50;params['deepReduction']=0;
trainMode=0;testMode=1;
performanceMeasures,avgPerf = detectSeg3DkerasDR_gh.detectSeg3DkerasDR(trainMode,testMode,params);
"""

"""
# croppedSeg3DkerasDR_gh
# train segmentation model (trainMode=0; testMode=1)
# test segmentation model (trainMode=1; testMode=0)

import croppedSeg3DkerasDR_gh
for netNdx in ['tNet']: #'Unet','denseNet','meshNet','rbUnet':
    for testSetNdx in [1,2]:
        for pcUsedNdx in [1]: #[0,1,2]:
            for tpUsedNdx in [50]:#[50,100]:            
                params={};params['networkToUse']=netNdx;
                params['TestSetNum']=testSetNdx;params['fileNumModel']='';params['selectedEpoch']='920';
                params['PcUsed']=pcUsedNdx;params['tDim']=tpUsedNdx;params['tpUsed']=tpUsedNdx;params['deepReduction']=0;
                trainMode=1;testMode=0;
                performanceMeasures,avgPerf=croppedSeg3DkerasDR_gh.croppedSeg3DkerasDR(trainMode,testMode,params);
"""


# croppedSeg3DkerasDR_gh
# test segmentation model
"""
######## test single
import croppedSeg3DkerasDR_ha_use
params={};params['networkToUse']='tNet'; #'rbUnet','Unet','meshNet','denseNet'
params['TestSetNum']=1;params['fileNumModel']=500;params['selectedEpoch']=80000;
params['PcUsed']=1;params['tDim']=50;params['tpUsed']=50;params['deepReduction']=0;
trainMode=0;testMode=1;
performanceMeasuresX,volumEstimError,performanceMeasures,avgPerf = croppedSeg3DkerasDR_gh.croppedSeg3DkerasDR(trainMode,testMode,params);
"""
