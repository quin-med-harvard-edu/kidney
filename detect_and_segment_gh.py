import sys
import os
from os import path
sys.path.insert(0, '/add-directory-path-where-needed/this-folder')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import nibabel as nib
import numpy as np
import pandas as pd

import funcs_gh
from sklearn.decomposition import PCA
import matplotlib
matplotlib.axes.Axes.plot
matplotlib.pyplot.plot
matplotlib.axes.Axes.legend
matplotlib.pyplot.legend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import scipy
from scipy.ndimage import zoom
from scipy import signal
from scipy.interpolate import interp1d
import pickle
from sklearn.metrics import precision_recall_curve
from skimage import morphology
from keras.models import Model,load_model,Sequential
from networks_gh import get_unet2, get_rbunet,get_meshNet,get_denseNet,calculatedPerfMeasures
from networks_gh import get_unetCnnRnn
from networks_gh import get_denseNet103, get_unet3
from selectTrainAndTestSubjects_gh import selectTrainAndTestSubjects


def singlePatientDetection(pName,subjectInfo, baseline, params):
    
    #TestSetNum=params['TestSetNum'];
    tDim = params['tDim'];
    #tpUsed = params['tpUsed'];
    deepRed = params['deepReduction'];
    PcUsed = params['PcUsed'];
    visEnabled = params['visualizeResults'];
    visSlider = params['visSlider']; 

    ##### access subject image data and temporal information
    vol4D00,_,_,_,_ = funcs_gh.readData4(pName,subjectInfo,reconMethod,0);    
    zDimOri = vol4D00.shape[2];
    
    timeRes0=subjectInfo['timeRes'][pName];
    if not isinstance(timeRes0, (int, float)): 
        timeRes=float(timeRes0.split("[")[1].split(",")[0]);
    else:
        timeRes=np.copy(timeRes0); 

    # start from baseline
    im = vol4D00[:,:,:,baseline:];
    medianFind = np.median(im);
    if medianFind == 0:
        medianFind = 1.0;
    im = im/medianFind;
    
    vol4D0 = np.copy(im);
    origTimeResVec=np.arange(0,vol4D0.shape[3]*timeRes,timeRes);
    resamTimeResVec=np.arange(0,50*6,6);   # resample to 50 data points
    
    if origTimeResVec[-1]<resamTimeResVec[-1]:
        print(pName)

    # interpolate image data to resamTimeResVec in time dimension
    f_out = interp1d(origTimeResVec,vol4D0, axis=3,bounds_error=False,fill_value=0);     
    vol4D0 = f_out(resamTimeResVec);

    # perform PCA (numPC)
    numPC = 5; #50
    pca = PCA(n_components=numPC);
    vol4Dvecs=np.reshape(vol4D0, (vol4D0.shape[0]*vol4D0.shape[1]*vol4D0.shape[2], vol4D0.shape[3]));
    PCs=pca.fit_transform(vol4Dvecs);
    vol4Dpcs=np.reshape(PCs, (vol4D0.shape[0],vol4D0.shape[1],vol4D0.shape[2], numPC));
        
    dpcs = np.copy(vol4Dpcs);
    dpcs=dpcs/dpcs.max();
    da = dpcs.T;

    # downsample imge data to 64 x 64 x 64
    dsFactor = 3.5; zDim = 64;
    im0 = zoom(da,(1,zDim/da.shape[1],1/dsFactor,1/dsFactor),order=0);
    
    sx = 0; xyDim = 64; 
    DataTest=np.zeros((1,zDim,xyDim,xyDim,tDim));
    DataTest[sx,:,:,:,:]=np.swapaxes(im0.T,0,2);
    
    #initialise detection model
    n_channels = tDim; n_classes = 3;
      
    #choose relevant detection model
    networkToUse = params['networkToUseDetect'];
    if networkToUse == 'rbUnet':
        model = get_rbunet(xyDim,zDim,n_channels,n_classes,deepRed,0);
    elif networkToUse =='Unet':
        model = get_unet3(xyDim,zDim,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'denseNet':
        model = get_denseNet(xyDim,zDim,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'tNet':
        model = get_denseNet103(xyDim,zDim,n_channels,n_classes,deepRed,0);    
    
    #address to detection model
    #fileNumModel='Net'+net+'_time'+str(tDim)+'_pcUsed'+str(PcUsed)+'_tpUsed'+str(tpUsed)+'_DR'+str(deepRed)+'_testSet'+str(TestSetNum);
    #address = "path-to-folder-containing-trained-detection-model(s)/"+fileNumModel+"/"    
    address = "path-to-folder-to-hold-detection-model(s)" + "/NetrbUnet_time5_pcUsed1_tpUsed50_DR0_testSet1/"
    #address = "path-to-folder-to-hold-detection-model(s)" + "/NetrbUnet_time5_pcUsed1_tpUsed50_DR0_testSet2/"
      
    #load detection model weights
    selectedEpoch=params['selectedEpochDetect'];
    model.load_weights(address+'detect3D_'+selectedEpoch+'.h5');
    
    #### perform prediction ####
    imgs_mask_test= model.predict(DataTest, verbose=1);
    
    multiHead = 0;
    if multiHead:
        labels_pred=np.argmax(imgs_mask_test[0], axis=4)
    else:
        labels_pred=np.argmax(imgs_mask_test, axis=4)
                
    # ensure all detected labels for right kidney are on the right half of x dimension
    labels_pred[:,:,:,0:int(xyDim/2)][labels_pred[:,:,:,0:int(xyDim/2)]==2]=1;
    labels_pred[:,:,:,int(xyDim/2):][labels_pred[:,:,:,int(xyDim/2):]==1]=2;            

    
    ##### generate bounding boxes (from coarse segmentation) #####
    
    si = 0;    
    
    left = labels_pred[si,:,:,:].T==2;
    left = left.astype(int);
    
    right = labels_pred[si,:,:,:].T==1;
    right = right.astype(int);

    ####### resample to original test image spatial dimensions
    
    xyDimOri = 224;
    
    KMR=zoom(right,(xyDimOri/np.size(right,0),xyDimOri/np.size(right,1),zDimOri/np.size(right,2)),order=0);
    KML=zoom(left,(xyDimOri/np.size(left,0),xyDimOri/np.size(left,1),zDimOri/np.size(left,2)),order=0);
    
    if np.sum(KMR) != 0:
        KMR=morphology.remove_small_objects(KMR.astype(bool), min_size=256,in_place=True).astype(int);
        KMR = KMR.astype(int);
    if np.sum(KML) != 0:
        KML=morphology.remove_small_objects(KML.astype(bool), min_size=256,in_place=True).astype(int);   
        KML = KML.astype(int);
        KML[KML>=1]=2;

    maskDetect = KMR + KML;
    
    ### generate prediction box
    boxDetect = [];

    aL=np.nonzero(KML==2);
    aR=np.nonzero(KMR==1);

    if aL[0].size!=0:
        boxL=np.array([int((min(aL[0])+max(aL[0]))/2),int((min(aL[1])+max(aL[1]))/2),int((min(aL[2])+max(aL[2]))/2),\
          (max(aL[0])-min(aL[0])),(max(aL[1])-min(aL[1])),(max(aL[2])-min(aL[2]))])
    else:
        boxL=np.zeros((6,));
        
    if aR[0].size!=0:
        boxR=np.array([int((min(aR[0])+max(aR[0]))/2),int((min(aR[1])+max(aR[1]))/2),int((min(aR[2])+max(aR[2]))/2),\
          (max(aR[0])-min(aR[0])),(max(aR[1])-min(aR[1])),(max(aR[2])-min(aR[2]))])
    else:
        boxR=np.zeros((6,));
    
    boxDetect=np.vstack([np.array(boxR),np.array(boxL)]);

    # identify whether right kidney exists
    # identify whether left kidney exists
    kidneyNone=np.nonzero(np.sum(boxDetect,axis=1)==0); #right/left
    if kidneyNone[0].size!=0:
        kidneyNone=np.nonzero(np.sum(boxDetect,axis=1)==0)[0][0]; #right/left
    
    # add extra margins to minimise impact of false-negative predictions
    KM = np.copy(maskDetect); KM[KM>1]=1;
    xSafeMagin=10;ySafeMagin=10;zSafeMagin=3;
    if boxDetect[0,2]+boxDetect[0,5]+3 >= KM.shape[2] or boxDetect[0,2]+boxDetect[0,5]-3 <0:
        boxDetect[:,[3,4,5]]=boxDetect[:,[3,4,5]]+[xSafeMagin,ySafeMagin,0];
    else:
        boxDetect[:,[3,4,5]]=boxDetect[:,[3,4,5]]+[xSafeMagin,ySafeMagin,zSafeMagin];
        
    # xSafeMagin=12;ySafeMagin=12;zSafeMagin=3;
    # boxDetect[:,[3,4,5]]=boxDetect[:,[3,4,5]]+[xSafeMagin,ySafeMagin,zSafeMagin];
    
    #### write masks to file ####
    predMaskR=np.zeros((1,xyDimOri,xyDimOri,zDimOri));
    predMaskL=np.zeros((1,xyDimOri,xyDimOri,zDimOri));
    
    sc = 0;
    predMaskR[sc,:,:,:]=KMR; 
    predMaskL[sc,:,:,:]=KML;    

    # if np.sum(predMaskR) != 0:
    #     predMaskL=morphology.remove_small_objects(predMaskL.astype(bool), min_size=256,in_place=True).astype(int);
    # if np.sum(predMaskL) != 0:
    #     predMaskR=morphology.remove_small_objects(predMaskR.astype(bool), min_size=256,in_place=True).astype(int);   
    
    Masks2Save={};

    predMaskR2=zoom(predMaskR[sc,:,:,:],(1,1,1),order=0);
    predMaskL2=zoom(predMaskL[sc,:,:,:],(1,1,1),order=0);
    
    Masks2Save['R']=np.copy(predMaskR2.astype(float));
    Masks2Save['L']=np.copy(predMaskL2.astype(float));

    # save predicted labels
    funcs_gh.writeMasksDetect(pName,subjectInfo,reconMethod,Masks2Save,1);
            
    return maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri

def singlePatientSegmentation(params, pName,subjectInfo, maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri):
    
    #TestSetNum=params['TestSetNum'];
    tDim = params['tDim'];
    #tpUsed = params['tpUsed'];
    deepRed = params['deepReduction'];
    PcUsed = params['PcUsed'];
    visEnabled = params['visualizeResults'];
    visSlider = params['visSlider']; 
    
    dx = 64; dy = 64; dz = 64;
    Box = np.copy(boxDetect);
    maskDetect[maskDetect>1]=1;
    
    exv = 0; # some test image files may require (+/- 5)
    if kidneyNone!=0:
        croppedData4DR_pcs=vol4Dpcs[Box[0,0]-int(Box[0,3]/2):Box[0,0]+int(Box[0,3]/2),\
                                Box[0,1]-int(Box[0,4]/2):Box[0,1]+int(Box[0,4]/2),\
                                Box[0,2]-int(Box[0,5]/2):Box[0,2]+int(Box[0,5]/2),:];
        croppedData4DR=vol4D0[Box[0,0]-int(Box[0,3]/2):Box[0,0]+int(Box[0,3]/2),\
                                Box[0,1]-int(Box[0,4]/2):Box[0,1]+int(Box[0,4]/2),\
                                Box[0,2]-int(Box[0,5]/2):Box[0,2]+int(Box[0,5]/2),:];
        
        croppedData4DR_pcs=zoom(croppedData4DR_pcs,(dx/np.size(croppedData4DR_pcs,0),dy/np.size(croppedData4DR_pcs,1),dz/np.size(croppedData4DR_pcs,2),1),order=0);
        croppedData4DR=zoom(croppedData4DR,(dx/np.size(croppedData4DR,0),dy/np.size(croppedData4DR,1),dz/np.size(croppedData4DR,2),1),order=0);
        
    if kidneyNone!=1:    
        croppedData4DL_pcs=vol4Dpcs[Box[1,0]-int(Box[1,3]/2)+exv:Box[1,0]+int(Box[1,3]/2)-exv,\
                                Box[1,1]-int(Box[1,4]/2)+exv:Box[1,1]+int(Box[1,4]/2)-exv,\
                                Box[1,2]-int(Box[1,5]/2)+exv:Box[1,2]+int(Box[1,5]/2)-exv,:]; 
            
        croppedData4DL=vol4D0[Box[1,0]-int(Box[1,3]/2)+exv:Box[1,0]+int(Box[1,3]/2)-exv,\
                                Box[1,1]-int(Box[1,4]/2)+exv:Box[1,1]+int(Box[1,4]/2)-exv,\
                                Box[1,2]-int(Box[1,5]/2)+exv:Box[1,2]+int(Box[1,5]/2)-exv,:];

        croppedData4DL_pcs=zoom(croppedData4DL_pcs,(dx/np.size(croppedData4DL_pcs,0),dy/np.size(croppedData4DL_pcs,1),dz/np.size(croppedData4DL_pcs,2),1),order=0);
        croppedData4DL=zoom(croppedData4DL,(dx/np.size(croppedData4DL,0),dy/np.size(croppedData4DL,1),dz/np.size(croppedData4DL,2),1),order=0);
        
    if kidneyNone==0:
        d=np.concatenate((croppedData4DL[np.newaxis,:,:,:,:],croppedData4DL[np.newaxis,:,:,:,:]),axis=0);
        dpcs=np.concatenate((croppedData4DL_pcs[np.newaxis,:,:,:,:],croppedData4DL_pcs[np.newaxis,:,:,:,:]),axis=0);
        
    elif kidneyNone==1:
        d=np.concatenate((croppedData4DR[np.newaxis,:,:,:,:],croppedData4DR[np.newaxis,:,:,:,:]),axis=0);
        dpcs=np.concatenate((croppedData4DR_pcs[np.newaxis,:,:,:,:],croppedData4DR_pcs[np.newaxis,:,:,:,:]),axis=0);
                  
    else:
        d=np.concatenate((croppedData4DR[np.newaxis,:,:,:,:],croppedData4DL[np.newaxis,:,:,:,:]),axis=0);
        dpcs=np.concatenate((croppedData4DR_pcs[np.newaxis,:,:,:,:],croppedData4DL_pcs[np.newaxis,:,:,:,:]),axis=0);
               

    d=d/d.max()
    dpcs=dpcs/dpcs.max();
    
    sc=0; n_channels = tDim;
    DataCroppedTest=np.zeros((2,dx,dy,dz,n_channels));
    DataCroppedTest[2*sc:2*sc+2,:,:,:,:]=dpcs;
    
    #address to segmentation model
    #fileNumModel='Net'+net+'_time'+str(tDim)+'_pcUsed'+str(PcUsed)+'_tpUsed'+str(tpUsed)+'_DR'+str(deepRed)+'_testSet'+str(TestSetNum);
    #address = "path-to-folder-containing-trained-segmentation-model(s)/"+fileNumModel+"/"
    address = "path-to-folder-to-hold-segmentation-model(s)" + "/NettNet_time5_pcUsed1_tpUsed50_DR0_testSet1/"
    #address = "path-to-folder-to-hold-segmentation-model(s)" + "/NettNet_time5_pcUsed1_tpUsed50_DR0_testSet2/"
    
    #choose relevant segmentation model
    n_classes = 2; # kidney, non-kidney
    networkToUse = params['networkToUseSegment'];
    if networkToUse == 'tNet':
        model = get_denseNet103(dx,dz,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'denseNet':
        model = get_denseNet(dx,dz,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'Unet':
        model = get_unet3(dx,dz,n_channels,n_classes,deepRed,0);
    if networkToUse == 'rbUnet':
        model = get_rbunet(dx,dz,n_channels,n_classes,deepRed,0);
    
    #load segmentation model weights
    selectedEpoch=params['selectedEpochSegment'];
    model.load_weights(address+'croppedSeg3D_'+selectedEpoch+'.h5');
    
    cropped_mask_test = model.predict(DataCroppedTest, verbose=1)
    if cropped_mask_test.min()<0:
        cropped_mask_test=abs(cropped_mask_test.min())+cropped_mask_test;
        
    imgs_mask_test2=np.copy(cropped_mask_test);
    imgs_mask_test2[:,:,:,:,0]=cropped_mask_test[:,:,:,:,0];
    imgs_mask_test2[:,:,:,:,1]=cropped_mask_test[:,:,:,:,1];
    labels_pred_2=np.argmax(imgs_mask_test2, axis=4);
    
    xyDim=224;
    predMaskR=np.zeros((1,xyDim,xyDim,zDimOri));
    predMaskL=np.zeros((1,xyDim,xyDim,zDimOri));
    
    if kidneyNone!=0:
        Rk=labels_pred_2[2*sc,:,:,:]
        croppedData4DR=signal.resample(Rk,Box[0,3], t=None, axis=0);
        croppedData4DR=signal.resample(croppedData4DR,Box[0,4], t=None, axis=1);
        croppedData4DR=signal.resample(croppedData4DR,Box[0,5], t=None, axis=2);
        croppedData4DR[croppedData4DR>0.5]=2;croppedData4DR[croppedData4DR<0.5]=0
        croppedData4DR[croppedData4DR==0]=1;croppedData4DR[croppedData4DR==2]=0  
        
        predMaskR[sc,int(Box[0,0]-Box[0,3]/2):int(Box[0,0]+Box[0,3]/2),\
                            int(Box[0,1]-Box[0,4]/2):int(Box[0,1]+Box[0,4]/2),\
                            int(Box[0,2]-Box[0,5]/2):int(Box[0,2]+Box[0,5]/2)]=croppedData4DR;
                        

        if kidneyNone!=1:     
            Lk=labels_pred_2[2*sc+1,:,:,:]
            croppedData4DL=signal.resample(Lk,Box[1,3], t=None, axis=0);
            croppedData4DL=signal.resample(croppedData4DL,Box[1,4], t=None, axis=1);
            croppedData4DL=signal.resample(croppedData4DL,Box[1,5], t=None, axis=2);
            croppedData4DL[croppedData4DL>0.5]=2; croppedData4DL[croppedData4DL<0.5]=0
            croppedData4DL[croppedData4DL==0]=1;croppedData4DL[croppedData4DL==2]=0    
            
            predMaskL[sc,int(Box[1,0]-Box[1,3]/2):int(Box[1,0]+Box[1,3]/2),\
                                int(Box[1,1]-Box[1,4]/2):int(Box[1,1]+Box[1,4]/2),\
                                int(Box[1,2]-Box[1,5]/2):int(Box[1,2]+Box[1,5]/2)]=croppedData4DL;    
        
        
        if np.sum(predMaskR) != 0:
            predMaskL=morphology.remove_small_objects(predMaskL.astype(bool), min_size=256,in_place=True).astype(int);
        if np.sum(predMaskL) != 0:
            predMaskR=morphology.remove_small_objects(predMaskR.astype(bool), min_size=256,in_place=True).astype(int);
    
        predMaskL2=np.copy(predMaskL);
        #predMaskL2[predMaskL2==1]=2;    
        
        Masks2Save={};
        
        predMaskR2=zoom(predMaskR[sc,:,:,:],(1,1,1),order=0);
        predMaskL2=zoom(predMaskL[sc,:,:,:],(1,1,1),order=0);
        maskSegment = predMaskR2 + predMaskL2;
        
        Masks2Save['R']=np.copy(predMaskR2.astype(float));
        Masks2Save['L']=np.copy(predMaskL2.astype(float));

        funcs_gh.writeMasks(pName,subjectInfo,reconMethod,Masks2Save,1);

    
    return maskSegment


# path to .xls sheet that contains time information for each test subject file (pName)
fileAddress='path-to-folder"+"/subjectDicomInfo_gh.xls';
subjectInfo=pd.read_excel(fileAddress, sheetname=0);
reconMethod='SCAN';

params={};
params['TestSetNum']=1; #0
params['tpUsed']= 50;
params['tDim']= params['tpUsed'];
params['PcUsed']= 1; #0
params['deepReduction']= 0;

params['networkToUseDetect']= 'rbUnet' #'denseNet'; #'tNet'; #'Unet'
params['networkToUseSegment']= 'tNet' #'denseNet'; #'Unet'; #'rbUnet'
params['selectedEpochDetect']='enter-number';
params['selectedEpochSegment']='enter-number';


if params['PcUsed']== 1:
    tDim=5;
    params['tDim']= tDim;
      
TestSetNum = 1; #2    
_,subjectNamesNormalTest,_,testKidCondTest,subjectNamesNormalTestBaselines=selectTrainAndTestSubjects(TestSetNum);
#subjectTrain,subjectTest,subjectTrainKidneyCondition,subjectTestKidneyCondition,subjectTestBaselines

# perform segmentation on a set of test subject files (subjectNamesNormalTest)
# or perform segmentation on an individual test subject file

for s in range(len(subjectNamesNormalTest)):
#for s in range(1):
    
    pName = subjectNamesNormalTest[s];
    baseline = subjectNamesNormalTestBaselines[s]
    
    #pName = 'image_file_name';
    #baseline = '8'; funcs_gh.baselineFinder2(pName,subjectInfo);
    
    print(pName)
    pathToFolderD = "path-to-folder-to-contain-detected-image-files"+"/detected/" + pName + '_seq1'
    if not os.path.exists(pathToFolderD):
        os.makedirs(pathToFolderD)
        
    pathToFolder = "path-to-folder-to-contain-segmented-image-files"+"/segmented/" + pName + '_seq1'
    if not os.path.exists(pathToFolder):
        os.makedirs(pathToFolder)

    # perform coarse segmentation (maskDetect) using detection model
    # and generate bounding box for each kidney (boxDetect)
    # and save coarse segmentation (detection) labels to a file
    maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri = singlePatientDetection(pName,subjectInfo, int(baseline),params);

    # perform segmentation (maskSegment) using segmentation model
    # and save to a file
    maskSegment = singlePatientSegmentation(params, pName,subjectInfo, maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri);
    
    
        
