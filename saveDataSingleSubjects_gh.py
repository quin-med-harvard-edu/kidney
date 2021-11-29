import sys
import os
from os import path
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import nibabel as nib
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/add-directory-path-where-needed/this-folder')

import funcs_gh
from scipy.interpolate import interp1d
import pickle
from sklearn.decomposition import PCA, KernelPCA
from scipy.ndimage import zoom
from sklearn.manifold import TSNE

import matplotlib
matplotlib.axes.Axes.plot
matplotlib.pyplot.plot
matplotlib.axes.Axes.legend
matplotlib.pyplot.legend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

print('saving3!')

# reference
# ['image_file_name',integer_value_of_baseline]

subjectNamesNormal0 =np.array([
    ['image_name001',5],             
    ['image_name002',8],
    ['image_name003',8],
    ['image_name004',10],
    ['image_name005',8], 
    ['image_name006',9],
    ['image_name007',10],             
    ['image_name008',4],
    ['image_name009',12],
    ['image_name010',9],
    ['image_name011',8], 
    ['image_name012',8]]);

subjectNamesNormal = list(subjectNamesNormal0[:,0]);


##### save single subject files of 4D image data for training detection model ####

if 1:
    
    numPC = 5; #50
    pca = PCA(n_components=numPC)
    
    dx=64;dy=64;dz=64;
    
    # path to .xls sheet that contains temporal information for each subject file (patientName)
    fileAddress='path-to-folder"+"/subjectDicomInfo_gh.xls';
    subjectInfo=pd.read_excel(fileAddress, sheetname=0);
    reconMethod='SCAN';
    
    for s in range(len(subjectNamesNormal)):
        Data4D={};
        Data4Ddpcs={};
        patientName=subjectNamesNormal[s];
        print(patientName)

        # extract image data (vol4D00), kidneys mask (KM), bounding box for kidneys (Box)
        vol4D00,KM,Box,_,_ = funcs_gh.readData4(patientName,subjectInfo,reconMethod,1);    
        numSlicesVol = vol4D00.shape[2];
        Box = Box.astype(int)

        # extract temporal information from subjectInfo
        timeRes0=subjectInfo['timeRes'][patientName];
        if not isinstance(timeRes0, (int, float)): 
            timeRes=float(timeRes0.split("[")[1].split(",")[0]);
        else:
            timeRes=np.copy(timeRes0); 

        # start from baseline 
        im=vol4D00[:,:,:,int(subjectNamesNormal0[s,1]):];

        medianFind = np.median(im);
        if medianFind == 0:
            medianFind = 1.0;
            
        #normalise to the median of intensities
        im=im/medianFind;
        
        vol4D0 = np.copy(im);
        origTimeResVec=np.arange(0,vol4D0.shape[3]*timeRes,timeRes);
        resamTimeResVec=np.arange(0,50*6,6);   # resample to 50 data points
        
        if origTimeResVec[-1]<resamTimeResVec[-1]:
            print(patientName)

        # interpolate image data to resamTimeResVec
        f_out = interp1d(origTimeResVec,vol4D0, axis=3,bounds_error=False,fill_value=0)     
        vol4D0 = f_out(resamTimeResVec);

        #resample to PCA (numPC)
        vol4Dvecs=np.reshape(vol4D0, (vol4D0.shape[0]*vol4D0.shape[1]*vol4D0.shape[2], vol4D0.shape[3]));
        PCs=pca.fit_transform(vol4Dvecs)
        vol4Dpcs=np.reshape(PCs, (vol4D0.shape[0],vol4D0.shape[1],vol4D0.shape[2], numPC))
    
        d = np.copy(vol4D0);
        d=d/d.max();
        
        dpcs = np.copy(vol4Dpcs);
        dpcs=dpcs/dpcs.max()
        #Box=Box/Box.max()
        
        Data4D[patientName+'D']=d;
        Data4D[patientName+'M']=KM;
        Data4D[patientName+'B']=Box;
     
        Data4Ddpcs[patientName+'D']=dpcs;
        Data4Ddpcs[patientName+'M']=KM;
        Data4Ddpcs[patientName+'B']=Box;
        
        pickle.dump(Data4Ddpcs, open("/path-to-folder-containing-images-for-detection-model/singleSubjectsV4pc_detect/"+patientName+".p", "wb" ))
        pickle.dump(Data4D, open("/path-to-folder-containing-images-for-detection-model/singleSubjectsV4_detect/"+patientName+".p", "wb" ))

 
##### save single subject files of 4D image data for training detection model ####


"""
##### save single subject files of 4D image data for training segmentation model ####

if 1:
    
    numPC=5; #5 #50
    pca = PCA(n_components=numPC)
    
    dx=64;dy=64;dz=64;
    
    for s in range(len(subjectNamesNormal)):
        
        croppedData4D={};
        croppedData4Ddpcs={};
        
        patientName=subjectNamesNormal[s];
        print(patientName)

        # path to .xls sheet that contains temporal information for each subject file (patientName)
        fileAddress='path-to-folder"+"/subjectDicomInfo_gh.xls';
        subjectInfo=pd.read_excel(fileAddress, sheetname=0);
        reconMethod='SCAN';
    
        # extract image data information
        vol4D00,KM,Box,_,_=funcs_gh.readData4(patientName,subjectInfo,reconMethod,1); 
        vol4D00 = np.nan_to_num(vol4D00);
        
        # start data from baseline
        vol4D01=vol4D00[:,:,:,int(subjectNamesNormal0[s,1]):];
        medianFind = np.median(vol4D01);
        if medianFind == 0:
            medianFind = 1.0;
            
        #normalise to the median of intensities
        vol4D0=vol4D01/medianFind;

        # extract temporal information from subjectInfo
        timeRes0=subjectInfo['timeRes'][patientName];
        if not isinstance(timeRes0, (int, float)):
            timeRes=float(timeRes0.split("[")[1].split(",")[0]);
        else:
            timeRes=np.copy(timeRes0); 
        origTimeResVec=np.arange(0,vol4D0.shape[3]*timeRes,timeRes);
        resamTimeResVec=np.arange(0,50*6,6); # resample to 50 data points
        
        if origTimeResVec[-1]<resamTimeResVec[-1]:
            print(patientName)

        # interpolate image data to resamTimeResVec
        f_out = interp1d(origTimeResVec,vol4D0, axis=3,bounds_error=False,fill_value=0)     
        vol4D0 = f_out(resamTimeResVec);
        
        Box=Box.astype(int)
        kidneyNone=np.nonzero(np.sum(Box,axis=1)==0); #right/left
        if kidneyNone[0].size!=0:
            kidneyNone=np.nonzero(np.sum(Box,axis=1)==0)[0][0]; #right/left

        # add extra margin to avoid skipping potential borderline true-positives
        KM[KM>1]=1;
        if Box[0,2]+Box[0,5]+3 >= KM.shape[2] or Box[0,2]+Box[0,5]-3 <0:
            Box[:,[3,4,5]]=Box[:,[3,4,5]]+[10,10,0];
        else:
            Box[:,[3,4,5]]=Box[:,[3,4,5]]+[10,10,3];

        #resample to PCA (numPC)
        vol4Dvecs=np.reshape(vol4D0, (vol4D0.shape[0]*vol4D0.shape[1]*vol4D0.shape[2], vol4D0.shape[3]));
        PCs=pca.fit_transform(vol4Dvecs)
        vol4Dpcs=np.reshape(PCs, (vol4D0.shape[0],vol4D0.shape[1],vol4D0.shape[2], numPC))
        
        exv = 0; #some subject files may require +/- 5
        if kidneyNone!=0:

            # crop right-kidney image data using Box[0,:]
            croppedData4DR_pcs=vol4Dpcs[Box[0,0]-int(Box[0,3]/2):Box[0,0]+int(Box[0,3]/2),\
                                    Box[0,1]-int(Box[0,4]/2):Box[0,1]+int(Box[0,4]/2),\
                                    Box[0,2]-int(Box[0,5]/2):Box[0,2]+int(Box[0,5]/2),:];
            croppedData4DR=vol4D0[Box[0,0]-int(Box[0,3]/2):Box[0,0]+int(Box[0,3]/2),\
                                    Box[0,1]-int(Box[0,4]/2):Box[0,1]+int(Box[0,4]/2),\
                                    Box[0,2]-int(Box[0,5]/2):Box[0,2]+int(Box[0,5]/2),:];
            KMR=KM[Box[0,0]-int(Box[0,3]/2):Box[0,0]+int(Box[0,3]/2),\
                                    Box[0,1]-int(Box[0,4]/2):Box[0,1]+int(Box[0,4]/2),\
                                    Box[0,2]-int(Box[0,5]/2):Box[0,2]+int(Box[0,5]/2)];  

            # interpolate data to 64 x 64 x 64
            croppedData4DR_pcs=zoom(croppedData4DR_pcs,(dx/np.size(croppedData4DR_pcs,0),dy/np.size(croppedData4DR_pcs,1),dz/np.size(croppedData4DR_pcs,2),1),order=0);
            croppedData4DR=zoom(croppedData4DR,(dx/np.size(croppedData4DR,0),dy/np.size(croppedData4DR,1),dz/np.size(croppedData4DR,2),1),order=0);    
            KMR=zoom(KMR,(dx/np.size(KMR,0),dy/np.size(KMR,1),dz/np.size(KMR,2)),order=0);
    
        if kidneyNone!=1:

            # crop left-kidney image data using Box [1,:]
            croppedData4DL_pcs=vol4Dpcs[Box[1,0]-int(Box[1,3]/2)+exv:Box[1,0]+int(Box[1,3]/2)-exv,\
                                    Box[1,1]-int(Box[1,4]/2)+exv:Box[1,1]+int(Box[1,4]/2)-exv,\
                                    Box[1,2]-int(Box[1,5]/2)+exv:Box[1,2]+int(Box[1,5]/2)-exv,:]; 
                
            croppedData4DL=vol4D0[Box[1,0]-int(Box[1,3]/2)+exv:Box[1,0]+int(Box[1,3]/2)-exv,\
                                    Box[1,1]-int(Box[1,4]/2)+exv:Box[1,1]+int(Box[1,4]/2)-exv,\
                                    Box[1,2]-int(Box[1,5]/2)+exv:Box[1,2]+int(Box[1,5]/2)-exv,:];  
                
            KML=KM[Box[1,0]-int(Box[1,3]/2)+exv:Box[1,0]+int(Box[1,3]/2)-exv,\
                                    Box[1,1]-int(Box[1,4]/2)+exv:Box[1,1]+int(Box[1,4]/2)-exv,\
                                    Box[1,2]-int(Box[1,5]/2)+exv:Box[1,2]+int(Box[1,5]/2)-exv];    
    
            # interpolate data to 64 x 64 x 64
            croppedData4DL_pcs=zoom(croppedData4DL_pcs,(dx/np.size(croppedData4DL_pcs,0),dy/np.size(croppedData4DL_pcs,1),dz/np.size(croppedData4DL_pcs,2),1),order=0);
            croppedData4DL=zoom(croppedData4DL,(dx/np.size(croppedData4DL,0),dy/np.size(croppedData4DL,1),dz/np.size(croppedData4DL,2),1),order=0);            
            KML=zoom(KML,(dx/np.size(KML,0),dy/np.size(KML,1),dz/np.size(KML,2)),order=0);

        if kidneyNone==0:
            d=np.concatenate((croppedData4DL[np.newaxis,:,:,:,:],croppedData4DL[np.newaxis,:,:,:,:]),axis=0);
            dpcs=np.concatenate((croppedData4DL_pcs[np.newaxis,:,:,:,:],croppedData4DL_pcs[np.newaxis,:,:,:,:]),axis=0);
            KM2=np.concatenate((KML[np.newaxis,:,:,:],KML[np.newaxis,:,:,:]),axis=0);   
        elif kidneyNone==1:
            d=np.concatenate((croppedData4DR[np.newaxis,:,:,:,:],croppedData4DR[np.newaxis,:,:,:,:]),axis=0);
            dpcs=np.concatenate((croppedData4DR_pcs[np.newaxis,:,:,:,:],croppedData4DR_pcs[np.newaxis,:,:,:,:]),axis=0);
            KM2=np.concatenate((KMR[np.newaxis,:,:,:],KMR[np.newaxis,:,:,:]),axis=0);          
        else:
            d=np.concatenate((croppedData4DR[np.newaxis,:,:,:,:],croppedData4DL[np.newaxis,:,:,:,:]),axis=0);
            dpcs=np.concatenate((croppedData4DR_pcs[np.newaxis,:,:,:,:],croppedData4DL_pcs[np.newaxis,:,:,:,:]),axis=0);
            KM2=np.concatenate((KMR[np.newaxis,:,:,:],KML[np.newaxis,:,:,:]),axis=0);        

        #d[d<0]=0;
        d=d/d.max()
        dpcs=dpcs/dpcs.max();
        Box=Box/Box.max();
     
        croppedData4Ddpcs[patientName+'D']=dpcs;
        croppedData4Ddpcs[patientName+'M']=KM2;
        croppedData4Ddpcs[patientName+'B']=Box;

        croppedData4D[patientName+'D']=d;
        croppedData4D[patientName+'M']=KM2;
        croppedData4D[patientName+'B']=Box;
        
        pickle.dump(croppedData4Ddpcs, open("/path-to-folder-containing-downsampled-images-for-segmentation-model/"+"singleSubjectsCroppedV4pc_segment/"+patientName+'_tp'+str(d.shape[4])+".p", "wb" ));
        pickle.dump(croppedData4D, open("/path-to-folder-containing-downsampled-images-for-segmentation-model/"+"singleSubjectsCroppedV4_segment/"+patientName+'_tp'+str(d.shape[4])+".p", "wb" ));


        ## plot sample images (2D) from cropped kidney images
        print('done ' + patientName)
        for si in range(KM2.shape[2]):

            fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(4, 4))
            axes = ax0, ax1, ax2, ax3
           
            uniqueX, countsX = np.unique(KMR[:,:,si], return_counts=True)
            ccX = len(countsX);
           
            ax0.imshow(KM2[0,:,:,si])
            ax0.set_title(str(ccX) + ' 0: ' + str(si))

            ax1.imshow(dpcs[0,:,:,si,0])
            ax1.set_title('0: ' + str(si))
           
            uniqueXR, countsXR = np.unique(KML[:,:,si], return_counts=True)
            ccXR = len(countsXR);
           
            ax2.imshow(KM2[1,:,:,si])
            #ax2.imshow(KML[:,:,si])
            ax2.set_title(str(ccXR) + ' 1: ' + str(si))
           
            ax3.imshow(dpcs[1,:,:,si,0])
            ax3.set_title('1: ' + str(si))
           
            for ax in axes:
                ax.axis('off')
               
            plt.gray()
            plt.show()
            plt.draw()
            

##### save single subject files of 4D image data for training segmentation model ####

"""



"""

if 1:
    numPC=5;
    pca = PCA(n_components=numPC)
    kpca = KernelPCA(kernel="rbf",n_components=numPC,eigen_solver='arpack',max_iter=20)
    #tsne0 = TSNE(n_components=3,n_iter=250, n_iter_without_progress=10,n_jobs=4)
    tsne0 = TSNE(n_components=3,n_iter=250, n_iter_without_progress=10,angle=.7,init='pca')
    
    dx = 224;
    dy = 224;
    
    varRatio=np.zeros((len(subjectNamesNormal),numPC))
    for s in range(len(subjectNamesNormal)):
    #for s in [0]:
        normalData4D={};
        patientName=subjectNamesNormal[s];
        
        vol4D00,KM,Box,_,_= funcs_gh.readData4(patientName,subjectInfo,reconMethod,1);  
        print(patientName + ' ' + str(vol4D00.shape[0]) + ' ' + str(vol4D00.shape[1]));
        
        #start data from baseline
        vol4D01=vol4D00[:,:,:,int(subjectNamesNormal0[s,1]):];
        #vol4D01=np.copy(vol4D00);
        #Normalize to the median of intensities
        vol4D0=vol4D01/np.median(vol4D01);
        #vol4D0=zoom( vol4D0,(dx/np.size(vol4D0,0),dy/np.size(vol4D0,1),1,1),order=0);
        
        timeRes0=subjectInfo['timeRes'][patientName];
        if not isinstance(timeRes0, (int, float)):
            timeRes=float(timeRes0.split("[")[1].split(",")[0]);
        else:
            timeRes=np.copy(timeRes0); 
        #print(timeRes)    
        origTimeResVec=np.arange(0,vol4D0.shape[3]*timeRes,timeRes);
        resamTimeResVec=np.arange(0,50*6,6);   # resample to 50 data points
        
        if origTimeResVec[-1]<resamTimeResVec[-1]:
            print(patientName)
        
        f_out = interp1d(origTimeResVec,vol4D0, axis=3,bounds_error=False,fill_value=0)     
        vol4D0 = f_out(resamTimeResVec); 
        
    
        if pc=='kpc/' or pc=='pc/' or pc=='tsne/':
            vol4Dvecs=np.reshape(vol4D0, (vol4D0.shape[0]*vol4D0.shape[1]*vol4D0.shape[2], vol4D0.shape[3]));
            PCs=pca.fit_transform(vol4Dvecs)
        
        if pc=='kpc/': 
            Box1=np.copy(Box);
            sB=np.where(np.sum(Box1,1)==0)[0];
            
            if sB.size:
               Box1[sB[0],:]=Box1[np.where(np.sum(Box1,1)!=0)[0][0],:];
           

            vol4D04kpca=vol4D0[int(Box1[0,0]-int(Box1[0,3]/2)):int(Box1[1,0]+int(Box1[1,3]/2)),\
                       int(Box1[0,1]-int(Box1[0,4]/2)):int(Box1[1,1]+int(Box1[1,4]/2)),int(Box1[0,2]-int(Box1[0,5]/2)):int(Box1[1,2]+int(Box1[1,5]/2)),:];
                               
            vol4Dvecs4kpca=np.reshape(vol4D04kpca, (vol4D04kpca.shape[0]*vol4D04kpca.shape[1]*vol4D04kpca.shape[2], vol4D04kpca.shape[3]));
            vol4Dvecs4kpca=vol4Dvecs4kpca[range(0,vol4Dvecs4kpca.shape[0],4),:];

            
            kpca.fit(vol4Dvecs4kpca)
            #kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)        
            
            KPCs=np.zeros((PCs.shape[0],PCs.shape[1]));
            ndsss=np.arange(0,vol4Dvecs.shape[0],int(vol4Dvecs.shape[0]/200));
            for i in range(len(ndsss)-1):
                if i!=len(ndsss):
                    KPCs[ndsss[i]:ndsss[i+1],:]=kpca.transform(vol4Dvecs[ndsss[i]:ndsss[i+1],:]);
                else:
                    KPCs[ndsss[i]:,:]=kpca.transform(vol4Dvecs[ndsss[i]:,:]);
            print(pca.explained_variance_ratio_) 
            varRatio[s,:]=pca.explained_variance_ratio_;
            
#        if pc=='tsne/': 
#            Box1=np.copy(Box);
#            sB=np.where(np.sum(Box1,1)==0)[0];
#            
#            if sB.size:
#               Box1[sB[0],:]=Box1[np.where(np.sum(Box1,1)!=0)[0][0],:];
#           
#
#            vol4D04kpca=vol4D0[int(Box1[0,0]-int(Box1[0,3]/2)-20):int(Box1[1,0]+int(Box1[1,3]/2)+20),\
#                       int(Box1[0,1]-int(Box1[0,4]/2)-20):int(Box1[1,1]+int(Box1[1,4]/2)+20),int(Box1[0,2]-int(Box1[0,5]/2)):int(Box1[1,2]+int(Box1[1,5]/2)),:];
#                               
#            vol4Dvecs4kpca=np.reshape(vol4D04kpca, (vol4D04kpca.shape[0]*vol4D04kpca.shape[1]*vol4D04kpca.shape[2], vol4D04kpca.shape[3]));
#            vol4Dvecs4kpca=vol4Dvecs4kpca[range(0,vol4Dvecs4kpca.shape[0],4),:];
#
#            
#            tsne0.fit(vol4Dvecs4kpca)
#    #        kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)        
#            
#            
#            KPCs=np.zeros((PCs.shape[0],PCs.shape[1]));
#            ndsss=np.arange(0,vol4Dvecs.shape[0],int(vol4Dvecs.shape[0]/200));
#            for i in range(len(ndsss)-1):
#                if i!=len(ndsss):
#                    KPCs[ndsss[i]:ndsss[i+1],:]=tsne0.fit_transform(vol4Dvecs[ndsss[i]:ndsss[i+1],:]);
#                else:
#                    KPCs[ndsss[i]:,:]=tsne0.fit_transform(vol4Dvecs[ndsss[i]:,:]);            
            

        if pc=='pc/': 
            vol4Dpcs=np.reshape(PCs, (vol4D0.shape[0],vol4D0.shape[1],vol4D0.shape[2], numPC))  
            normalData4D[patientName+'D']=vol4Dpcs/vol4Dpcs.max();
    #    if vol4Dpcs.min()<0:
    #        vol4Dpcs=vol4Dpcs+abs(vol4Dpcs.min())
        elif pc=='kpc/':
            vol4Dkpcs=np.reshape(KPCs, (vol4D0.shape[0],vol4D0.shape[1],vol4D0.shape[2], numPC))
            normalData4D[patientName+'D']=vol4Dkpcs/vol4Dkpcs.max();
                        
        elif pc=='/':
            normalData4D[patientName+'D']=vol4D0/vol4D0.max();
            
        elif pc=='tsne/':
            KPCs=tsne0.fit_transform(vol4Dvecs)#[:,752640:])
#           vol4Dvecs=np.reshape(vol4D0, (vol4D0.shape[0]*vol4D0.shape[1]*vol4D0.shape[2], vol4D0.shape[3]));
#           tsne0.fit(vol4Dvecs)
#           X_embedded = tsne0.transform(vol4Dvecs)
#           X_embedded=tsne(vol4Dvecs, no_dims=numPC)
            vol4Dpcs=np.reshape(KPCs, (vol4D0.shape[0],vol4D0.shape[1],vol4D0.shape[2], 3))  
            normalData4D[patientName+'D']=vol4Dpcs/vol4Dpcs.max();

        print(patientName)    
        normalData4D[patientName+'M']=KM;
        normalData4D[patientName+'B']=Box; #/Box.max();
        #pickle.dump(normalData4D, open("folder-path/singleSubjectsV4kpc/"+patientName+".p", "wb" ))
        pickle.dump(normalData4D, open("folder-path/singleSubjectsV4pc_gh/"+patientName+".p", "wb" ))
        #pickle.dump(normalData4D, open("folder-path/singleSubjectsV4"+pc+patientName+".p", "wb" ))

"""
