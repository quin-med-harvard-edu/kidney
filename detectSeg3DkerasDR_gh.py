def detectSeg3DkerasDR(trainMode,testMode,params):
    print('21!')
    import os, sys
    sys.path.insert(0, '/add-directory-path-where-needed/this-folder')
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    import funcs_gh
    import matplotlib.pyplot as plt
    import pickle
    from datetime import datetime, timedelta
    import numpy as np
    from keras.models import Model,load_model,Sequential
    from keras.layers import Reshape,Input, concatenate, Conv3D,Dense,TimeDistributed
    from keras.layers import MaxPooling3D, UpSampling3D, LSTM,ConvLSTM2D
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint,TensorBoard
    from keras import backend as K
    from keras.preprocessing.image import ImageDataGenerator
    from matplotlib import pyplot
    import tensorflow as tf
    from networks_gh import get_unet2, get_unet3, get_rbunet, get_meshNet,get_denseNet,calculatedPerfMeasures
    from networks_gh import get_unetCnnRnn, augment_sample
    from networks_gh import get_denseNet103, augmentation, generateAugmentation, IoU3D
    from selectTrainAndTestSubjects_gh import selectTrainAndTestSubjects
    from scipy.ndimage import zoom
    from scipy import signal
    from skimage import morphology
    from skimage import data
    from skimage.feature import corner_harris, corner_subpix, corner_peaks
    from skimage.transform import warp, AffineTransform

    import pandas as pd

    TestSetNum=params['TestSetNum'];
    fileNumModel=params['fileNumModel'];
    tDim=params['tDim'];
    tpUsed=params['tpUsed'];
    PcUsed=params['PcUsed'];
    deepRed=params['deepReduction'];
    #multiHead=params['multiHead'];
    
    if PcUsed:
        tDim=5;

    xDim=64; yDim=64; zDim=64;
    xyDim = 64;
    dsFactor = 3.5;

    n_channels = tDim;
    n_classes = 3 # (background, left-kidney, right-kidney)
    
    if PcUsed==1:
        pc='pc/';
    elif PcUsed==2:
        pc='kpc/';
    elif PcUsed==3:
        pc='tsne/';    
    elif PcUsed==4:
        pc='/'
    
    net=params['networkToUse'];
    if net=='meshNet':
        xyDim=96;zDim=96;
        
    ############ stratify between train and test data
    subjectNamesNormalTrain, subjectNamesNormalTest, _ ,testKidCond ,subjectBaselinesTest = selectTrainAndTestSubjects(TestSetNum);
     
    ############ generate train batch data
    def generate_batch():    
        for samples in generate_samples():
            label_batch=np.zeros((len(samples),zDim,xyDim,xyDim,3))
            boxedLabel_batch=np.zeros((len(samples),zDim,xyDim,xyDim,3))
            image_batch=np.zeros((len(samples),zDim,xyDim,xyDim,tDim))

            for s in range(len(samples)):
                
                data4D = pickle.load(open("/path-to-folder-containing-downsampled-images-for-detection-model/singleSubjectsV4pc_detect/"+subjectNamesNormalTrain[samples[s]]+".p","rb" ));
                da=data4D[subjectNamesNormalTrain[samples[s]]+'D'].T;
                im0 = zoom(da,(1,zDim/da.shape[1],1/dsFactor,1/dsFactor),order=0);
                la0 = zoom(data4D[subjectNamesNormalTrain[samples[s]]+'M'].T,(zDim/da.shape[1],1/dsFactor,1/dsFactor),order=0);
                Labels=la0[:,:,:,np.newaxis].astype(int);
                Labels[Labels>2]=2;
                boxedlab = np.copy(Labels)
                uniq=np.unique(Labels);

                for i in range(1,len(uniq)):
                    lNdx=np.where(Labels==uniq[i]);
                    boxedlab[min(lNdx[0]):max(lNdx[0]),min(lNdx[1]):max(lNdx[1]),min(lNdx[2]):max(lNdx[2])]=uniq[i];
                
                lRb=np.copy(boxedlab);lLb=np.copy(boxedlab);lBb=np.copy(boxedlab);
                lRb[lRb!=1]=0;lLb[lLb!=2]=0;lLb[lLb!=0]=1;lBb[lBb==0]=5;lBb[lBb!=5]=0;lBb[lBb==5]=1;
                labelsBoxed=np.concatenate((lBb,lRb,lLb),axis=3);
                
                lR=np.copy(Labels);lL=np.copy(Labels);lB=np.copy(Labels);
                lR[lR!=1]=0;lL[lL!=2]=0;lL[lL!=0]=1;lB[lB==0]=5;lB[lB!=5]=0;lB[lB==5]=1; 
                labels=np.concatenate((lB,lR,lL),axis=3);
                im1=np.swapaxes(im0.T,0,2);
                
                image_batch[s,:,:,:,:]= im1;
                label_batch[s,:,:,:,:] = labels;
                boxedLabel_batch[s,:,:,:,:] = labelsBoxed;
                
            # add data augmentation    
            for i in range(image_batch.shape[0]):
                image_batch[i], label_batch[i],boxedLabel_batch[i] = augment_sample(image_batch[i], label_batch[i],boxedLabel_batch[i])
            
            yield(image_batch, label_batch,boxedLabel_batch)
    
    
    n_samples = 46 # number of train subjects
    batch_size = 2 #3 #4 #batch size
    n_batches = int(n_samples/batch_size); # number of batches
            
    def generate_samples():
        sample_ids = np.random.permutation(n_samples)
        for i in range(n_batches):
            inds = slice(i*batch_size, (i+1)*batch_size)
            yield sample_ids[inds]
    
    
    ################### visual check for cropped segmentaation  
    #for i in range(40,42):
    #    #    plt.figure();plt.imshow(Data[i,:,:,10,0].T);
    #    f, axarr = plt.subplots(1, 2);
    #    
    #    # axarr[0].imshow(DataTest[0,zFix,:,:,1].T);
    #    axarr[0].imshow(Data[i,:,:,40,2].T);
    #    axarr[1].imshow(Labels[i,:,:,40,0].T);
    ########################## set weights,train and test ##############################
    
    ### dice accuracy
    def dice_coef(y_true, y_pred):
        y_true_f = y_true.flatten();
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))
    
    ### tversky_coefficient
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

    ### tversky_coef loss
    def tversky_loss(alpha, beta, weights=False):
        def tversky(y_true, y_pred):
            return -tversky_coef(y_true, y_pred, alpha, beta, weights)
        return tversky

    ### tversky_coef loss
    tversky = tversky_loss(alpha=0.3, beta=0.7, weights=False)
    
    # initial class weights
    class_weights=np.array([0.05,0.6,0.6]);    
    
    if net=='rbUnet':
        model = get_rbunet(xyDim,zDim,n_channels,n_classes,deepRed,0);
    elif  net=='Unet':
        model = get_unet3(xyDim,zDim,n_channels,n_classes,deepRed,0);
    elif  net=='Unet-rnn':
        predLayersCat=get_unetCnnRnn(xyDim,zDim,1,n_classes,deepRed,0);  
        predLayersCat=Reshape((1,64, 64, 64))(predLayersCat)
        for i in range(1,n_channels):
            predLayer = get_unetCnnRnn(xyDim,zDim,1,n_classes,deepRed,0); 
            predLayer=Reshape((1,64, 64, 64))(predLayer)
            predLayersCat= concatenate([predLayersCat, predLayer], axis=1)
        model = Sequential()
        model.add(ConvLSTM2D(64, kernel_size=3, padding='same',input_shape=(50, 64, 64,64)))
        #lstm_num_predictions=1;
        #model.add(Dense(lstm_num_predictions))
        model = Model(inputs=[inputs], outputs=[Pred])
        model.compile(optimizer=Adam(lr=1e-4),  loss=dice_coef_loss2)
    elif net=='meshNet':
        xyDim=96;zDim=96;
        model = get_meshNet(xyDim,zDim,n_channels,n_classes,deepRed,0);
    elif net== 'denseNet':
        model = get_denseNet(xyDim,zDim,n_channels,n_classes,deepRed,0);
    elif net== 'tNet':
        model = get_denseNet103(xyDim,zDim,n_channels,n_classes,deepRed,0);    
     
    ### create folder to hold detection model(s)   
    if isinstance(fileNumModel, (int)): 
        fileNumModel=str(fileNumModel);
    else:
        fileNumModel='Net'+net+'_time'+str(tDim)+'_pcUsed'+str(PcUsed)+'_tpUsed'+str(tpUsed)+'_DR'+str(deepRed)+'_testSet'+str(TestSetNum);

    address = "path-to-folder-to-hold-detection-model(s)/"+fileNumModel+"/"  
    if trainMode:    
        os.system('mkdir '+address);
        #current_time = datetime.now() + timedelta(hours=-5)
        #log_dir=address+str(current_time)[:19]   
        #callbacks = [
        #TensorBoard(address+'tbevents',histogram_freq=0, write_graph=True, write_images=False),
        #ModelCheckpoint(address+'conv3Dkeras.h5',verbose=1,monitor='val_loss', save_best_only=True, save_weights_only=True),
        #]

        # set multiHead 
        multiHead = 0;
        
        nb_epoch = 400; #500 #100
        epCounter = 0;
        for e in range(nb_epoch):
            print("epoch %d" % e)
            for image_batch, label_batch,boxedLbatch in generate_batch(): 
                print(epCounter,label_batch.shape[0]);
                #epCounter+=(batch_size+1);
                xx=(batch_size*2)+1;
                epCounter+=xx;
                
                if multiHead == 1:
                    labelsDict={'seg':label_batch,'box':boxedLbatch};
                    classWdict={'seg':class_weights,'box':class_weights};
                else:
                    labelsDict={'seg':label_batch};
                    classWdict={'seg':class_weights};
                                        
                model.fit(image_batch, label_batch, batch_size=batch_size*2, class_weight=class_weights,
                           initial_epoch =epCounter, epochs=epCounter+xx,verbose=1, shuffle=True,validation_split=0.5); #,callbacks=callbacks);

                if epCounter >= 100 and epCounter <= 500:
                    model.save(address+'detect3D_'+str(epCounter)+'.h5')
                    
                if epCounter >= 2000 and epCounter <= 3000:
                    model.save(address+'detect3D_'+str(epCounter)+'.h5')
                    
                if epCounter >= 5000 and epCounter <= 10000:
                    model.save(address+'detect3D_'+str(epCounter)+'.h5')
                 
                if epCounter >= 15000 and epCounter <= 20000:
                    model.save(address+'detect3D_'+str(epCounter)+'.h5')
                    
                if epCounter >= 28000 and epCounter <= 35000:
                    model.save(address+'detect3D_'+str(epCounter)+'.h5')   
                    
                if epCounter >= 40000:
                    model.save(address+'detect3D_'+str(epCounter)+'.h5')
    
    performanceMeasures,avgPerf,missedVoxels=[],[],[];
     
    if testMode == 1:

        fileNumModel='Net'+net+'_time'+str(tDim)+'_pcUsed'+str(PcUsed)+'_tpUsed'+str(tpUsed)+'_DR'+str(deepRed)+'_testSet'+str(TestSetNum);
        address = "path-to-folder-containing-trained-detection-model(s)/"+fileNumModel+"/"
        #address = "path-to-folder-to-hold-detection-model(s)" + "/NetrbUnet_time5_pcUsed1_tpUsed50_DR0_testSet1/"
        #address = "path-to-folder-to-hold-detection-model(s)" + "/NetrbUnet_time5_pcUsed1_tpUsed50_DR0_testSet2/"
        
        
        selectedEpoch=params['selectedEpoch'];
        # if isinstance(selectedEpoch, (int)): 
        #     selectedEpoch=str(selectedEpoch);
        # else:
        #     txt_file = open(address+'selectedEpoc.txt','r')
        #     selectedEpoch=str(int(txt_file.read()))
        model.load_weights(address+'detect3D_'+selectedEpoch+'.h5');

        # perform detection (coarse segmentation) for each test subject
        # and save to a file
        for s in range(len(subjectNamesNormalTest)):
            
            xyDim = 64; zDim = 64;
            LabelsTest=np.zeros((1,zDim,xyDim,xyDim));
            DataTest=np.zeros((1,zDim,xyDim,xyDim,tDim));
            boxTest=np.zeros((1,12));
        
            # access test input data
            data4D = pickle.load(open("/path-to-folder-containing-downsampled-images-for-detection-model/"+"singleSubjectsV4pc_detect/"+subjectNamesNormalTest[s]+".p","rb" ));
            
            #### extract ground-truth labels and 'box' to evaluate performance of detection model  
            da = data4D[subjectNamesNormalTest[s]+'D'].T;
            im0 = zoom(da,(1,zDim/da.shape[1],1/dsFactor,1/dsFactor),order=0);
            la0 = zoom(data4D[subjectNamesNormalTest[s]+'M'].T,(zDim/da.shape[1],1/dsFactor,1/dsFactor),order=0);
            B=data4D[subjectNamesNormalTest[s]+'B'];
            B[:,[0,1,3,4]]=(B[:,[0,1,3,4]]/dsFactor).astype(int);
            B[:,[2,5]]=(B[:,[2,5]]*(zDim/da.shape[1])).astype(int);    
            
            sx = 0;
            LabelsTest[sx,0:np.size(la0,0),:,:]=la0;
            DataTest[sx,:,:,:,:]=np.swapaxes(im0.T,0,2); 
            boxTest[sx,:]=B.flatten();

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
            #LabelsTest2=np.copy(LabelsTest);labels_pred2=np.copy(labels_pred);
            #LabelsTest2[LabelsTest2!=0]=1;labels_pred2[labels_pred2!=0]=1;

            ##### generate bounding boxes from coarse segmentation #####
            boxPred=np.zeros((DataTest.shape[0],12))
                
            si = 0;    
            aL=np.nonzero(labels_pred[si,:,:,:].T==2); #left
            aR=np.nonzero(labels_pred[si,:,:,:].T==1); #right
            
            left = labels_pred[si,:,:,:].T==2;
            left = left.astype(int);
            
            right = labels_pred[si,:,:,:].T==1;
            right = right.astype(int);
        
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
            
            boxes=np.hstack([np.array(boxR),np.array(boxL)]);
            boxPred[si,:]=boxes;
            
            
            # #### calculate number of missed voxels (number of voxels out of bounding box)
            # columns = ['Name','kidney Condition','F1-Score', 'Prec','Rec','VEE','testSet','Model','IoU','missedVoxelsR','missedVoxelsL'];
            # index=np.arange(1);
            # performanceMeasures= pd.DataFrame(index=index, columns=columns)
            # performanceMeasures= performanceMeasures.fillna(0);
            
            # IoU,missedVoxels=IoU3D(boxPred[si,:],boxTest[si,:],labels_pred2[si,:,:,:].T);
            # avgPerfOverKidneys=calculatedPerfMeasures(LabelsTest2[si,:,:,:],labels_pred2[si,:,:,:]);
            # performanceMeasures.ix[s]=pd.Series({'Name':subjectNamesNormalTest[s],'kidney Condition':testKidCond[s],'F1-Score':avgPerfOverKidneys[0]*100,'Prec':avgPerfOverKidneys[1]*100,\
            #          'Rec':avgPerfOverKidneys[2]*100,'VEE':avgPerfOverKidneys[4],'testSet':TestSetNum,'Model':net,'IoU':IoU,'missedVoxelsR':missedVoxels[0],'missedVoxelsL':missedVoxels[1]});
               
   
            ##### save detection results as coarse segmentation masks for each test subject ####
            pathToFolder = 'path-to-folder-to-contain-segmented-image-files' + '/detected/' + 'subjectNamesNormalTest[s]' + '_seq1';
            if not os.path.exists(pathToFolder):
                os.makedirs(pathToFolder)

            # extract test image volume    
            # fileAddress = path to .xls sheet that contains time information for each test subject file   
            # fileAddress='path-to-folder"+"/subjectDicomInfo_gh.xls';
            # subjectInfo=pd.read_excel(fileAddress, sheetname=0);
            # reconMethod='SCAN'; 
            # vol4D00,_,_,_,_ = funcs_gh.readData4(subjectNamesNormalTest[s],subjectInfo,reconMethod,0);

            # extract test image volume dimensions
            vol4D00 = data4D[subjectNamesNormalTest[s]+'D'];
            zDimm = vol4D00.shape[2];
            xyDim = 224;

            # resample predicted labels (right, left) to original test image spatial dimensions
            KMR = zoom(right,(xyDim/np.size(right,0),xyDim/np.size(right,1),zDimm/np.size(right,2)),order=0);
            KML = zoom(left,(xyDim/np.size(left,0),xyDim/np.size(left,1),zDimm/np.size(left,2)),order=0);
            
            predMaskR=np.zeros((1,xyDim,xyDim,zDimm));
            predMaskL=np.zeros((1,xyDim,xyDim,zDimm));
            
            sc = 0;
            predMaskR[sc,:,:,:] = KMR; 
            predMaskL[sc,:,:,:] = KML;    
    
            if np.sum(predMaskR) != 0:
                predMaskL=morphology.remove_small_objects(predMaskL.astype(bool), min_size=256,in_place=True).astype(int);
            if np.sum(predMaskL) != 0:
                predMaskR=morphology.remove_small_objects(predMaskR.astype(bool), min_size=256,in_place=True).astype(int);   
            
            Masks2Save={};
        
            predMaskR2=zoom(predMaskR[sc,:,:,:],(1,1,1),order=0);
            predMaskL2=zoom(predMaskL[sc,:,:,:],(1,1,1),order=0);
            
            Masks2Save['R']=np.copy(predMaskR2.astype(float));
            Masks2Save['L']=np.copy(predMaskL2.astype(float));

            # path to .xls sheet that contains time information for each test subject file (subjectNamesNormalTest[s])
            fileAddress='path-to-folder"+"/subjectDicomInfo_gh.xls';
            subjectInfo=pd.read_excel(fileAddress, sheetname=0);
            reconMethod='SCAN';

            # write coarse segmentation (detection) results to file
            funcs_gh.writeMasksDetect(subjectNamesNormalTest[s],subjectInfo,reconMethod,Masks2Save,1);
            
            """
            #### calculate number of missed voxels (number of voxels out of bounding box)
            columns = ['Name','kidney Condition','F1-Score', 'Prec','Rec','VEE','testSet','Model','IoU','missedVoxelsR','missedVoxelsL'];
            index=np.arange(len(subjectNamesNormalTest));
            performanceMeasures= pd.DataFrame(index=index, columns=columns)
            performanceMeasures= performanceMeasures.fillna(0);
            
            for s in range(DataTest.shape[0]):
                IoU,missedVoxels=IoU3D(boxPred[s,:],boxTest[s,:],labels_pred2[s,:,:,:].T);
                avgPerfOverKidneys=calculatedPerfMeasures(LabelsTest2[s,:,:,:],labels_pred2[s,:,:,:]);
                performanceMeasures.ix[s]=pd.Series({'Name':subjectNamesNormalTest[s],'kidney Condition':testKidCond[s],'F1-Score':avgPerfOverKidneys[0]*100,'Prec':avgPerfOverKidneys[1]*100,\
                     'Rec':avgPerfOverKidneys[2]*100,'VEE':avgPerfOverKidneys[4],'testSet':TestSetNum,'Model':net,'IoU':IoU,'missedVoxelsR':missedVoxels[0],'missedVoxelsL':missedVoxels[1]});
                
            normalPerf=performanceMeasures[performanceMeasures['kidney Condition'] == 'N'].iloc[:,2:6].mean().tolist();        
            abnormalPerf=performanceMeasures[performanceMeasures['kidney Condition'] == 'A'].iloc[:,2:6].mean().tolist();
            avgPerf=normalPerf+abnormalPerf
    
            if 1:
                volumEstimError=np.zeros((labels_pred.shape[0],))
                performanceMeasures=np.zeros((len(subjectNamesNormalTest),));
                for s in range(DataTest.shape[0]):
                    zAx=40;
                    f, axarr = plt.subplots(2, 3);
                    #axarr[0].imshow(imgs_mask_test[0,:,:,0].T,cmap='gray');
                    axarr[0,0].imshow(labels_pred[s,zAx,:,:],cmap='gray');axarr[0,0].set_title('Predicted');
                    axarr[0,1].imshow(LabelsTest[s,zAx,:,:],cmap='gray');axarr[0,1].set_title('Original');
                    axarr[0,2].imshow(DataTest[s,zAx,:,:,1],cmap='gray');axarr[0,2].set_title(subjectNamesNormalTest[s]);
                    
                    axarr[1,0].imshow(labels_pred[s,:,int(xyDim/2),:],cmap='gray');
                    axarr[1,1].imshow(LabelsTest[s,:,int(xyDim/2),:],cmap='gray');
                    axarr[1,2].imshow(DataTest[s,:,int(xyDim/2),:,1],cmap='gray');    
                    
                    volumEstimError[s]=np.count_nonzero(LabelsTest2[s,:,:,:])-np.count_nonzero(labels_pred2[s,:,:,:]);
                    performanceMeasures[s]=dice_coef(LabelsTest2[s,:,:,:],labels_pred2[s,:,:,:]);
            """
        
    
    return performanceMeasures,avgPerf     

    
    
    
    
    
    
    
    
    
    
    
    
    
    
