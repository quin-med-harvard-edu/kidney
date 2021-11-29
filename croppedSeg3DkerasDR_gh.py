def croppedSeg3DkerasDR(trainMode,testMode,params):
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
    from networks_gh import get_unet2, get_unet3, get_rbunet, get_meshNet, get_denseNet, calculatedPerfMeasures
    from networks_gh import get_unetCnnRnn
    from networks_gh import get_denseNet103, augment_sample_segment
    from selectTrainAndTestSubjects_gh import selectTrainAndTestSubjects
    from scipy.ndimage import zoom
    from scipy import signal
    from skimage import morphology
    from skimage import data
    from skimage.feature import corner_harris, corner_subpix, corner_peaks
    from skimage.transform import warp, AffineTransform
    from sklearn.decomposition import PCA, KernelPCA
    from scipy.interpolate import interp1d
    import pandas as pd
    
    
    TestSetNum=params['TestSetNum'];
    fileNumModel=params['fileNumModel'];
    tDim=params['tDim'];
    tpUsed=params['tpUsed'];
    PcUsed=params['PcUsed'];
    deepRed=params['deepReduction'];
    
    if PcUsed:
        tDim=5;

    xDim=64; yDim=64; zDim=64;
    xyDim=64;
    
    n_channels = tDim;
    n_classes = 2 # total classes (kidney, non-kidney)
    
    if PcUsed==1:
        pc='pc';
    elif PcUsed==2:
        pc='kpc/';
    elif PcUsed==3:
        pc='tsne/';    
    elif PcUsed==4:
        pc='/'
    
    net=params['networkToUse'];
    if net=='meshNet':
        xyDim=96;zDim=96;
        
    ############ stratify train and test data #########
    subjectNamesNormalTrain, subjectNamesNormalTest, _ ,testKidCond ,subjectBaselinesTest = selectTrainAndTestSubjects(TestSetNum);
     
    ############ generate train batch data #########
    def generate_batch():    
        for samples in generate_samples():
            label_batch=np.zeros((len(samples)*2,zDim,xyDim,xyDim,2))
            image_batch=np.zeros((len(samples)*2,zDim,xyDim,xyDim,tDim))
            
            for s in range(len(samples)):
                data4D = pickle.load(open("/path-to-folder-containing-downsampled-images-for-segmentation-model/singleSubjectsCroppedV4pc_segment/"+subjectNamesNormalTrain[samples[s]]+'_tp'+str(tpUsed)+".p","rb" ));
                labels=data4D[subjectNamesNormalTrain[samples[s]]+'M'];
                labels[labels>1]=1;labels[labels<1]=0;    
                labels=labels[:,:,:,:,np.newaxis];labels=np.concatenate((labels,1-labels),axis=4);
                da = data4D[subjectNamesNormalTrain[samples[s]]+'D'];
                
                if net=='meshNet':
                    n=16;
                    da2=np.pad(da,((0,0),(n,n), (n, n),(n,n),(0,0)), 'edge');
                    labels2=np.pad(labels,((0,0),(n,n),(n,n),(n,n),(0,0)), 'edge');
                
                image_batch[2*s:2*s+2,:,:,:,:]= da;
                label_batch[2*s:2*s+2,:,:,:,:]= labels;

            # generate data augmentation    
            #for ix in range(image_batch.shape[0]):
            #    image_batch[ix], label_batch[ix] = augment_sample_segment(image_batch[ix],label_batch[ix]); #,[2,5],10)
            
            yield(image_batch, label_batch)
    
    
    n_samples = 46; # number of training image files
    batch_size = 3  # size of batch
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
    
    
    ### dice accuracy
    def dice_coef(y_true, y_pred):
        y_true_f = y_true.flatten();
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f));
    
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

    ### tversky_coefficient loss
    def tversky_loss(alpha, beta, weights=False):
        def tversky(y_true, y_pred):
            return -tversky_coef(y_true, y_pred, alpha, beta, weights)
        return tversky

    tversky = tversky_loss(alpha=0.3, beta=0.7, weights=False)
    
    ### initial class weights
    class_weights=np.array([0.5,0.5]);
    #class_weights=np.array([0.2,0.8]);
    #class_weights=np.array([0.4,0.6]);
    
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
    

    ### create folder to hold trained segmentation model(s)       
    if isinstance(fileNumModel, (int)): 
        fileNumModel=str(fileNumModel);
    else:
        fileNumModel='Net'+net+'_time'+str(tDim)+'_pcUsed'+str(PcUsed)+'_tpUsed'+str(tpUsed)+'_DR'+str(deepRed)+'_testSet'+str(TestSetNum);
        
    address = "path-to-folder-to-hold-segmentation-model(s)/"+fileNumModel+"/"  
    if trainMode:    
        os.system('mkdir '+address);
        #current_time = datetime.now() + timedelta(hours=-5)
        #log_dir=address+str(current_time)[:19]   
        #callbacks = [
        #TensorBoard(address+'tbevents',histogram_freq=0, write_graph=True, write_images=False),
        #ModelCheckpoint(address+'conv3Dkeras.h5',verbose=1,monitor='val_loss', save_best_only=True, save_weights_only=True),
        #]
        
        nb_epoch = 400; #800
        epCounter = 0;
        for e in range(nb_epoch):
            print("epoch %d" % e)
            for image_batch, label_batch in generate_batch(): 
                print(epCounter,label_batch.shape[0]);
                xx=(batch_size*2)+1;
                epCounter+=xx;
                
                model.fit(image_batch, label_batch, batch_size=batch_size*2, class_weight=class_weights,
                           initial_epoch =epCounter, epochs=epCounter+xx,verbose=1, shuffle=True,validation_split=0.5); #,callbacks=callbacks);
                
                model.save(address+'croppedSeg3D_'+str(epCounter)+'.h5')

            """
            #save model between defined epCounter
            
            if epCounter >= 500 and epCounter <= 1000:
                model.save(address+'croppedSeg3D_'+str(epCounter)+'.h5')
                     
            if epCounter >= 5000 and epCounter <= 10000:
                model.save(address+'croppedSeg3D_'+str(epCounter)+'.h5')
             
            if epCounter >= 15000 and epCounter <= 20000:
                model.save(address+'croppedSeg3D_'+str(epCounter)+'.h5')
                
            if epCounter >= 28000 and epCounter <= 35000:
                model.save(address+'croppedSeg3D_'+str(epCounter)+'.h5')   
                
            if epCounter >= 40000 and epCounter <= 45000:
                model.save(address+'croppedSeg3D_'+str(epCounter)+'.h5')
                
            if epCounter >= 60000:
                model.save(address+'croppedSeg3D_'+str(epCounter)+'.h5')
            """
                
                # # list all data in history
                # print(history.history.keys())
                # # summarize history for accuracy
                # plt.plot(history.history['acc'])
                # plt.plot(history.history['val_acc'])
                # plt.title('model accuracy')
                # plt.ylabel('accuracy')
                # plt.xlabel('epoch')
                # plt.legend(['train', 'val'], loc='upper left')
                # plt.show()
                
                # # summarize history for loss
                # plt.plot(history.history['loss'])
                # plt.plot(history.history['val_loss'])
                # plt.title('model loss')
                # plt.ylabel('loss')
                # plt.xlabel('epoch')
                # plt.legend(['train', 'val'], loc='upper left')
                # plt.show()
    
        
    if testMode == 1:
        
        performanceMeasuresX,volumEstimError,performanceMeasures,avgPerf = [],[],[],[];

        fileNumModel='Net'+net+'_time'+str(tDim)+'_pcUsed'+str(PcUsed)+'_tpUsed'+str(tpUsed)+'_DR'+str(deepRed)+'_testSet'+str(TestSetNum);
        address = "path-to-folder-containing-trained-segmentation-model(s)/"+fileNumModel+"/"
        #address = "path-to-folder-to-hold-segmentation-model(s)" + "/NettNet_time5_pcUsed1_tpUsed50_DR0_testSet1/"
        #address = "path-to-folder-to-hold-segmentation-model(s)" + "/NettNet_time5_pcUsed1_tpUsed50_DR0_testSet2/"
        
        selectedEpoch=params['selectedEpoch'];
        # if isinstance(selectedEpoch, (int)): 
        #     selectedEpoch=str(selectedEpoch);
        # else:
        #     txt_file = open(address+'selectedEpoc.txt','r')
        #     selectedEpoch=str(int(txt_file.read()))
        #
        model.load_weights(address+'croppedSeg3D_'+selectedEpoch+'.h5');

        # perform segmentation for each test subject using ground-truth detection 'box'
        # to check the segmentation model's performance
        dx=64; dy=64; dz=64; n_channels=5; #50
        for s in range(len(subjectNamesNormalTest)): 
            
            pName = subjectNamesNormalTest[s];
            DataCroppedTest=np.zeros((2,dx,dy,dz,n_channels));

            # extract test image volume
            sc=0;
            data4D = pickle.load(open("/path-to-folder-containing-downsampled-images-for-segmentation-model"+"/singleSubjectsCroppedV4pc_segment/"+pName+'_tp'+str(tpUsed)+".p","rb"));           
            da=data4D[pName+'D'];
            DataCroppedTest[2*sc:2*sc+2,:,:,:,:]=da;

            # perform prediction using trained segmentation model
            cropped_mask_test = model.predict(DataCroppedTest, verbose=1)
            if cropped_mask_test.min()<0:
                cropped_mask_test=abs(cropped_mask_test.min())+cropped_mask_test;

            # extract predicted labels
            imgs_mask_test2=np.copy(cropped_mask_test);
            imgs_mask_test2[:,:,:,:,0]=cropped_mask_test[:,:,:,:,0];
            imgs_mask_test2[:,:,:,:,1]=cropped_mask_test[:,:,:,:,1];
            labels_pred_2=np.argmax(imgs_mask_test2, axis=4);

            # path to .xls sheet that contains time information for each test subject file (pName)
            fileAddress='path-to-folder"+"/subjectDicomInfo_gh.xls';
            subjectInfo=pd.read_excel(fileAddress, sheetname=0);
            reconMethod='SCAN';

            # extract ground-truth kidney mask (KM) and bounding box for each kidney (Box)
            vol4D00, KM, Box, _, _ = funcs_gh.readData4(pName,subjectInfo,reconMethod,1);
            KM[KM>1]=1;
            zDimOrig = vol4D00.shape[2];
            Box=np.reshape(Box,[2,6]).astype('int');
            
            # identify whether right kidney exists
            # identify whether left kidney exists
            kidneyNone=np.nonzero(np.sum(Box,axis=1)==0); #right/left
            if kidneyNone[0].size!=0:
                kidneyNone=np.nonzero(np.sum(Box,axis=1)==0)[0][0]; #right/left
            
            # add extra margins
            xSafeMagin=10;ySafeMagin=10;zSafeMagin=3;
            if Box[0,2]+Box[0,5]+3 >= KM.shape[2] or Box[0,2]+Box[0,5]-3 <0:
                Box[:,[3,4,5]]=Box[:,[3,4,5]]+[xSafeMagin,ySafeMagin,0];
            else:
                Box[:,[3,4,5]]=Box[:,[3,4,5]]+[xSafeMagin,ySafeMagin,zSafeMagin];
        
            # add extra margins
            # xSafeMagin=12;ySafeMagin=12;zSafeMagin=3;
            # Box[:,[3,4,5]]=Box[:,[3,4,5]]+[xSafeMagin,ySafeMagin,zSafeMagin];

            # resample predicted kidney labels to appropriate size and location
            # of original test subject dimensions
            xyDim=224;
            predMaskR=np.zeros((1,xyDim,xyDim,zDimOrig));
            predMaskL=np.zeros((1,xyDim,xyDim,zDimOrig));
            
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
                predMaskL2[predMaskL2==1]=2;    
                
                Masks2Save={};
                
                predMaskR2=zoom(predMaskR[sc,:,:,:],(1,1,1),order=0);
                predMaskL2=zoom(predMaskL[sc,:,:,:],(1,1,1),order=0);
                
                Masks2Save['R']=np.copy(predMaskR2.astype(float));
                Masks2Save['L']=np.copy(predMaskL2.astype(float));
        
                print(pName)
                pathToFolder = "path-to-folder-to-contain-segmented-image-files"+"/segmented/" + pName + '_seq1'
                if not os.path.exists(pathToFolder):
                    os.makedirs(pathToFolder)
                funcs_gh.writeMasks(pName,subjectInfo,reconMethod,Masks2Save,1);

        """
        xDim=64; yDim=64; zDim=64; tDim=5;
        LabelsTest2=np.zeros((len(subjectNamesNormalTest)*2,xDim,yDim,zDim))
        DataTest2=np.zeros((len(subjectNamesNormalTest)*2,xDim,yDim,zDim,tDim))
        #LabelsTest2=np.zeros((1*2,xDim,yDim,zDim))
        #DataTest2=np.zeros((1*2,xDim,yDim,zDim,tDim))
        
        # obtain input image data and ground-truth for total test data (cropped)
        for s in range(len(subjectNamesNormalTest)):            
            data4D = pickle.load(open("/fileserver/abd/marzieh/preprocessedData_ha/singleSubjectsCroppedV4pc_segment/"+subjectNamesNormalTest[s]+'_tp'+str(tpUsed)+".p","rb" ));   
            la=data4D[subjectNamesNormalTest[s]+'M'];
            LabelsTest2[2*s:2*s+2,:,:,:]=la;
            da=data4D[subjectNamesNormalTest[s]+'D'];
            DataTest2[2*s:2*s+2,:,:,:,:]=da;
            
        if net=='meshNet':
            n=16;
            DataTest2=np.pad(DataTest2,((0,0),(n,n), (n, n),(n,n),(0,0)), 'edge');
            LabelsTest2=np.pad(LabelsTest2,((0,0),(n,n), (n, n),(n,n)), 'edge');

        # obtain prediction for total test data (cropped)
        imgs_mask_test = model.predict(DataTest2, verbose=1)
        if imgs_mask_test.min()<0:
            imgs_mask_test=abs(imgs_mask_test.min())+imgs_mask_test
        imgs_mask_test2=np.copy(imgs_mask_test);
        imgs_mask_test2[:,:,:,:,0]=imgs_mask_test[:,:,:,:,0]
        imgs_mask_test2[:,:,:,:,1]=imgs_mask_test[:,:,:,:,1]
        labels_pred=np.argmax(imgs_mask_test2, axis=4);
        labels_pred[labels_pred>1]=1;labels_pred[labels_pred<0]=0;
    
        #import pandas as pd
        columns = ['Name','kidney Condition','F1-Score', 'Prec','Rec','VEE','testSet','Model'];
        index=np.arange(len(subjectNamesNormalTest));
        performanceMeasures= pd.DataFrame(index=index, columns=columns)
        performanceMeasures= performanceMeasures.fillna(0);
        
        # compute spatial overlap performance of prediction for total test data (cropped)
        for s in range(len(subjectNamesNormalTest)):
            right=calculatedPerfMeasures(LabelsTest2[2*s,:,:,:],labels_pred[2*s,:,:,:]);
            left=calculatedPerfMeasures(LabelsTest2[(2*s)+1,:,:,:],labels_pred[(2*s)+1,:,:,:]);
            avgPerfOverKidneys=np.mean([right,left],axis=0);
            performanceMeasures.ix[s]=pd.Series({'Name':subjectNamesNormalTest[s],'kidney Condition':testKidCond[s],'F1-Score':avgPerfOverKidneys[0]*100,'Prec':avgPerfOverKidneys[1]*100,
                 'Rec':avgPerfOverKidneys[2]*100,'VEE':avgPerfOverKidneys[4],'testSet':TestSetNum,'Model':net+'-DR'+str(deepRed)+'-PC'+str(PcUsed)});

        normalPerf=performanceMeasures[performanceMeasures['kidney Condition'] == 'N'].iloc[:,2:6].mean().tolist();
        abnormalPerf=performanceMeasures[performanceMeasures['kidney Condition'] == 'A'].iloc[:,2:6].mean().tolist();
        avgPerf=np.round(normalPerf+abnormalPerf,2)
        
        volumEstimError=np.zeros((labels_pred.shape[0],))
        performanceMeasuresX=np.zeros((2*len(subjectNamesNormalTest),));
    
        whatIs = labels_pred.shape[0]
        for i in range(labels_pred.shape[0]):
            # zAx=20;
            # f, axarr = plt.subplots(1, 3);
            # axarr[0].imshow(imgs_mask_test[0,:,:,0].T,cmap='gray');
            # axarr[0].imshow(labels_pred[i,:,:,zAx].T,cmap='gray');axarr[0].set_title('prediction');
            # axarr[1].imshow(LabelsTest2[i,:,:,zAx].T,cmap='gray');axarr[1].set_title('manual label');
            # axarr[2].imshow(DataTest2[i,:,:,zAx,40].T,cmap='gray'); 
                
            volumEstimError[i]= np.count_nonzero(LabelsTest2[i,:,:,:])-np.count_nonzero(labels_pred[i,:,:,:]);
            performanceMeasuresX[i]= dice_coef(LabelsTest2[i,:,:,:],labels_pred[i,:,:,:])*100;
            
            test = 1;
         """

    return performanceMeasuresX,volumEstimError,performanceMeasures,avgPerf

    
    
    
    
    
    
    
    
    
    
    
    
    
    
