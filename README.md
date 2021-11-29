The source code in this GitHub respiratory is based on the methods proposed in:

(1) 'Automatic renal segmentation in DCE-MRI using convolutional neural networks (2018)' by Marzieh Haghighi, Simon K. Warfield and Sila Kurugol. https://doi.org/10.1109/ISBI.2018.8363865

(2) '3D Deep Learning for Anatomical Structure Segmentation in Multiple Imaging Modalities (2021)' by Barbara Villarini, Hykoush Asaturyan, Sila Kurugol, Onur Afacan, Jimmy D. Bell and E. Louise Thomas. https://doi.org/10.1109/CBMS52027.2021.00066

The program presented in this GitHub respiratory was built using Python 3.0 (Keras and TensorFlow).

Information about each source file:

**1)  trainTest_gh.py**

This file is the main program for executing the training stage (trainMode) or testing stage (testMode) of a kidneys detection model or a kidneys segmentation model. Please comment and un-comment the model you wish to train/test and select network type (netNdx) and change additional parameters where necessary.

**2) selectTrainAndTestSubjects_gh.py**

This file provides a template for adding the name of files that will contain 4D image data. The contents in this file are merely examples that you can substitute with the names of your own experimental data. 

**3) saveDataSingleSubjects_gh.py**

This file details the implementation for generating downsampled and interpolated image and kidney mask (ground-truth) arrays for training/testing the detection model and segmentation model.

**4) funcs_gh.py**

This file delivers a number of functions that (i) process 4D .nii image data and generate its corresponding 4D image data array; (ii) process 3D .nii kidney ground-truth mask data and generate its corresponding mask array; (iii) generate the minimal bounding boxes for each kidney via the ground-truth masks; (iv) process segmentation/detection masks and save to a .nii file. The file paths in these functions serve as a template based on the original implementation.

**4a) subjectDicomInfo_gh.xls**

This file provides an example template based on the original file used to contain information that relates to the time taken for sequence 1 and sequence 2 during image acquisition. All information and values in this spreadsheet are theoretical.

**5) detectSeg3DkerasDR_gh.py**

This file details the implementation for training (trainMode) and testing (testMode) the detection (coarse segmentation) deep learning model.

**6) croppedSeg3DkerasDR_gh.py**

This file details the implementation for training (trainMode) and testing (testMode) the segmentation deep learning model.

**7) networks_gh.py**

This file details the architectures of experimented deep learning models for both detection and segmentation.

**8) densenet_gh.py**

This file details a range of architectures implemented from the original group of DenseNets architectures using Keras.

**9) subpixel_gh**

This file details the implementation of a sub-pixel convolutional upscaling layer based on the paper "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" (https://arxiv.org/abs/1609.05158).

**10) tensorflow_backend_gh**

This file details the implementation of a phase shift algorithm to convert channels/depth for spatial resolution.

**11) detect_and_segment_gh.py**

This file can process one or multiple 4D DCE-MRI volumes. Firstly, detection is performed to generate a bounding box over each kidney of interest. Second, segmentation is performed using the cropped region containing the kidney(s) of interest. Please ensure you have a trained model for each stage.
    
