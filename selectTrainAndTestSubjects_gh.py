#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def selectTrainAndTestSubjects(TestSetNum):
    
    # reference
    # ['image_file_name',integer_value_of_baseline]

    subjectNamesBaselines=np.array([
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
 
   
    # reference
    # ['image_file_name',kidney_condition]
    # kidney_condition: N = normal; A = abnormal

    subjectNames=np.array([
    ['image_name001','N'],             
    ['image_name002','N'],
    ['image_name003','A'],
    ['image_name004','A'],
    ['image_name005','N'], 
    ['image_name006','N'],
    ['image_name007','A'],             
    ['image_name008','A'],
    ['image_name009','N'],
    ['image_name010','N'],
    ['image_name011','N'], 
    ['image_name012','A']]);
   
    # create a separate test group (in this case two groups) for each network to train
    Test1 =[0,1,2,3,4];
    Test2 =[8,9,10,11];
    
    if TestSetNum==1:
        subjectTest=subjectNames[Test1,0];
        subjectTestBaselines = subjectNamesBaselines[Test1,1];
        
        subjectTestKidneyCondition=subjectNames[Test1,1];  
        subjectTrain=subjectNames[list(set(range(len(subjectNames)))-set(Test1)),0];
        subjectTrainKidneyCondition=subjectNames[list(set(range(len(subjectNames)))-set(Test1)),0];

    elif TestSetNum==2:  
        subjectTest=subjectNames[Test2,0];
        subjectTestBaselines = subjectNamesBaselines[Test2,1];
        
        subjectTestKidneyCondition=subjectNames[Test2,1];  
        subjectTrain=subjectNames[list(set(range(len(subjectNames)))-set(Test2)),0];
        subjectTrainKidneyCondition=subjectNames[list(set(range(len(subjectNames)))-set(Test2)),1];
    
    return subjectTrain,subjectTest,subjectTrainKidneyCondition,subjectTestKidneyCondition,subjectTestBaselines
