from sklearn.model_selection import StratifiedKFold
import pandas as pd
import glog as log
import numpy as np
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
from collections import defaultdict
def readData(path):
    data = pd.read_csv(path)
    return data
def dummy_df(df, todummy_list):
    """
    This function creates dummy variables from the column names given in
    todummy_list.
    
    Params:
      df..............The dataframe to which the dummy columns are created
      todummy_list....The list of columns to convert into dummy
    Returns:
      df.......The modified df
    """
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df

def printValues(df,field,value,onlyPercentage=False):
    t=0
    idx=[]
    s=0
    for val in train[field]:
        if str(val)==str(value):
            idx.append(t)
            if df.iloc[t,-1] == 1:
                s+=1
        t+=1
    print(field,"has ",str(s/float(len(idx))),"detections for ",len(idx),"observations")
    if onlyPercentage:
        return
    print(df.iloc[idx,:].head(len(idx)))
def compare(df,fieldVal):
    for field,val in fieldVal.items():
        c = df[df[field]==val].HasDetections
        if len(c)==0:
            print(field,"has ",str(sum(c))+"/"+str(float(len(c))),"detections for ",len(c),"observations",fieldVal)
        else:
            print(field,"has ",str(sum(c)/float(len(c))),"detections for ",len(c),"observations",fieldVal)

def categoricalMapping(df,field,k=5):
    """
    The current version does not create only k groups but
    it does map a category to an integer given by the percentage
    of detections in each category.
    Important that the singleton categories will be merged to avoid 
    overfitting. This function is an effort to keep the variance of the categorical variables
    Please make sure to delete all NaN values beforehand
    #TODO:
    Rewrite the field in the dataframe with values from 0 to k-1 where
    the 0 has less amount of detections while the k-1 has the highest 
    detection rate. 
    

    Params:
      df.......The dataframe that will be modified including the labels as last
               column
      field....The field of the dataframe column name
      k .......[default = 5] The number of values that the answer will map into
    Returns:
      df.......The modified df
      mapping..A mapping dictionary from the old value to the new integer
               value map to each category. This is useful to map the test data
    """
    dictValues = defaultdict(lambda: [0,0])
    t=0
    for value in df[field]:
        #print(value)
        dictValues[value]=(dictValues[value][0] + 1 , dictValues[value][1] + df.iloc[t,-1]) 
        t+=1
    answer=[]
    singletons=[]
    singles=0
    stot=0
    for key,(tot,par) in dictValues.items():
        stot+=tot
        if tot<3:
            singletons.append(key)
            singles+=par
        else:
            answer.append(([key],par/float(tot)))
    if len(singletons):
        answer.append((singletons,singles/len(singletons)))
    lim=0
    
    dictValues = {}
    answer =sorted(answer,key=lambda x:x[1])
    t=0
    for (items,pers) in answer:
        for key in items:
            dictValues[key]=t #int(t/(len(answer)/k))
        t+=1
    finalAnswer=[]
    for x in df[field]:
        finalAnswer.append(dictValues[x])
    if len(finalAnswer) != len(df[field]):
        print('make Sure there are not nan values')
    else:
        df.loc[:,field]=finalAnswer
        return df,dictValues


def preProcessing(trainData,testData,labels):
    """
    This function takes trainData, testData, labels of the train data
    and creates imputers for all data such as mean, median, most_frequent
    and constant imputed values. 
    Following that, it will create dummy columns for some categorical
    values and use the categorical mapping function for discretization 
    for some other attributes.
    It will return a clean train and test data to pass to the classifier
    #TODO:
    Tune the dealings with categorical data to gain more information
    
    Params:
      trainData.......The dataframe containing the training data
      testData .......The dataframe containing the training data
      labels .........The train data labels (HasDetections)
    Returns:
      train_df.......The preprocessed data for training classifier
      test_df .......The preprocessed data for testing classifier
    """
    train_df=trainData.copy()
    test_df=testData.copy()
    train_df.loc[:,'datasource'] = ['train' for x in range(len(train_df['EngineVersion']))]
    test_df.loc[:,'datasource'] = ['test' for x in range(len(test_df['EngineVersion']))]
    test_df.loc[:,'testlabels'] = [0 for x in range(len(test_df['EngineVersion']))]
    train_df.loc[:,'testlabels']=labels
    
    complete_df = pd.concat([train_df, test_df], sort=False)
    completeMapping = {}
    conv2Int=['AVProductStatesIdentifier',
             'DefaultBrowsersIdentifier',
             'AVProductsInstalled',
             'AVProductsEnabled',
              'CityIdentifier',
             'OrganizationIdentifier',
             'GeoNameIdentifier',
             'IsProtected',
             'IeVerIdentifier',
             'Firewall',
             'Census_OEMNameIdentifier',
             'Census_OEMModelIdentifier',
             'Census_ProcessorCoreCount',
             'Census_ProcessorManufacturerIdentifier',
             'Census_ProcessorModelIdentifier']
    categMap={'CountryIdentifier':9,
              'EngineVersion':4,
              'AppVersion':5,
              'AvSigVersion':6,
              'DefaultBrowsersIdentifier':4,
              'AVProductStatesIdentifier':6,
              'CityIdentifier':7,
              'OrganizationIdentifier':6,
              'GeoNameIdentifier':8,
              'LocaleEnglishNameIdentifier':12,
              'OsBuildLab':12,
              'IeVerIdentifier':14,
              'Census_OEMNameIdentifier':10,
              'Census_OEMModelIdentifier':4,
              'Census_ProcessorModelIdentifier':15
             }
    columns2Ignore=['MachineIdentifier','IsBeta','AutoSampleOptIn','PuaMode','SMode','UacLuaenable',
                   'Census_DeviceFamily',
                    'Census_ProcessorClass',
                    'Census_InternalBatteryType',
                    'Census_InternalBatteryNumberOfCharges',
                    'Census_OSVersion',
                   'Census_IsFlightingInternal',
                   'Census_FirmwareVersionIdentifier']
    columns2Dummy=['ProductName',
                   'Platform',
                   'Processor',
                   'OsVer',
                   'OsBuild',
                   'OsSuite',
                   'OsPlatformSubRelease',
                   'SkuEdition',
                   'SmartScreen',
                   'Census_MDC2FormFactor',
                   'Census_PrimaryDiskTypeName',
                   'Census_ChassisTypeName',
                   'Census_PowerPlatformRoleName',
                   'Census_OSArchitecture',
                  'Census_OSBuildNumber',
                  'Census_OSBuildRevision',
                  'Census_OSEdition',
                  'Census_OSBranch',
                   'Census_OSSkuName',
                  'Census_OSInstallTypeName',
                  'Census_OSInstallLanguageIdentifier',
                  'Census_OSWUAutoUpdateOptionsName',
                  'Census_GenuineStateName',
                  'Census_ActivationChannel',
                  'Census_FlightRing',
                   'Wdft_RegionIdentifier',
                   'Census_FirmwareManufacturerIdentifier'
                  ]
    most_frequents = ['Census_OSInstallLanguageIdentifier','AVProductStatesIdentifier','GeoNameIdentifier']
    medians =['Census_TotalPhysicalRAM',
                     'Census_InternalPrimaryDiagonalDisplaySizeInInches',
                     'Census_InternalPrimaryDisplayResolutionHorizontal',
                    'Census_InternalPrimaryDisplayResolutionVertical']
    means = ['Census_SystemVolumeTotalCapacity',
             'Census_PrimaryDiskTotalCapacity']
    imputeAs={'Census_IsFlightsDisabled':1,
              'Census_ThresholdOptIn' :- 1,
              'Census_IsWIMBootEnabled':1,
             'Census_IsVirtualDevice':0.0,
             'Census_IsAlwaysOnAlwaysConnectedCapable':-1,
             'Wdft_IsGamer':2,
             'DefaultBrowsersIdentifier':1,
               'AVProductStatesIdentifier':1,
               'AVProductsEnabled':2.0,
               'AVProductsInstalled':3.0,
             'RtpStateBitfield':7.5,
             'CityIdentifier':0,
             'OrganizationIdentifier':0,
             'IsProtected':-1,
             'IeVerIdentifier':0,
             'SmartScreen':'unknown',
             'Firewall':-1,
             'Census_OEMNameIdentifier':0,
             'Census_OEMModelIdentifier':0,
             'Census_ProcessorCoreCount':0,
             'Census_ProcessorManufacturerIdentifier':0,
             'Census_ProcessorModelIdentifier':0}
    imputer = SimpleImputer(strategy='mean')
    complete_df.loc[:,means]=imputer.fit_transform(complete_df[means])
    imputer = SimpleImputer(strategy='median')
    complete_df.loc[:,medians]=imputer.fit_transform(complete_df[medians])
    imputer = SimpleImputer(strategy='most_frequent')
    complete_df.loc[:,most_frequents]=imputer.fit_transform(complete_df[most_frequents])
    complete_df.loc[:,'Census_OSEdition'] = [x if x in ['Core' ,'CoreCountrySpecific','CoreSingleLanguage', 'Professional'] else 'Other' for x in complete_df['Census_OSEdition'] ] 
    complete_df.loc[:,'Census_OSBranch'] = [x if x in ['rs1_release','rs2_release','rs3_release','rs3_release_svc_escrow','rs4_release','th1_st1','th2_release','th2_release_sec' ] else 'Other' for x in complete_df['Census_OSBranch'] ]
    complete_df.loc[:,'Census_PowerPlatformRoleName'] = [x if x in ['Desktop','Mobile'] else 'Other' for x in complete_df['Census_PowerPlatformRoleName'] ]
    complete_df.loc[:,'Census_ChassisTypeName'] = [x if x in ['AllinOne','Laptop','Notebook','Desktop','Portable'] else 'Other' for x in complete_df['Census_ChassisTypeName'] ]
    complete_df.loc[:,'Census_PrimaryDiskTypeName'] = [x if x in ['HDD','SSD'] else 'Other' for x in complete_df['Census_PrimaryDiskTypeName'] ]

    for field, replaceValue in imputeAs.items():
        imputer = SimpleImputer(strategy='constant',fill_value=replaceValue)
        complete_df.loc[:,field]=imputer.fit_transform(complete_df[[field]])
    for field in conv2Int:
        complete_df.loc[:,field]=[int(x) for x in complete_df[field]]
    complete_df = complete_df.drop(labels=columns2Ignore,axis=1)
    complete_df = dummy_df(df=complete_df,todummy_list = columns2Dummy)
    
    for category, mapping in categMap.items():
        complete_df.loc[complete_df.datasource=='train',:],completeMapping[category]=categoricalMapping(complete_df[complete_df.datasource=='train'],field=category,k=mapping)
        fillTestValues=[]
        for element in complete_df[complete_df.datasource=='test'][category]:
            if element in completeMapping[category]:
                fillTestValues.append(completeMapping[category][element])
            else:
                fillTestValues.append(mapping)
        complete_df.loc[complete_df.datasource=='test',category]=fillTestValues
    train_df=complete_df[complete_df.datasource=='train']
    test_df =complete_df[complete_df.datasource=='test']
    train_df  = train_df.drop(labels=['datasource','testlabels'],axis=1)
    test_df  = test_df.drop(labels=['datasource','testlabels'],axis=1)
    return train_df,test_df

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)

def cross_validation(data,k,file_out=False,filter_columns=False,min_sample=2):
    """ This function takes the raw data and divided into
    k chunks of train and test samples. Given these train
    and test samples, the data is clean in the preprocessing
    function and then the random forest fits the training data
    and gets a prediction for the test data. Finally the average
    accuracy is calculated and returned.
    Params:
      data.......Raw data from the kaggle dataset
      k..........The number of kfold validation
    Returns:
      avg........Average cross validation accuracy among
                 the k-fold
    """
    cv = KFold(n_splits=k)
    accuracies = []
    t=0
    clf=RandomForestClassifier(min_samples_split=min_sample)
    X=data.iloc[:,:-1]
    labels=data.iloc[:,-1]
    for train_idx, test_idx in cv.split(X):
        xlabel=labels[train_idx]
        ylabel=labels[test_idx]
        x,y=preProcessing(X.iloc[train_idx,:],X.iloc[test_idx,:],xlabel)
        if filter_columns:
            with open("columns_saved","w") as ofile:
                for column in [x.columns]:
                    ofile.write(str(column)+"\n")
            clf.fit(x.loc[:,filter_columns], xlabel)
            predicted = clf.predict(y.loc[:,filter_columns])
            acc = accuracy_score(ylabel, predicted)
        else:
            
            clf.fit(x, xlabel)
            predicted = clf.predict(y)
            acc = accuracy_score(ylabel, predicted)
            
        if file_out:
            with open(file_out,'a') as f:
                f.write(str(acc)+"\n")
        accuracies.append(acc)
    avg = np.mean(accuracies)
    #print(accuracies)
    return avg

def countNan(series):
    c=0
    for val in series:
        if str(val) == 'nan':
            c+=1
    return c

    