import preprocessing
from sklearn.model_selection import KFold
import importlib
from sklearn.ensemble import RandomForestClassifier

data=preprocessing.readData('train_samples.csv')

cv = KFold(n_splits=2)

t=0
clf=RandomForestClassifier()
X=data.iloc[:,:-1]
labels=data.iloc[:,-1]
from itertools import combinations
fields=['Processor','Census_OSVersion','OsBuild','SkuEdition', \
             'Census_OSWUAutoUpdateOptionsName','Census_GenuineStateName', \
             'AppVersion','AvSigVersion']
categMap={'CountryIdentifier':9,
          'EngineVersion':4,
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
          'Census_ProcessorModelIdentifier':15}
fields2filter = []
for key, _ in categMap.items():
    fields2filter.append(key)
for field in fields:
    fields2filter.append(field)

    column_filtering=[]
    
    
for train_idx ,test_idx in cv.split(X):
    xlabel=labels[train_idx]
    ylabel=labels[test_idx]
    x,y=preprocessing.preProcessing(X.iloc[train_idx,:],X.iloc[test_idx,:],xlabel)
    for combs in combinations(fields2filter,5):
        t=0
        cols = list(combs)
        if not t:
            column_filtering=[]
            for column in x.columns:
                for col in cols:
                    if col in column:
                        column_filtering.append(column)
        clf.fit(x.loc[:,column_filtering], xlabel)
        predicted = clf.predict(y.loc[:,column_filtering])
        acc = preprocessing.accuracy_score(ylabel, predicted)
        #accuracies.append(acc)
        with open("accuracies_per_fields","a") as ofile:
            ofile.write(str(acc)+"; "+str(column_filtering)+"\n")