import preprocessing
data=preprocessing.readData('train_samples500.csv')
while True:
    preprocessing.cross_validation(data=data,k=5,file_out='accuracy_test_500k')