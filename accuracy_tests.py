import preprocessing
data=preprocessing.readData('train_samples.csv')
t=2
outFile='prunning_test'
while t<30:
    preprocessing.cross_validation(data=data,k=5,file_out=outFile,min_sample=t)
    t+=1
    with  open(outFile,'a') as ofile:
        ofile.write(str(t)+"\n")