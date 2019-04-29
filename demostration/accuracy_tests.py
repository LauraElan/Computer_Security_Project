import preprocessing
import matplotlib.pyplot as plt
data=preprocessing.readData('train_samples.csv')
outFile='demostration_accuracy'
preprocessing.cross_validation(data=data,k=5,file_out=outFile,min_sample=127)

fig= plt.figure(figsize=(18,9))

axes= fig.add_axes([0.1,0.1,0.8,0.8])

x= [1,2,3,4,5]
y=[]
avg=[]
with open(outFile,'r') as inFile:
    for accuracy in inFile:
        y.append(float(accuracy))
plt.title("Demostration Accuracy")
plt.xlabel('Trial')
plt.ylabel('Accuracy')
a=sum(y)/len(y)
avg=[a for t in range(5)]
axes.plot(x,y)
axes.plot(x,avg)
plt.savefig('Demostration_graph.png')
plt.show()
