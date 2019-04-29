import preprocessing
from sklearn.model_selection import KFold
import importlib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
#data=preprocessing.readData('./microsoft-malware-prediction/train.csv')
with open('./microsoft-malware-prediction/test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    columns={}
    with open("density_graph_data.csv",'w') as ofile:
        for row in csv_reader:
            if line_count == 0:
                ofile.write("AvSigVersion"+"\n")
                t=0
                for field in row:
                    columns[field]=t
                    t+=1
                print("Column names are {}".format(", ".join(row)))
                line_count += 1
            else:
                
                ofile.write(row[columns["AvSigVersion"]] +"\n")
                line_count += 1
    print('Processed {} lines.'.format(line_count))
data=preprocessing.readData('density_graph_data.csv')
byDate= data[["AvSigVersion"]].values.tolist()
signatures=[]
detections=[]
dateCount=defaultdict(lambda : 0)
readingErrors=0
for line in byDate:
    si = line[0].split(".")
    try:
        signature=int(("000"+str(si[1]))[-4:]+("000"+str(si[2]))[-4:])
        dateCount[signature] = dateCount[signature]+1
    except:
        readingErrors+=1
print("OK, just :",readingErrors,"Reading Errors")
    #signatures.append(int(("000"+str(si[1]))[-4:]+("000"+str(si[2]))[-4:]))
    #detections.append(line[1])
#for x in range(len(signatures)):
#    dateCount[signatures[x]]=dateCount[signatures[x]]+1
#    dateSum[signatures[x]]=dateSum[signatures[x]]+detections[x]
plot2=[]
for s,d in dateCount.items():
    plot2.append((s,d))
plotSig=sorted(plot2,key=lambda x:x[0])
################################################################

with open("density_dictionary",'w') as ans_dict:
    ans_dict.write(str(dateCount)+"\n")
yAxis=[y for (x,y) in plot2]
xAxis=[x for (x,y) in plot2]

fig= plt.figure(figsize=(18,10))

axes= fig.add_axes([0.1,0.1,0.8,0.8])

x= xAxis
y=yAxis
plt.xlabel('Time based on AvSigVersion')
plt.ylabel('Density rate')
axes.plot(x,y)
plt.savefig('dsty_graph.png')
plt.show()
