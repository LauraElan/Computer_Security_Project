import preprocessing
from sklearn.model_selection import KFold
import importlib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
#data=preprocessing.readData('./microsoft-malware-prediction/train.csv')
with open('./microsoft-malware-prediction/train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    columns={}
    with open("graph_data.csv",'w') as ofile:
        for row in csv_reader:
            if line_count == 0:
                ofile.write("AvSigVersion"+ "," +"HasDetections"+"\n")
                t=0
                for field in row:
                    columns[field]=t
                    t+=1
                print("Column names are {}".format(", ".join(row)))
                line_count += 1
            else:
                
                ofile.write(row[columns["AvSigVersion"]] + "," + row[columns["HasDetections"]]+"\n")
                line_count += 1
    print('Processed {} lines.'.format(line_count))
data=preprocessing.readData('graph_data.csv')
byDate= data[["AvSigVersion",'HasDetections']].values.tolist()
signatures=[]
detections=[]
dateCount=defaultdict(lambda : 0)
dateSum=defaultdict(lambda : 0)
readingErrors=0
for line in byDate:
    si = line[0].split(".")
    try:
        signature=int(("000"+str(si[1]))[-4:]+("000"+str(si[2]))[-4:])
        dateCount[signature] = dateCount[signature]+1
        dateSum[signature]=dateSum[signature]+line[1]
    except:
        readingErrors+=1
print("OK, just :",readingErrors,"Reading Errors")
    #signatures.append(int(("000"+str(si[1]))[-4:]+("000"+str(si[2]))[-4:]))
    #detections.append(line[1])
#for x in range(len(signatures)):
#    dateCount[signatures[x]]=dateCount[signatures[x]]+1
#    dateSum[signatures[x]]=dateSum[signatures[x]]+detections[x]
plotSig=[]
plot2=[]
for s,d in dateCount.items():
    plotSig.append((s,dateSum[s]/dateCount[s]))
    plot2.append((s,d))
plotSig=sorted(plotSig,key=lambda x:x[0])
answers={}
for key,value in plotSig:
    answers[key]=value
with open("answer_dictionary",'w') as ans_dict:
    ans_dict.write(str(answers)+"\n")
yAxis=[y for (x,y) in plotSig]
xAxis=[x for x in range(len(plotSig))]

fig= plt.figure(figsize=(18,10))

axes= fig.add_axes([0.1,0.1,0.8,0.8])

x= xAxis

y=yAxis
plt.xlabel('Time based on AvSigVersion')
plt.ylabel('detection rate')
axes.plot(x,y)
plt.savefig('time_graph.png')
plt.show()
################################################################

with open("density_dictionary",'w') as ans_dict:
    ans_dict.write(str(dateCount)+"\n")
yAxis=[y for (x,y) in plot2]
xAxis=[x for x in range(len(plot2))]

fig= plt.figure(figsize=(18,10))

axes= fig.add_axes([0.1,0.1,0.8,0.8])

x= xAxis
y=yAxis
plt.xlabel('Time based on AvSigVersion')
plt.ylabel('Density rate')
axes.plot(x,y)
plt.savefig('density_graph.png')
plt.show()
