import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import normalize

dataList=pd.read_csv("trainData.csv")
labelsList=pd.read_csv("trainLabels.csv")

tdata=[]

for i in range(len(list(dataList.iterrows()))):
	tdata.append(dataList.ix[i].values.tolist())



for i in range(12019):
	for j in range(784):
		tdata[i][j]=(float(tdata[i][j])-0)/(255-0)




tlabels=[]

for i in range(len(list(labelsList.iterrows()))):
	tlabels.append(labelsList.ix[i].values.tolist())

from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=3)

#Training data
xtrain=tdata
train_labels=tlabels


classifier.fit(xtrain,train_labels)



#test data
xtest=pd.read_csv("testData.csv")

normalize(xtest)


testdata=[]

for i in range(len(list(xtest.iterrows()))):
	testdata.append(xtest.ix[i].values.tolist())


for i in range(5151):
	for j in range(784):
		testdata[i][j]=(float(testdata[i][j])-0)/(255-0)

# #real labels
# labels=pd.read_csv("kaggleTestSubsetLabels.csv")


# testlabels=[]

# for i in range(len(list(labels.iterrows()))):
# 	testlabels.append(labels.ix[i].values.tolist())




p=[]

for i in range(0,5151):
	temp=[]

	temp.append([])

	temp[0].append(testdata[i])
	p.append([i+1,classifier.predict(temp[0])[0]])




# count=0

# for i in range(0,2575):
# 	count+=1 if p[i][1]==testlabels[i][0] else 0

# print(count)


csvfile = "154055knn3a.csv"

i=0
with open(csvfile, "wb") as output:
    writer = csv.writer(output)
    if(i==0):
        writer.writerow(["ID","Label"])
    i+=1
    writer.writerows(p)