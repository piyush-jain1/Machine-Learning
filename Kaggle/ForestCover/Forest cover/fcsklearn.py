import pandas as pd
import numpy as np
from numpy import *
from sklearn import linear_model
import csv
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB




## loading the dataset
dataset = pd.read_csv('train.csv', sep=',')

## id or serial number is nt any useful attribute or tels us info about the cover
dataset = dataset.drop('Id',axis=1)

## sklearn takes input as array so have to convert dataframe into 2-d array 
data = array(dataset)


### slicing the array 
X = data[:,:-1]
Y = data[:,-1]
print(Y)


### standard preprocessor
# scaler = preprocessing.StandardScaler().fit(X)
# X = scaler.transform(X)                               
# print(X)
# X = X.astype(int)
# print(X)


# ## standard model object 
# logreg = linear_model.LogisticRegression(penalty='l2',C=1e5)

# ## fitting the model
# logreg.fit(X, Y)


# ## similar preprocessing for 
# test_data = pd.read_csv('test.csv', sep=',')
# test_data = test_data.drop('Id',axis=1)
# test_data = array(test_data)

# ## applying scaling to test set
# # test_data = scaler.transform(test_data)                


# ## predicting labels for output values
# y_test = logreg.predict(test_data)


# ## writing files as standard outputs
# with open('predict.csv', 'w') as csvfile:
#     fieldnames = ['Id', 'Cover_type']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     i = 15121
#     for point in y_test:
# 		writer.writerow({'Id':str(i) , 'Cover_type': str(point)})
# 		i=i+1





#names of all the columns
cols = dataset.columns

#print(cols.values.tolist()[:10])
#number of rows=r , number of columns=c
r,c = dataset.shape

#Create a new dataframe with r rows, one column for each encoded category, and target in the end
data = pd.DataFrame(index=np.arange(0, r),columns=cols.values.tolist()[:10]+['Wilderness_Area','Soil_Type','Cover_Type'])

#Make an entry in 'data' for each r as category_id, target value
for i in range(0,r):
    w=0;
    s=0;
    # Category1 range
    print(i)
    for j in range(0,10):
    	data.iloc[i,j]=dataset.iloc[i,j]

    for j in range(10,14):
        if (dataset.iloc[i,j] == 1):
            data.iloc[i,10] = j-9
            break
    

    # Category2 range        
    for k in range(14,54):
        if (dataset.iloc[i,k] == 1):
            data.iloc[i,11] = k-13
            break
    #Make an entry in 'data' for each r as category_id, target value        
    data.iloc[i,12]=dataset.iloc[i,c-1]

print(data.shape)

#print("testing other models")
# random_forest = RandomForestClassifier(n_estimators=40)
# random_forest.fit(X, Y)
# y_test = random_forest.predict(test_data)

# svm = SVC()
# svm.fit(X, Y)
# y_test = svm.predict(test_data)