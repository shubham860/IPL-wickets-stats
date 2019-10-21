import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv('deliveries.csv')

y = dataset.iloc[:,19].values
X = dataset.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20]].values


temp = pd.DataFrame(X[:,[18,19]])

temp[0] = temp[0].fillna("No-wicket")
temp[1] = temp[1].fillna("No-wicket")

X[:,[18,19]] = temp
del(temp)

temp2 = pd.DataFrame(y)

temp2[0] = temp2[0].fillna("No-wicket")
y = temp2
del(temp2)

le = LabelEncoder()

X[:,2] = le.fit_transform(X[:,2])
X[:,3] = le.fit_transform(X[:,3])
X[:,6] = le.fit_transform(X[:,6])
X[:,7] = le.fit_transform(X[:,7])
X[:,8] = le.fit_transform(X[:,8])
X[:,18] = le.fit_transform(X[:,18])
X[:,19] = le.fit_transform(X[:,19])

ohe = OneHotEncoder(categorical_features=[2,3,6,7,8,18,19])
X = ohe.fit_transform(X)
X = X.toarray()

y = le.fit_transform(y)

std = StandardScaler()
X = std.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc.score(X_train,y_train)
dtc.score(X_test,y_test)



from sklearn.metrics import confusion_matrix
cnn = confusion_matrix(y,y_pred)

from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y,y_pred)
recall_score(y,y_pred)
f1_score(y,y_pred)




























