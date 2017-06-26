# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')
# Importing the dataset
training_set = pd.read_csv('datatraining.txt')
test_set = pd.read_csv('datatest.txt')


# In[111]:

training_set.head()


# In[112]:

training_set.describe()


# The training set data has 8143 rows and 7 columns. Occupancy is the Dependent Variable i.e y varaible. The other features except date forms the set of Independent Variables.
# 
# Also , there are no NA's and NaN's present.

# ## Data Preprocessing

# In[113]:

#Splitting training set
X_train = training_set.iloc[:,1:6].values
y_train = training_set.iloc[:,6].values
#Splitting test set
X_test = test_set.iloc[:,1:6].values
y_test = test_set.iloc[:,6].values


# Now visualize a pairplot of the training set data.

# In[114]:

sb.pairplot(training_set, hue='Occupancy',palette="husl",markers=["o", "s"])


# There are some Outliers in the Light variable. Lets remove them.

# In[115]:

training_set = training_set.loc[(training_set['Light'] <=750)]
sb.pairplot(training_set, hue='Occupancy',palette="husl",markers=["o", "s"])


# Now proceed to feature scaling

# In[116]:

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Building a Classification model

# In[118]:

# Fit a Logistic Regression for classification
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[119]:

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Printing the confusion matrix
from sklearn.metrics import confusion_matrix
cm_logit = confusion_matrix(y_test, y_pred)
cm_logit


# From the confusion matrix, evaluation parameters such as Accuracy, Precision, Recall and F1 Score are to be evaluated. Before that, lets fit other classification methods to the same data.

# In[120]:

# Fitting KNN CLASSIFIER to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred)
cm_knn


# In[121]:

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, y_pred)
cm_svm


# In[122]:

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_KSVM = confusion_matrix(y_test, y_pred)
cm_KSVM


# In[123]:

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_NB = confusion_matrix(y_test, y_pred)
cm_NB


# In[124]:

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_dtree = confusion_matrix(y_test, y_pred)
cm_dtree


# In[125]:

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RF = confusion_matrix(y_test, y_pred)
cm_RF


# Create a list of Confusion Matrices and compute model evaluation metrics.

# In[126]:

cmlist = [cm_logit,cm_knn,cm_svm,cm_KSVM,cm_NB,cm_dtree,cm_RF]


# In[127]:

Accuracy = []
Precision = []
Recall =[]
F1Score =[]


# In[128]:

for i in range(7):
    temp = cmlist[i] 
    Accuracy.append((temp[0,0]+temp[1,1])/2665)
    Precision.append(temp[1,1]/(temp[1,1]+temp[0,1]))
    Recall.append(temp[1,1]/(temp[1,1]+temp[1,0]))
    F1Score.append(2*Precision[i]*Recall[i]/(Precision[i]+Recall[i]))


# In[129]:

Algorithm = ['Logistic Regression','KNN ','SVM (Linear)','SVM(Kernel)','Naive Bayes','Decision Tree', 'Random Forests']


# In[130]:

df = pd.DataFrame.from_items([('Algorithm',Algorithm), ('Accuracy' , Accuracy), ('Precision' ,Precision), 
                              ('Recall' , Recall), ('F1 Score', F1Score)])
df
