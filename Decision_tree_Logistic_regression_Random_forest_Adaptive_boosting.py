import os 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
os.chdir("/Users/Asus UX310/Desktop/Project")
data=pd.read_csv("data/UCI_Credit_Card.csv")
data.head()
data.info()
data = data.rename(columns={'PAY_0': 'PAY_1'})

#create X and y before splitting for test and train
y = data['default.payment.next.month'].copy()
X = data.drop(columns='default.payment.next.month')
X.columns

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=123)


# decision tree classifier
# first create model and fit to the training dataset
classifier = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=10,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=30,
            min_weight_fraction_leaf=0.0, presort=False, random_state=123,
            splitter='best')
classifier.fit(X_train, y_train)
# see the performance on the testing dataset
predictions = classifier.predict(X_test)
accuracy_score(y_true = y_test, y_pred = predictions)
rmse_test=MSE(y_test, predictions)**(1/2)
print ('Test set RMSE of fr: {:.2f}'.format(rmse_test))


# logistic regression
from sklearn.linear_model import LogisticRegression
logistic=LogisticRegression(random_state=123)
logistic.fit(X_train, y_train)
predictions = logistic.predict(X_test)
accuracy_score(y_true = y_test, y_pred = predictions)
rmse_test=MSE(y_test, predictions)**(1/2)
print ('Test set RMSE of fr: {:.2f}'.format(rmse_test))


#random forest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=500, min_samples_leaf=1, random_state=123)
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
accuracy_score(y_true = y_test, y_pred = predictions)
rmse_test=MSE(y_test, y_pred)**(1/2)
print ('Test set RMSE of fr: {:.2f}'.format(rmse_test))


# adaptive boosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
adb_clf=AdaBoostClassifier(base_estimator=classifier, n_estimators=100)
adb_clf.fit(X_train, y_train)
y_pred=adb_clf.predict(X_test)
accuracy_score(y_true = y_test, y_pred = predictions)
rmse_test=MSE(y_test, y_pred)**(1/2)
print ('Test set RMSE of fr: {:.2f}'.format(rmse_test))
# roc auc score
y_pred_proba=adb_clf.predict_proba(X_test)[:,1]
adb_clf_roc_auc_score=roc_auc_score(y_test, y_pred_proba)
print('ROC AUC SCORE: {:.2f}'.format(adb_clf_roc_auc_score))


























































































