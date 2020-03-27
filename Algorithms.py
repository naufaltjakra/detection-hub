import pandas as pd
import numpy as np
import pickle

#Import dataset
dataset = pd.read_csv('dataset_small.csv')

X = dataset.iloc[:, 0:16].values
y = dataset.iloc[:, 16].values

data_feature_names = [ 'IP', 'URL Length', 'A Symbol', 'Prefix Suffix', 'Sub Domain', 'HTTPS Token',
                    'Request URL', 'URL Anchor', 'SFH', 'Abnormal URL', 'Redirect', 'On Mouseover',
                    'Pop Up Window', 'Age of Domain', 'DNS Record', 'Web Traffic' ]

#Train Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

### Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=121, criterion='entropy', bootstrap=True, max_depth=8)
rf_classifier.fit(X_train, y_train)
rf_y_pred = rf_classifier.predict(X_test)

### Neural Network
from sklearn.neural_network import MLPClassifier
nn_classifier = MLPClassifier()
nn_classifier.fit(X_train, y_train)
nn_y_pred = nn_classifier.predict(X_test)

### Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_y_pred = nb_classifier.predict(X_test)

#Pickle
# pickle.dump(rf_classifier, open('model_RF.pkl','wb'))
# pickle.dump(nn_classifier, open('model_NN.pkl','wb'))
# pickle.dump(nb_classifier, open('model_NB.pkl','wb'))

from sklearn.model_selection import cross_val_score
#Random Forest Accuracy in K-Fold
rf_accuracy = cross_val_score(estimator = rf_classifier, X = X_train, y = y_train, cv = 5)
#Neural Network Accuracy in K-Fold
nn_accuracy = cross_val_score(estimator = nn_classifier, X = X_train, y = y_train, cv = 5)
#Naive Bayes Accuracy in K-Fold
nb_accuracy = cross_val_score(estimator = nb_classifier, X = X_train, y = y_train, cv = 5)


#Random Forest Detailed Accuracy
k_rf_acc = rf_accuracy*100
rf_mean_acc = '%.3f%%' % (sum(rf_accuracy)/float(len(rf_accuracy))*100)
rf_max_acc = 'Maximum Accuracy: %.3f%%' % max(rf_accuracy*100)
rf_min_acc = 'Minimum Accuracy: %.3f%%' % min(rf_accuracy*100)
print(" ")
print("### RANDOM FOREST ###")
rf_num = 1
for i in k_rf_acc:
    print(f"Fold {rf_num} : {i}")
    rf_num += 1
print(f"Mean Accuracy : {rf_mean_acc}")
print(rf_max_acc)
print(rf_min_acc)
print(" ")

#Neural Network Detailed Accuracy
k_nn_acc = nn_accuracy*100
nn_mean_acc = '%.3f%%' % (sum(nn_accuracy)/float(len(nn_accuracy))*100)
nn_max_acc = 'Maximum Accuracy: %.3f%%' % max(nn_accuracy*100)
nn_min_acc = 'Minimum Accuracy: %.3f%%' % min(nn_accuracy*100)
print(" ")
print("### NEURAL NETWORK ###")
nn_num = 1
for i in k_nn_acc:
    print(f"Fold {nn_num} : {i}")
    nn_num += 1
print(f"Mean Accuracy : {nn_mean_acc}")
print(nn_max_acc)
print(nn_min_acc)
print(" ")

#Naive Bayes Detailed Accuracy
k_nb_acc = nb_accuracy*100
nb_mean_acc = '%.3f%%' % (sum(nb_accuracy)/float(len(nb_accuracy))*100)
nb_max_acc = 'Maximum Accuracy: %.3f%%' % max(nb_accuracy*100)
nb_min_acc = 'Minimum Accuracy: %.3f%%' % min(nb_accuracy*100)
print(" ")
print("### NAIVE BAYES ###")
nb_num = 1
for i in k_nb_acc:
    print(f"Fold {nb_num} : {i}")
    nb_num += 1
print(f"Mean Accuracy : {nb_mean_acc}")
print(nb_max_acc)
print(nb_min_acc)
print(" ")

# # Confusion matrix detail
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))