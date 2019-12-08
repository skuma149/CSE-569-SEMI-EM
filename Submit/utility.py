from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from copy import deepcopy

def cross_validation(label_data,label_targets,classifier,num_fold=5):
    sfold = StratifiedKFold(num_fold)
    count = 1
    accuracies = []
    for train_set,valid_set in sfold.split(label_data,label_targets):

        print("Fold " , count)
        train_data_item = label_data[train_set]
        train_data_target = label_targets[train_set]
        valid_data_item = label_data[valid_set]
        valid_data_target = label_targets[valid_set]

        classifier.fit(train_data_item,train_data_target)

        predicted_labels = classifier.predict(valid_data_item)
        accuracies.append(metrics.accuracy_score(valid_data_target,predicted_labels))
        count+=1
    print("average accuracy",np.mean(np.array(accuracies)))
    return classifier,np.mean(np.array(accuracies))


def getRowsFromMatrix(list_indices,matrix):

    # result = [for i,x in enumerate(matrix) if i in list_indices]
    result = []
    for i,row in enumerate(matrix):
        if(i in list_indices):
            result.append(row)

    return np.array(result)

def cross_validation_EM(label_data,label_targets,unlabelled,classifier,num_fold=5):
    sfold = StratifiedKFold(num_fold)
    count = 1
    accuracies = []
    for train_set,valid_set in sfold.split(label_data,label_targets):

        print("Fold " , count)
        train_data_item = label_data[train_set]
        train_data_target = label_targets[train_set]
        valid_data_item = label_data[valid_set]
        valid_data_target = label_targets[valid_set]

        classifier = classifier.fit(train_data_item,train_data_target,unlabelled)

        predicted_labels = classifier.predict(valid_data_item)
        accuracy = metrics.accuracy_score(valid_data_target,predicted_labels)
        accuracies.append(accuracy)
        print("average accuracy " , accuracy)
        print("log likelihood",classifier.log_lkh)
        count+=1
    return (classifier.clf,classifier.log_lkh)


