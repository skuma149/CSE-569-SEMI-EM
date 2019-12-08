from nltk.corpus import stopwords, reuters
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import random
from scipy.sparse import coo_matrix, vstack
from utility import getRowsFromMatrix,cross_validation,cross_validation_EM
from scipy import sparse
from sklearn.metrics import accuracy_score
from copy import deepcopy
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
from random import randrange
nltk.download('reuters')
stop_words = stopwords.words("english")

documents = reuters.fileids()
train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                            documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                           documents))

train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

vectorizer = TfidfVectorizer(stop_words=stop_words)

vectorised_train_documents = vectorizer.fit_transform(train_docs)
vectorised_test_documents = vectorizer.transform(test_docs)

mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id)
                                  for doc_id in train_docs_id])
test_labels = mlb.transform([reuters.categories(doc_id)
                             for doc_id in test_docs_id])

copy_train_labels = deepcopy(train_labels)
copy_test_labels = deepcopy(test_labels)
# 10 most popular
max_num_each_categ = np.count_nonzero(train_labels,axis=0)

index_max_class = sorted(range(len(max_num_each_categ)), key=lambda i: max_num_each_categ[i], reverse=True)[:10]


# create labelled & unlabelled dataset in binary representation & word frequency vector
vectorised_train_documents_ndArray = vectorised_train_documents.toarray()
num_lab = 2105
lab_index = random.sample(range(0,train_labels.shape[0]), 2105)

new_lb_train = []
new_vec_lb_train=[]

for i,row in enumerate(train_labels):
    if(i in lab_index):
        new_lb_train.append(row)
        new_vec_lb_train.append(vectorised_train_documents_ndArray[i,:])

new_lb_train = np.array(new_lb_train)
new_vec_lb_train = np.array(new_vec_lb_train)
print("new_lb_train",new_lb_train.shape)
print("new_vec_lb_train",new_vec_lb_train.shape)
new_unlb_train = np.delete(train_labels,lab_index,axis=0)
new_vec_unlb_train = np.delete(vectorised_train_documents_ndArray,lab_index,axis=0)

training_label_set={}
training_data_set={}
# creating 10 training set
for index,max_index in enumerate(index_max_class):
    training_label_set[index+1]=None
    training_data_set[index+1]=None
    # choose positive set
    class_col = new_lb_train[:,max_index]
    doc_index = np.where(class_col==1)[0]
    ran_doc_index = random.sample(doc_index.tolist(), 10)
    training_label_set[index+1] = getRowsFromMatrix(ran_doc_index,new_lb_train)
    new_lb_train = np.delete(new_lb_train,ran_doc_index,axis=0)

    # for vector dataset
    training_data_set[index+1] = getRowsFromMatrix(ran_doc_index,new_vec_lb_train)
    new_vec_lb_train = np.delete(new_vec_lb_train,ran_doc_index,axis=0)

    #  chose negative set
    class_col = new_lb_train[:,max_index]
    doc_index = np.where(class_col!=1)[0]
    ran_doc_index = random.sample(doc_index.tolist(), 40)
    negative_set = getRowsFromMatrix(ran_doc_index,new_lb_train)
    training_label_set[index+1] = np.vstack((training_label_set[index+1],negative_set))
    new_lb_train = np.delete(new_lb_train,ran_doc_index,axis=0)
    training_label_set[index+1] = training_label_set[index+1][:,max_index]

    #for vector dataset
    training_data_set[index+1] = np.vstack((training_data_set[index+1],getRowsFromMatrix(ran_doc_index,new_vec_lb_train)))
    new_vec_lb_train = np.delete(new_vec_lb_train,ran_doc_index,axis=0)

 
# create binary classifiers
binary_classifiers=[]
for index in range(10):
    nb = MultinomialNB(alpha=0.01)
    nb.fit(sparse.csr_matrix(training_data_set[index+1]),training_label_set[index+1])
    binary_classifiers.append(nb)
test_binary_label=[]
for row in vectorised_test_documents:
    generated_label=[]
    for classifier in binary_classifiers:
        generated_label.append((nb.predict(row))[0])
    test_binary_label.append(generated_label)


test_binary_label = np.array(test_binary_label)
#remove all other classes
all_class_index = [item for item in range(0,test_labels.shape[1])]
col_to_delete = [x for x in all_class_index if x not in index_max_class]
test_labels = np.delete(test_labels,col_to_delete,axis=1)

# print("test_binary_label",test_binary_label[:3,:])
# print("test_labels",test_labels[:3,:])
# print(accuracy_score(test_labels[:3,:],test_binary_label[:3,:]))

nb = MultinomialNB(0.01)
one_classifier = OneVsRestClassifier(nb)

one_classifier.fit(vectorised_train_documents,copy_train_labels)

test_predict = one_classifier.predict(vectorised_test_documents)

print(accuracy_score(copy_test_labels,test_predict))
# print(test_predict.shape)

# print(((classifier.estimators_)[0].class_count_))


classes = mlb.classes_
classes = ([classes[item] for item in index_max_class])
print("classes ", classes)
# nb_accuracy={}
# for i in range(10):
#     nb_classifier = one_classifier.estimators_[index_max_class[i]]
#     print("count ",i+1)
#     acc = cross_validation(sparse.csr_matrix(training_data_set[i+1]),training_label_set[i+1],nb_classifier)
#     nb_accuracy[i+1]= acc

# em_accuracy={}
# for i in range(10):
#     nb_classifier = one_classifier.estimators_[index_max_class[i]]
#     em_nb_clf = Semi_EM_MultinomialNB(alpha=0.01,classifier=nb_classifier)
#     print("count ",i+1)
#     acc = cross_validation_EM(sparse.csr_matrix(training_data_set[i+1]),training_label_set[i+1],sparse.csr_matrix(new_vec_unlb_train),em_nb_clf,5)
#     em_accuracy[i+1] = acc

# print(nb_accuracy,em_accuracy)

# print(training_data_set[2][10:].shape)

# # NMF Vectorizer
# from sklearn.decomposition import NMF
# model = NMF(n_components=10, init='random', random_state=0)
# W = model.fit_transform(training_data_set[2][10:])
# H = model.components_


# calculate binary accuracy for each category using naive bayes

# for index,index_val in enumerate(index_max_class):
#     nb_classier = binary_classifiers[index]
#     test_label = copy_test_labels[:,index_val]
#     pred_test_label = nb_classier.predict(vectorised_test_documents)
#     print("category",classes[index])
#     print("accuracy",accuracy_score(test_label,pred_test_label))

# for index,index_val in enumerate(index_max_class):
#     nb_classier = binary_classifiers[index]
#     em_classifier = Semi_EM_MultinomialNB(alpha=1e-2,print_log_lkh=False)
#     (updated_classifier,_) = cross_validation_EM(sparse.csr_matrix(new_vec_lb_train[:100]),new_lb_train[:100,index_val],sparse.csr_matrix(new_vec_unlb_train),em_classifier,5)
#     test_label = copy_test_labels[:,index_val]
#     pred_test_label = updated_classifier.predict(vectorised_test_documents)
#     print("category",classes[index])
#     print("accuracy",accuracy_score(test_label,pred_test_label))

num_z_class = [10,10,15,20,10,20,10,40,3,40]
for i in range(len(classes)):
    print("for class",classes[i])
    num_clusters = num_z_class[i]
    num_train_doc = copy_train_labels.shape[0]
    # step 1 Initial estimate
    z_doc_distr = []
    for k in range(num_train_doc):
        topic_given_class=[]
        for j in range(num_clusters):
            topic_given_class.append(randrange(1,9))
        topic_given_class = np.asarray(topic_given_class)/sum(topic_given_class)
        z_doc_distr.append(topic_given_class)

    z_doc_distr = np.asarray(z_doc_distr)
    num_partition = np.logspace(2.1,3.8,num=9,dtype=int)
    for partition in num_partition:
        label_train_data = z_doc_distr[:partition]
        label_train_label = copy_train_labels[:partition]
        unlabel_train_data = z_doc_distr[partition:]

        z_classifier = MultinomialNB(0.01)
        z_classifier.fit(label_train_data,label_train_label[:,index_max_class[i]])
        count = 0
        prev_log = float("-inf")
        current_log = float("inf")
        while(abs(current_log-prev_log) > 1e-6):

            print("count",count+1)
            #In E step
            unlabel_predict = z_classifier.predict(unlabel_train_data)
            X_new = vstack([label_train_data, unlabel_train_data])
            Y_new = np.concatenate((label_train_label[:,index_max_class[i]], unlabel_predict), axis=0)

            print("In M step")
            z_classifier.fit(X_new,Y_new)

            print("calculate log likelihood") 
            class_log_prior = (z_classifier.class_log_prior_).tolist()
            word_given_class = z_classifier.feature_log_prob_
            class_size = len(z_classifier.class_count_)
            count +=1
            un_sum_outer = 0
            X_new = X_new.toarray()
            for doc in X_new:
                sum_inner = 0
                for index in range(class_size):
                    sum_inner += (class_log_prior[index] * np.sum(word_given_class[index,:]))
                un_sum_outer += sum_inner
            
            lb_sum = 0
            for index,doc in enumerate(X_new):
                sum_inner = 0
                given_label = Y_new[index]
                sum_inner = (class_log_prior[given_label]* np.sum(word_given_class[given_label,:]))
                lb_sum += sum_inner

            log_likelihood = (-1 * (lb_sum + un_sum_outer))
            prev_log = current_log
            current_log = log_likelihood
            print("log_likelihood ",log_likelihood)






 
 
