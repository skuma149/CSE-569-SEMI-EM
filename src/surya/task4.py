from nltk.corpus import stopwords, reuters
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import random
from utility import getRowsFromMatrix
from scipy import sparse
from sklearn.metrics import accuracy_score

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

print("test_binary_label",test_binary_label[:2,:])
print("test_labels",test_labels[:2,:])
print(accuracy_score(test_labels[:2,:],test_binary_label[:2,:]))

    

# print(test_predict.shape)

# print(((classifier.estimators_)[0].class_count_))

 
