from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import coo_matrix, vstack
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from utility import cross_validation,cross_validation_EM
from EM import Semi_NB
from SEMI_NB import Semi_EM_MultinomialNB
from sklearn.metrics import accuracy_score

class EM_NB:

    def __init__(self,alpha=0.01):
        super().__init__()
        all_dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=True)
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(all_dataset.data, all_dataset.target, test_size=0.2, stratify=all_dataset.target)
        self.vectorizer = TfidfVectorizer()

    def perform_EM(self,X_l,y_l,X_u):
        nb_clf = MultinomialNB(alpha=0.01)
        nb_clf.fit(X_l,y_l)
        
        # calculate log likelihood 
        class_log_prior = (nb_clf.class_log_prior_).tolist()
        word_given_class = nb_clf.feature_log_prob_
        class_size = len(nb_clf.class_count_)
        un_sum_outer = 0
        for doc in X_u:
            sum_inner = 0
            for index in range(class_size):
                sum_inner += (class_log_prior[index] * np.sum(word_given_class[index,:]))
            un_sum_outer += sum_inner
        
        lb_sum = 0
        for index,doc in enumerate(X_l):
            sum_inner = 0
            given_label = y_l[index]
            sum_inner = (class_log_prior[given_label]* np.sum(word_given_class[given_label,:]))
            lb_sum += sum_inner

        log_likelihood = (-1 * (lb_sum + un_sum_outer))
        prev_log = float("-inf")
        current_log = log_likelihood
        count = 0
        # remove this line
        while(abs(current_log-prev_log) > 1e-6):
            # Estimation step
            Y_u = nb_clf.predict(X_u)

            # Maximize step
            X_new = vstack([X_l, X_u])
            Y_new = np.concatenate((y_l, Y_u), axis=0)
            nb_clf.fit(X_new, Y_new)

            # calculate log likelihood 
            class_log_prior = (nb_clf.class_log_prior_).tolist()
            word_given_class = nb_clf.feature_log_prob_
            class_size = len(nb_clf.class_count_)
            count +=1
            un_sum_outer = 0
            for doc in X_u:
                sum_inner = 0
                for index in range(class_size):
                    sum_inner += (class_log_prior[index] * np.sum(word_given_class[index,:]))
                un_sum_outer += sum_inner
            
            lb_sum = 0
            for index,doc in enumerate(X_l):
                sum_inner = 0
                given_label = y_l[index]
                sum_inner = (class_log_prior[given_label]* np.sum(word_given_class[given_label,:]))
                lb_sum += sum_inner

            log_likelihood = (-1 * (lb_sum + un_sum_outer))
            prev_log = current_log
            current_log = log_likelihood
            print("log_likelihood ",log_likelihood)
        return nb_clf
            
        
        
    def get_accuracy(self,use="NB"):
        num_labeled_doc = np.logspace(2.1,3.8,num=9,dtype=int)
        nb_accuracy_score = []
        num_labeled = []
        em_nb_log_lik=[]
        print("Using ",use)
        for num in num_labeled_doc:
            train_data_vectors = self.vectorizer.fit_transform(self.train_X)
            # train_label_vectors = self.labeled_dataset_target[:num]
            # unlabeled_label_vectors = self.vectorizer.fit_transform(self.unlabeled_dataset)
            split_ratio = num # labeled vs unlabeled
            X_l, X_u, y_l, y_u = train_test_split(train_data_vectors, self.train_Y, train_size=10000, stratify=self.train_Y)
            num_labeled.append(train_data_vectors[:num].shape[0])
            if(use=="NB"):
                nb_clf = MultinomialNB(0.01)
                nb_clf,accuracy = cross_validation(train_data_vectors[:num],self.train_Y[:num],nb_clf,5)
                nb_accuracy_score.append(accuracy)
            else:
                # em_nb_clf = Semi_EM_MultinomialNB(alpha=1e-2,print_log_lkh=True)
                # only for EM
                em_nb_clf = Semi_NB()
                print("train",(train_data_vectors[:num]).shape)
                print("unlabeled",(train_data_vectors[num:]).shape)
                (nb_clf,em_log_lkh,accuracy) = cross_validation_EM(train_data_vectors[:num], self.train_Y[:num], train_data_vectors[num:],em_nb_clf,5)
                nb_accuracy_score.append(accuracy)
                em_nb_log_lik.append(em_log_lkh)

            # Test Accuracy
            test_data_vector = self.vectorizer.transform(self.test_X)
            test_predict = nb_clf.predict(test_data_vector)
            print("accuracy", accuracy_score(self.test_Y,test_predict))
            
        return (num_labeled,nb_accuracy_score)

        
        

if __name__ == "__main__":
    bayes = EM_NB()
    print(bayes.get_accuracy("EM"))