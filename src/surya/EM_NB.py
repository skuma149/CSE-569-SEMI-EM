from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import coo_matrix, vstack
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

class EM_NB:

    def __init__(self,alpha=0.01):
        super().__init__()
        self.bayes_classifier = MultinomialNB(alpha)
        self.train_dataset = fetch_20newsgroups(subset='train',shuffle=True)
        self.test_dataset = fetch_20newsgroups(subset='test',shuffle=True)

        self.labeled_dataset = (self.train_dataset.data[:2200])
        # self.labeled_dataset.extend(self.test_dataset.data[:4500])
        # print(len(self.labeled_dataset))
        self.labeled_dataset_target = (self.train_dataset.target[:2200]).tolist()
        # self.labeled_dataset_target.extend(self.test_dataset.data[:4500])
        self.unlabeled_dataset = self.train_dataset.data[2200:]


        
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
            un_sum_outer += np.log(sum_inner)
        
        lb_sum = 0
        for index,doc in enumerate(X_l):
            sum_inner = 0
            given_label = y_l[index]
            sum_inner = (class_log_prior[given_label]* np.sum(word_given_class[given_label,:]))
            lb_sum += sum_inner

        log_likelihood = lb_sum + un_sum_outer
        prev_likelihood = 0
        new_likelihood = 0
        count = 0
        while(count < 5):
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
                un_sum_outer += np.log(sum_inner)
            
            lb_sum = 0
            for index,doc in enumerate(X_l):
                sum_inner = 0
                given_label = y_l[index]
                sum_inner = (class_log_prior[given_label]* np.sum(word_given_class[given_label,:]))
                lb_sum += sum_inner

            log_likelihood = lb_sum + un_sum_outer
            prev_likelihood = new_likelihood
            new_likelihood = log_likelihood
            print("difference " , new_likelihood - prev_likelihood)
            print("log_likelihood ",log_likelihood)
            
                


        
        
        
    def get_accuracy(self):

        count = 1
        accuracy = 0
        total_number = len(self.train_dataset.data)- 1000
        num_labeled_doc = [0.2,0.5]
        for i,num in enumerate(num_labeled_doc):
            self.bayes_classifier = MultinomialNB(0.01)

            train_data_vectors = self.vectorizer.fit_transform(self.train_dataset.data)
            # train_label_vectors = self.labeled_dataset_target[:num]
            # unlabeled_label_vectors = self.vectorizer.fit_transform(self.unlabeled_dataset)

            split_ratio = 0.2 # labeled vs unlabeled
            X_l, X_u, y_l, y_u = train_test_split(train_data_vectors, self.train_dataset.target, train_size=split_ratio, stratify=self.train_dataset.target)

            em_nb_clf = self.perform_EM(X_l, y_l, X_u)
            # Test Accuracy
            test_data_vector = self.vectorizer.transform(self.test_dataset.data[:])
            num_test_samples = len(self.test_dataset.data[:])
            test_predict = self.bayes_classifier.predict(test_data_vector)
            not_matched=0
            for i,label in enumerate(test_predict):
                if(label!=self.test_dataset.target[:][i]):
                    not_matched+=1

            accuracy =  (num_test_samples-not_matched)/num_test_samples
            print("num ",num,"accuracy " , accuracy)
            total_number-=1000
            count+=1
            
        return accuracy

        
        

if __name__ == "__main__":
    bayes = EM_NB()
    bayes.get_accuracy()