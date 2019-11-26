from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import coo_matrix, vstack
from SEMI_NB import Semi_EM_MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

class EM_NB:

    def __init__(self,alpha=0.01):
        super().__init__()
        self.bayes_classifier = MultinomialNB(alpha)
        self.train_dataset = fetch_20newsgroups(subset='train',shuffle=True)
        self.test_dataset = fetch_20newsgroups(subset='test',shuffle=True)

        self.labeled_dataset = (self.train_dataset.data[:2262])
        # self.labeled_dataset.extend(self.test_dataset.data[:4500])
        # print(len(self.labeled_dataset))
        self.labeled_dataset_target = (self.train_dataset.target[:2262]).tolist()
        # self.labeled_dataset_target.extend(self.test_dataset.data[:4500])
        self.unlabeled_dataset = self.train_dataset.data[2262:]

        
        self.vectorizer = TfidfVectorizer()        
        
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

            em_nb_clf = Semi_EM_MultinomialNB(alpha=1e-2,print_log_lkh=True).fit(X_l, y_l, X_u)

            print("Used EM")
            self.bayes_classifier.fit(X_l,y_l)
            # Test Accuracy
            test_data_vector = self.vectorizer.transform(self.test_dataset.data)
            num_test_samples = len(self.test_dataset.data)
            test_predict = self.bayes_classifier.predict(test_data_vector)
            not_matched=0
            for i,label in enumerate(test_predict):
                if(label!=self.test_dataset.target[i]):
                    not_matched+=1

            accuracy =  (num_test_samples-not_matched)/num_test_samples
            print("num ",num,"accuracy " , accuracy)
            total_number-=1000
            count+=1
            
        return accuracy

        
        

if __name__ == "__main__":
    bayes = EM_NB()
    print(bayes.get_accuracy())