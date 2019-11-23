from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import coo_matrix, vstack

class EM_NB:

    def __init__(self,alpha=0.01):
        super().__init__()
        self.bayes_classifier = MultinomialNB(alpha)
        self.train_dataset = fetch_20newsgroups(subset='train',shuffle=True)
        self.test_dataset = fetch_20newsgroups(subset='test',shuffle=True)

        self.labeled_dataset = (self.train_dataset.data[:5500])
        # self.labeled_dataset.extend(self.test_dataset.data[:4500])
        # print(len(self.labeled_dataset))
        self.labeled_dataset_target = (self.train_dataset.target[:5500]).tolist()
        # self.labeled_dataset_target.extend(self.test_dataset.data[:4500])
        self.unlabeled_dataset = self.train_dataset.data[5500:]

        self.train_dataset_mod = (self.test_dataset.data[4500:])
        self.train_dataset_label = (self.test_dataset.data[4500:])
        
        self.vectorizer = TfidfVectorizer()

    def perform_EM(self,train_data_vectors,train_data_label):
        # Estimation step
        unlabeled_data_vectors = self.vectorizer.transform(self.train_dataset.data[1000:])
        test_predict = self.bayes_classifier.predict(unlabeled_data_vectors)

        # Maximize 
        new_labeled_target = train_data_label
        new_labeled_target.extend(test_predict)
        total_dataset_target = np.asarray(new_labeled_target)
        train_data_vectors = vstack([train_data_vectors,unlabeled_data_vectors]).toarray()
        self.bayes_classifier.fit(train_data_vectors,total_dataset_target)
        
        
    def get_accuracy(self):

        count = 1
        accuracy = 0
        total_number = len(self.train_dataset.data)- 1000
        num_labeled_doc = [20,30,50,70,85,100,200,350,500,650,750,900,1000,1500,2000,5000,5500]
        for i,num in enumerate(num_labeled_doc):
            self.bayes_classifier = MultinomialNB(0.01)
            train_data_vectors = self.vectorizer.fit_transform(self.labeled_dataset[:num])
            train_label_vectors = self.labeled_dataset_target[:num]
            self.bayes_classifier.fit(train_data_vectors,train_label_vectors)
            self.perform_EM(train_data_vectors,train_label_vectors)
            # Test Accuracy
            test_data_vector = self.vectorizer.transform(self.test_dataset.data[4500:])
            num_test_samples = len(self.test_dataset.data[4500:])
            test_predict = self.bayes_classifier.predict(test_data_vector)
            not_matched=0
            for i,label in enumerate(test_predict):
                if(label!=self.test_dataset.target[4500:][i]):
                    not_matched+=1

            accuracy =  (num_test_samples-not_matched)/num_test_samples
            print("num ",num,"accuracy " , accuracy)
            total_number-=1000
            count+=1
            
        return accuracy

        
        

if __name__ == "__main__":
    bayes = EM_NB()
    print(bayes.get_accuracy())