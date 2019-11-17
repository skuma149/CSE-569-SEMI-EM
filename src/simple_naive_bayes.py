from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
from sklearn.naive_bayes import MultinomialNB

class simple_bayes:

    def __init__(self,alpha=0.01):
        super().__init__()
        self.bayes_classifier = MultinomialNB(alpha)
        self.train_dataset = fetch_20newsgroups(subset='train',shuffle=True)
        self.test_dataset = fetch_20newsgroups(subset='test',shuffle=True)

        
    def get_accuracy(self):
        vectorizer = TfidfVectorizer()
        train_data_vectors = vectorizer.fit_transform(self.train_dataset.data)
        self.bayes_classifier.fit(train_data_vectors,self.train_dataset.target)
        test_data_vectors = vectorizer.transform(self.test_dataset.data)
        num_test_samples = test_data_vectors.shape[0]
        test_predict = self.bayes_classifier.predict(test_data_vectors)
        not_matched=0
        for i,label in enumerate(test_predict):
            if(label!=self.test_dataset.target[i]):
                not_matched+=1

        return (num_test_samples-not_matched)/num_test_samples

if __name__ == "__main__":
    bayes = simple_bayes()
    print(bayes.get_accuracy())





