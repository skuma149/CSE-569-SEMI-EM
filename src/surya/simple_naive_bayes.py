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

        print(self.train_dataset.filenames.shape)
        print(self.train_dataset.target.shape)


        
    def get_accuracy(self):
        vectorizer = TfidfVectorizer()
        num_labeled_doc = [20,30,50,70,85,100,200,350,500,650,750,900,1000,1500,2000,5000,5500]
        for i,num in enumerate(num_labeled_doc):
            self.bayes_classifier = MultinomialNB(0.01)
            train_data_vectors = vectorizer.fit_transform(self.train_dataset.data[:num])
            self.bayes_classifier.fit(train_data_vectors,self.train_dataset.target[:num])

            print("class log prior ",len(self.bayes_classifier.class_log_prior_))
            # print("feature log probability ",self.bayes_classifier.feature_log_prob_)
            # break
            test_data_vectors = vectorizer.transform(self.test_dataset.data)
            num_test_samples = test_data_vectors.shape[0]
            test_predict = self.bayes_classifier.predict(test_data_vectors)
            not_matched=0
            for i,label in enumerate(test_predict):
                if(label!=self.test_dataset.target[i]):
                    not_matched+=1
            accuracy = (num_test_samples-not_matched)/num_test_samples
            print("num ",num,"accuracy " , accuracy)
        


if __name__ == "__main__":
    bayes = simple_bayes()
    print(bayes.get_accuracy())





