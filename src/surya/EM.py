import numpy as np
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import coo_matrix, vstack
class Semi_NB:

    def __init__(self):
        super().__init__()
        self.nb_clf = MultinomialNB(alpha=0.01)

    def fit(self,X_l,y_l,X_u):
        self.nb_clf = MultinomialNB(alpha=0.01)
        self.nb_clf.fit(X_l,y_l)
        # calculate log likelihood 
        class_log_prior = (self.nb_clf.class_log_prior_).tolist()
        word_given_class = self.nb_clf.feature_log_prob_
        class_size = len(self.nb_clf.class_count_)
        un_sum_outer = 0
        for doc in X_u:
            sum_inner = 0
            for index in range(class_size):
                sum_inner += (class_log_prior[index]* np.sum(word_given_class[index,:]))
            un_sum_outer += np.log(sum_inner)
        
        lb_sum = 0
        for index,doc in enumerate(X_l):
            sum_inner = 0
            given_label = y_l[index]
            sum_inner = (class_log_prior[given_label]* np.sum(word_given_class[given_label,:]))
            lb_sum += sum_inner

        log_likelihood = lb_sum + un_sum_outer
        count = 0
        while(count < 6):

            print("Iteration ",(count+1))
            # Estimation step
            Y_u = self.nb_clf.predict(X_u)

            # Maximize step
            X_new = vstack([X_l, X_u])
            Y_new = np.concatenate((y_l, Y_u), axis=0)
            self.nb_clf.fit(X_new, Y_new)

            # calculate log likelihood 
            class_log_prior = (self.nb_clf.class_log_prior_).tolist()
            word_given_class = self.nb_clf.feature_log_prob_
            class_size = len(self.nb_clf.class_count_)
            count +=1
            un_sum_outer = 0
            for doc in X_u:
                sum_inner = 0
                for index in range(class_size):
                    sum_inner += (class_log_prior[index]* np.sum(word_given_class[index,:]))
                un_sum_outer += np.log(sum_inner)
            
            lb_sum = 0
            for index,doc in enumerate(X_l):
                sum_inner = 0
                given_label = y_l[index]
                sum_inner = (class_log_prior[given_label]* np.sum(word_given_class[given_label,:]))
                lb_sum += sum_inner

            log_likelihood = lb_sum + un_sum_outer
            print("log_likelihood ",log_likelihood)

    def predict(self, X):
        return self.nb_clf.predict(X)

