import numpy as np
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import coo_matrix, vstack
class Semi_NB:

    def __init__(self):
        super().__init__()
        self.nb_clf = MultinomialNB(alpha=0.01)
        self.log_lkh = None

    def fit(self,X_l,y_l,X_u):
        self.nb_clf.fit(X_l,y_l)
        
        # calculate log likelihood 
        class_log_prior = (self.nb_clf.class_log_prior_).tolist()
        word_given_class = self.nb_clf.feature_log_prob_
        class_size = len(self.nb_clf.class_count_)
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
            # print("given label",given_label)
            # print("class log prior",len(class_log_prior))
            sum_inner = (class_log_prior[given_label]* np.sum(word_given_class[given_label,:]))
            lb_sum += sum_inner

        log_likelihood = (-1 * (lb_sum + un_sum_outer))
        prev_log = float("-inf")
        current_log = log_likelihood
        count = 0
        # remove this line
        while(abs(current_log-prev_log) > 1e-6):
            print("count",count+1)
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
            self.log_lkh = log_likelihood
            print("log_likelihood ",log_likelihood)
        return self

    def predict(self, X):
        return self.nb_clf.predict(X)

