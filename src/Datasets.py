# test git
from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import reuters 
from nltk.corpus import stopwords
from nltk import PorterStemmer
import re

def clean_text(text, stem = True):
    porter = PorterStemmer()
    stopword_set = list(stopwords.words('english'))

    #1: remove punctuation and numbers
    text = re.sub(r"\n|(\\(.*?){)|}|[!$%^&*#()_+|~\-={}\[\]:\";'<>?,.\/\\]|[0-9]|[@]", ' ', text)

    #2: remove extra space
    text = re.sub('\s+', ' ', text)

    #3: make a list + lower
    text = [word.lower() for word in text.split()] 

    #4: remove stop words 
    text = [word for word in text if word not in stopword_set] 

    #5: Stem
    if stem:
        text = [porter.stem(word) for word in text]

    text = ' '.join(text)

    return text

def get_20newsgroups(clean = True, stem = True):
    train_dataset = fetch_20newsgroups(subset='train',shuffle=True)
    test_dataset = fetch_20newsgroups(subset='test',shuffle=True)

    train_X = train_dataset["data"]
    train_Y = train_dataset["target"]

    test_X = test_dataset["data"]
    test_Y = test_dataset["target"]

    Y_names = train_dataset["target_names"]

    if clean: 
        for i, text in enumerate(train_X):
            train_X[i] = clean_text(text, stem)

        for i, text in enumerate(test_X):
            test_X[i] = clean_text(text, stem)

    return train_X, train_Y, test_X, test_Y, Y_names

def get_reuters():
    pass

def get_20newsgroups_vectorized():
    pass

def get_reuters_vectorized():
    pass

# for testing 
if __name__ == "__main__":
    train_X, train_Y, test_X, test_Y, Y_names = get_20newsgroups()
    
    print(train_X[:3])
    print(train_Y[:3])
    print(Y_names)

    