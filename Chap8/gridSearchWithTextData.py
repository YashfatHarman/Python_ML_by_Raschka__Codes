#remove html tags from text 
import re
def preprocessor(text): 
    text = re.sub("<[^>]*>", '', text)  #replace anything under the html tags with none
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)    #catch emoticons and save them in a list named emoticons
    text = re.sub("[\W]+"," ", text.lower()) + ' '.join(emoticons).replace("-","")  #remove all non-word chars, convert text to lowercase, remove the nose '-' from emoticons, add the emoticons at the end of text
    return text
    

def tokenizer(text):
    return text.split()
    

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenize_porter(text):
    return [porter.stem(word) for word in text.split()]
    
    
#remove stop words
import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
stop = stopwords.words("english")
    
    
#open our data
import pandas as pd
df = pd.read_csv("./movie_data.csv")


X_train = df.loc[:25000, "review"].values
y_train = df.loc[:25000, "sentiment"].values
X_test = df.loc[25000:, "review"].values
y_test = df.loc[25000:, "sentiment"].values

#now use gridsearch to find the optimal set of parameters
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents = None, lowercase = False, preprocessor = None)

param_grid = [{ "vect__ngram_range" : [(1,1)], 
                "vect__stop_words" : [stop,None], 
                "vect__tokenizer" : [tokenizer, tokenize_porter], 
                "clf__penalty" : ["l1","l2"], 
                "clf__C" : [1.0, 10.0, 100.0] },
              { "vect__ngram_range" : [(1,1)], 
                "vect__stop_words" : [stop,None], 
                "vect__tokenizer" : [tokenizer, tokenize_porter],
                "vect__use_idf" : [False],
                "vect__smooth_idf" : [False],
                "vect__norm" : [None], 
                "clf__penalty" : ["l1","l2"], 
                "clf__C" : [1.0, 10.0, 100.0] }  
            ]

lr_tfidf = Pipeline([("vect",tfidf), ("clf",LogisticRegression(random_state = 0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring = "accuracy", cv = 5, verbose = 1, n_jobs = -1)
gs_lr_tfidf.fit(X_train, y_train)

#grid search done. print the best parameters found
print("Best parameter set: ", gs_lr_tfidf.best_params_)

print("CV accuracy: {:.3f}".format(gs_lr_tfidf.best_score_))

clf = gs_lr_tfidf.best_estimator_

print("Test accuracy: {:.3f}".format(clf.score(X_test, y_test)))

