import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#just an example of bag-of-words model: how to create 1-gram feature vector model from a list of words

count = CountVectorizer()
docs = np.array(["The sun is shining",
                "The weather is sweet",
                "The sun is shining and the weather is sweet"])
bag = count.fit_transform(docs)

print(count.vocabulary_)

print(bag.toarray())

#now convert the frequencies to tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
np.set_printoptions(precision = 2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

#remove html tags from text 
import re
def preprocessor(text): 
    text = re.sub("<[^>]*>", '', text)  #replace anything under the html tags with none
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)    #catch emoticons and save them in a list named emoticons
    text = re.sub("[\W]+"," ", text.lower()) + ' '.join(emoticons).replace("-","")  #remove all non-word chars, convert text to lowercase, remove the nose '-' from emoticons, add the emoticons at the end of text
    return text
    
print(preprocessor("</a>This :) is :( a test :-)!"))

def tokenizer(text):
    return text.split()
    
print(tokenizer("runners like running and thus they run"))

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

def tokenize_porter(text):
    return [porter.stem(word) for word in text.split()]
    
print(tokenize_porter("runners like running and thus they run"))


#remove stop words
import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
stop = stopwords.words("english")
print([w for w in tokenize_porter("a runner likes running and runs a lot") if w not in stop]) 

#apply preprocessor on our movie reviews
#import pandas as pd
#df = pd.read_csv("./movie_data.csv")
#print(df.head(10))

#df["review"] = df["review"].apply(preprocessor)
#print(df.head(10))

#df["review"] = df["review"].apply(tokenize_porter)
#print(df.head(10))
