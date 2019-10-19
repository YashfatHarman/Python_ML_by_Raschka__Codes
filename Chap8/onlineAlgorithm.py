import numpy as np
import re

from nltk.corpus import stopwords

stop = stopwords.words("english")

def tokenizer(text):
	text = re.sub("<[^>]*>", "", text)	#anything between html tags is ignored
	emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text.lower())
	text = re.sub("[\W]+"," ", text.lower()) + " ".join(emoticons).replace("-","")
	tokenized = [w for w in text.split() if w not in stop]
	return tokenized
	pass

#generator function that reads in and returns one document at a time
def stream_docs(path):
	with open(path, "r") as csv:
		next(csv) #skip header
		for line in csv:
			text, label = line[:-3], int(line[-2])
			yield text,label
	pass


def get_minibatch(doc_stream, size):
	docs, y = [], []
	try:
		for _ in range(size):
			text,label = next(doc_stream)
			docs.append(text)
			y.append(label)
	except	StopIteration:
		return None, None

	return docs, y
		
	pass


#CountVecotrizer or TfidfVectorizer are not suitable for out-of-core learning.
#however, HashingVectorizer can be used for this purpose.

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error = "ignore", n_features = 2**21, preprocessor = None, tokenizer = tokenizer)
clf = SGDClassifier(loss = "log", random_state = 1, n_iter = 1)
	#SGDClassifier is a regularized linear model with stochastic gradient descent (SGD) learning.
	#SGD means the gradient of the loss will be estimated each sample at a time. 
	#By setting loss = log, we are selcting a logistic regresson as our classifier. 

doc_stream = stream_docs(path = "./movie_data.csv")

import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0,1])

for _ in range(45):
	X_train, y_train = get_minibatch(doc_stream, size = 1000)
	if not X_train:
		break
	X_train = vect.transform(X_train)
	clf.partial_fit(X_train, y_train, classes = classes)
	pbar.update()

#use test data to evaluate the model
X_test, y_test = get_minibatch(doc_stream, size = 5000)
X_test = vect.transform(X_test)
print("Accuracy: {:.3f} ".format(clf.score(X_test, y_test)))

#finally, use the 5000 test data to update the model
clf = clf.partial_fit(X_test, y_test)
