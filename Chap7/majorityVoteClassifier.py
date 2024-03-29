#implementing a MajorityVectorClassifier 

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six    #probably not needed in python 3
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """
    A majorit vote ensemble classifier
    
    Parameters:
    
    Classifiers: array-like, shape = [n_classifiers]
        Diferent classifiers for ensemble
        
    vote:   str, {"classlabel", "probability"}
        Default: "Classlabel"
        If "classlabel" the prediction is based on the argmax of class labels.
        Else If "probability" the argmax of the sum of probabilities is used to predict the class label (recommended for the calibrated classifiers).
        
    Weights: array-like, shape = [n_classifiers]
        Optional, default: None
        If a list of "Int" or "float" values ar eprovided, the clasifiers are weighted ny improtance; Uses uniform weights if "weights = None"
    
    """
    
    def __init__(self, classifiers, vote = "classlabel", weights = None):
        self.classifiers = classifiers
        self.named_classifiers = {key:value for key,value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
        
    def fit(self, X, y):
        """
        Fit classifiers
        
        Parameters:
        x: {array-like, sparse matrix}
            shape = [n_samples. n_features]
            Matrix of training samples.
        y: array-like, shape = [n_shapes]
            Vector of target class labels.
        """
        #use label encoder to ensure class labels start with 0, which is important for np.argmax.
        #call in self.predict
        
        self.labelenc_ = LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_ = self.labelenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.labelenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        
        return self
            
    def predict(self, X):
        '''
            Returns:
                maj_vote: array-like, shape = [n_samples]
                Predicted class labels
        '''
        
        if self.vote == "probability":
            maj_vote = np.argmax(self.predict_proba(X), axis = 1)
        else:   #"classlablel" vote
        
            #collect results from clf.predict calls
            predictions = np.asarray( [clf.predict(X) for clf in self.classifiers_] ).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights = self.weights)), axis = 1, arr = predictions)
        
        maj_vote = self.labelenc_.inverse_transform(maj_vote)
        return maj_vote
            
    def predict_proba(self, X):
        '''
            Predict class probabilties for X.
            
            Returns:
                avg_proba: array-like, shape = [n_amples, n_classes]
                Weighted average probability for each class per sample.
                
        '''
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        
        
        avg_proba = np.average(probas, axis = 0, weights = self.weights)
        
        return avg_proba
           
        pass
        
    def get_params(self, deep = True):
        "Get classifier parameter names for gridsearch"
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out= self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key,value in six.iteritems(step.get_params(deep = True)):
                    out["{}__{}".format(name,key)] = value
            return out
        pass
