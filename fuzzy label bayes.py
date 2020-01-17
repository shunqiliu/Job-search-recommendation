import csv   
import nltk
import numpy as np
import os
from gensim.models import word2vec
import gensim
import string
from nltk.corpus import stopwords
from rake_nltk import Rake
from tqdm import tqdm
from nltk.metrics import edit_distance
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import collections

def load_feature_data():
    '''
    load raw data linkedin.zip, convert to list
    '''
    #load data
    with open('feature.csv','rt') as features:
        reader = csv.reader(features)
        frow = [row for row in reader]

    resume=[]
    r = Rake()
    for i in range(1,len(frow)):
        v=[]
        #print(resume)
        #os.system("Pause")
        for j in range(2,len(frow[i])):
            if j==7 or j==8 or j==9 or j==10 or j==12 or j==15 or j==16:
                continue
            ff=frow[i][j].lower()
            remove = str.maketrans('','',string.punctuation) 
            ff = ff.translate(remove)
            ff=nltk.word_tokenize(ff)
            
            v.extend(ff)
            
        resume.append(v)
    return resume

def load_label_data():
    with open('fuzzy label.csv','rt') as features:
        reader = csv.reader(features)
        frow = [row for row in reader]
    label=[]
    confidence=[]
    for i in frow:
        label.append(i[0])
        confidence.append(i[1])
    return label,confidence

def get_vocabulary(D):
    """
    Given a list of documents, where each document is represented as
    a list of tokens, return the resulting vocabulary. The vocabulary
    should be a set of tokens which appear more than once in the entire
    document collection plus the "<unk>" token.
    """
    # TODO
    aset=set()
    vocab=set()
    for i in D:
        for j in i:
            if j in aset:
                vocab.add(j)
            aset.add(j)
    vocab.add('<unk>')
    return vocab

class BBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the binary bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
        vdict=dict()
        for i in doc:
            if i in vocab:
                vdict[i]=1
            else:
                vdict['<unk>']=1
        return vdict

class CBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the count bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
        vdict=collections.defaultdict(int)
        for i in doc:
            if i in vocab:
                vdict[i]+=1
            else:
                vdict['<unk>']+=1
        return vdict

def compute_idf(D, vocab):
    """
    Given a list of documents D and the vocabulary as a set of tokens,
    where each document is represented as a list of tokens, return the IDF scores
    for every token in the vocab. The IDFs should be represented as a dictionary that
    maps from the token to the IDF value. If a token is not present in the
    vocab, it should be mapped to "<unk>".
    """
    # TODO
    D_len=len(D)
    vocab_dict=collections.defaultdict(float)
    
    for d in D:
        unk=True
        for token in set(d):
            if token in vocab:
                vocab_dict[token]+=1
            elif unk:
                vocab_dict['<unk>']+=1
                unk=False

    for i in list(vocab_dict.keys()):
        vocab_dict[i]=np.log(D_len/vocab_dict[i])

    return vocab_dict
    
class TFIDFFeaturizer(object):
    def __init__(self, idf):
        """The idf scores computed via `compute_idf`."""
        self.idf = idf
    
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and
        the vocabulary as a set of tokens, compute
        the TF-IDF feature representation. This function
        should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
        vdict=collections.defaultdict(float)
        for i in doc:
            if i in vocab:
                vdict[i]+=1.0
            else:
                vdict['<unk>']+=1.0
        for i in list(vdict.keys()):
            vdict[i]=self.idf[i]*vdict[i]*100
        return vdict

def convert_to_features(D, featurizer, vocab):
    X = []
    for doc in D:
        X.append(featurizer.convert_document_to_feature_dictionary(doc, vocab))
    return X

def train_naive_bayes(X, y, c, k, vocab):
    """
    Computes the statistics for the Naive Bayes classifier.
    X is a list of feature representations, where each representation
    is a dictionary that maps from the feature name to the value.
    y is a list of integers that represent the labels.
    k is a float which is the smoothing parameters.
    c is the confidence of y
    vocab is the set of vocabulary tokens.
    
    Returns two values:
        p_y: A dictionary from the label to the corresponding p(y) score
        p_v_y: A nested dictionary where the outer dictionary's key is
            the label and the innner dictionary maps from a feature
            to the probability p(v|y). For example, `p_v_y[1]["hello"]`
            should be p(v="hello"|y=1).
    """
    # TODO
    p_y=collections.defaultdict(float)
    county=0
    for i in range(len(y)):
        if y[i]==1:
            county+=1
        p_y[i]+=c[i]
    py1=county/len(y)
    py0=1-py1
    p_y[0]*=py0
    p_y[1]*=py1

    den=collections.defaultdict(float)
    for v in list(vocab):
        for i in range(len(y)):
            if v in X[i]:
                den[y[i]]+=X[i][v]*c[i]*100
                den[1-y[i]]+=X[i][v]*(1-c[i])*100

    p_v_y={}
    y_key=[0,1]
    for key in y_key:
        p_v_y[key]=collections.defaultdict(float)
    
    for i in range(len(y)):
        for key in list(X[i].keys()):
            p_v_y[y[i]][key]+=X[i][key]*c[i]*100
            p_v_y[1-y[i]][key]+=X[i][key]*(1-c[i])*100

    for k1 in y_key:
        for k2 in list(p_v_y[k1].keys()):
            p_v_y[k1][k2]=(p_v_y[k1][k2]+k)/(den[k1]+k*len(vocab))
        for ks in list(vocab):
            if not ks in p_v_y[k1]:
                p_v_y[k1][ks]=k/(den[k1]+k*len(vocab))
    return p_y,p_v_y

def predict_naive_bayes(D, p_y, p_v_y):
    """
    Runs the prediction rule for Naive Bayes. D is a list of documents,
    where each document is a list of tokens.
    p_y and p_v_y are output from `train_naive_bayes`.
    
    Note that any token which is not in p_v_y should be mapped to
    "<unk>". Further, the input dictionaries are probabilities. You
    should convert them to log-probabilities while you compute
    the Naive Bayes prediction rule to prevent underflow errors.
    
    Returns two values:
        predictions: A list of integer labels, one for each document,
            that is the predicted label for each instance.
        confidences: A list of floats, one for each document, that is
            p(y|d) for the corresponding label that is returned.
    """
    # TODO
    predictions=[]
    confidences=[]
    for i in range(len(D)):
        y0=0
        y1=0
        pdy0=1
        pdy1=1
        for token in D[i]:
            if token in p_v_y[0]: 
                y0+=np.log(p_v_y[0][token])
                pdy0=y0
            else:
                y0+=np.log(p_v_y[0]["<unk>"])
                pdy0=y0
            if token in p_v_y[1]:
                y1+=np.log(p_v_y[1][token])
                pdy1=y1
            else:
                y1+=np.log(p_v_y[1]["<unk>"])
                pdy1=y1
        y0+=np.log(p_y[0])
        y1+=np.log(p_y[1])
        con0=pdy0+np.log(p_y[0])-(np.log(p_y[0]*np.exp(pdy0)+p_y[1]*np.exp(pdy1)))
        con1=pdy1+np.log(p_y[1])-(np.log(p_y[0]*np.exp(pdy0)+p_y[1]*np.exp(pdy1)))
        if y0>y1:
            predictions.append(0)
            confidences.append(np.exp(con0))
            
        else:
            predictions.append(1)
            confidences.append(np.exp(con1))
            
    return predictions,confidences


x_train,x_test=load_feature_data(),load_feature_data()
y_train=list(map(float,load_label_data()[0]))
c_train=list(map(float,load_label_data()[1]))
y_test=list(map(float,load_label_data()[0]))
c_test=list(map(float,load_label_data()[1]))


vocab = get_vocabulary(x_train)
featurizer = TFIDFFeaturizer(compute_idf(x_train, vocab))
x_t = convert_to_features(x_train, featurizer, vocab)
p_y,p_v_y=train_naive_bayes(x_t, y_train,c_train, 1.0, vocab)
predictions,confidences=predict_naive_bayes(x_test, p_y, p_v_y)

count=0
for i in range(len(predictions)):
    if predictions[i]==y_test[i]:
        count+=1
print(count/len(predictions))
