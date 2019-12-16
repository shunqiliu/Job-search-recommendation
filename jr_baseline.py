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
            
            r.extract_keywords_from_text(ff)
            
            k=r.get_ranked_phrases()
            
            tok=[]
            for ii in range(len(k)):
                if ii>2:
                    break
                tokens=nltk.word_tokenize(k[ii])
                if len(tokens)<=3:
                    tok.extend(tokens)
                else:
                    tok.extend(tokens[:3])
            
            
            v.append(tok)
            
        resume.append(v)
    return resume

def load_label_data():
    '''
    load raw data linkedin.zip, convert the job titles to list
    '''
    with open('label.csv','rt') as features:
        reader = csv.reader(features)
        frow = [row for row in reader]
    
    job=[]
    for i in range(1,len(frow)):
        tokens=frow[i][2].lower()
        remove = str.maketrans('','',string.punctuation) 
        tokens=tokens.translate(remove)
        tokens=nltk.word_tokenize(tokens)
        tokens=" ".join(tokens)
        job.append(tokens)
    return job

def load_model():
    '''
    load gensim models
    '''
    model1 = gensim.models.KeyedVectors.load_word2vec_format("model/Model.bin", binary=True)
    model2 = gensim.models.KeyedVectors.load_word2vec_format("model/GoogleNews-vectors-negative300.bin", binary=True)
    return model1,model2

def write_csv(res):
    '''
    write the fuzzy labels to csv
    '''
    with open("result.csv", "a", newline='', encoding='utf-8') as file:
        writer = csv.writer(file ,delimiter=',')
        for row in res:
            writer.writerow(row)

def load_result():
    '''
    load fuzzy labels
    '''
    with open('result.csv','rt') as features:
        reader = csv.reader(features)
        frow = [row for row in reader]
    return frow

def base_line(x_test,x_train,model1,model2):
    '''
    base line experiment
    input:
        x_test: test data features
        x_train: train data features
        model1:gensim model 1, include daily words models
        model2:gensim model 2, include job title models
    output:
        write the job title to csv file
    '''
    res=[]
    pre=-1.0
    weight=[0.045,0.27,0.09,0.154,0.009,0.136,0.045,0.09,0.054,0.009,0.09]
    for i in tqdm(range(len(x_test))):
        for j in range(len(x_train)):
            score=0.0
            for k in range(len(x_train[0])):
                if x_train[j][k]=="" or x_test[i][k]=="":
                    continue

                s=0.0
                c=0.0
                for ii in x_train[j][k]:
                    for jj in x_test[i][k]:
                        flag1=False
                        flag2=False
                        c+=1
                        try:
                            s+=model1.similarity(ii,jj)
                        except:
                            flag1=True

                        try:
                            s+=model2.similarity(ii,jj)
                        except:
                            flag2=True
                        if (not flag1) and (not flag2):
                            s/=2
                        if flag1 and flag2:
                            c-=1

                if c==0:
                    score+=fuzz.token_sort_ratio(x_train[j][k],x_test[i][k])/100
                    score*=weight[k]
                else:
                    score+=s/c
            
            if (score)>pre:
                pre=score
                result=y_train[j]
        res.append([result])
    write_csv(res)

def base_acc(y_te,r):
    '''
    compute accuracy of base line method
    '''
    count=0
    for i in range(len(y_te)):
        if y_te[i]==r[i]:
            count+=1
    print(count/len(y_te))

def convert_class(y,result,model1,model2,job):
    '''
    fuzzy labels
    input:
        y: job titles in raw data
        result: predicted job titles in unsupervised data(base line)
        model1: gensim model 1, include daily words models
        model2: gensim model 2, include job title models
        job: given job title
    output:
        flabel: fuzzy labels in raw data
        rlabel: labels in baseline prediction
        clabel: labels in baseline raw data
    '''
    clabel=[]
    rlabel=[]
    flabel=[]

    jt=nltk.word_tokenize(job)
    for t in tqdm(y):
        tw=nltk.word_tokenize(t)
        s=0
        count=0
        for i in tw:
            for j in jt:
                s1=0
                s2=0
                count+=1
                try:
                    s1=model1.similarity(i,j)
                except:
                    pass
                try:
                    s2=model2.similarity(i,j)
                except:
                    pass
                if s1==0 and s2==0:
                    s+=0
                elif s1==0 and s2!=0:
                    s+=s2
                elif s1!=0 and s2==0:
                    s+=s1
                else:
                    s+=0.2*s1+0.8*s2
            s/=count/2
        if s>=0.5:
            flabel.append([1,s])
            clabel.append(1)
        else:
            flabel.append([0,1-s])
            clabel.append(0)

    for t in tqdm(result):
        tw=nltk.word_tokenize(t[0])
        s=0
        count=0
        for i in tw:
            for j in jt:
                s1=0
                s2=0
                count+=1
                try:
                    s1=model1.similarity(i,j)
                except:
                    pass
                try:
                    s2=model2.similarity(i,j)
                except:
                    pass
                if s1==0 and s2==0:
                    s+=0
                elif s1==0 and s2!=0:
                    s+=s2
                elif s1!=0 and s2==0:
                    s+=s1
                else:
                    s+=0.2*s1+0.8*s2
            s/=count/2
        if s>=0.5:
            rlabel.append(1)
        else:
            rlabel.append(0)
    return clabel,rlabel,flabel
    

model1,model2=load_model()
x_train,y_train,x_test,y_test=load_feature_data()[:1925],load_label_data()[:1925],load_feature_data()[1925:],load_label_data()[1925:]

#base_line(x_test,x_train,model1,model2)

result=load_result()
job="product manager"
y,r,fy=convert_class(y_train+y_test,result,model1,model2,job)

base_acc(y[1925:],r)
