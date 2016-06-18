# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 22:17:47 2016

@author: ekardee
"""
print "deepak"
import pandas as pd
import numpy as np
from collections import defaultdict
data = defaultdict(list)
filename_jan04 = "C:\\ISB\\BigTapp\\TwitterBigTap\\data_jan_4_2015.csv"
tweetFields = ['text','user_lang']
df = pd.read_csv(filename_jan04,error_bad_lines=False, low_memory=False,usecols=tweetFields)
for row in df.itertuples():
    data[row[2]].append(row [1])

################ POC of the tweeet data with 20 tweets ###############################################
df5 = df.head(n=1000)
data5 = defaultdict(list)
for row in df5.itertuples():
    data5[row[2]].append(row [1])
print data5.keys()
tweets_en = data5['en']

print tweets_en
#np.savetxt(r"C:\\Python_Eclipse\\jan04_tweets\\tweets_head_1000.txt", tweets_en, fmt='%s')

tweets_es = data5['es']
tweets_tr = data5['tr']
for en in tweets_en:
    print en
    print '\n'
for es in tweets_es:
    print es
for tr in tweets_tr:
    print tr  
    
    
################### word Tokenize with tweet tokenizer ############################################
import operator
from collections import Counter
terms_all = []
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
for text in tweets_en:
    terms_all.append(tknzr.tokenize(text))
np.savetxt(r"C:\\Python_Eclipse\\jan04_tweets\\tweets_head_1000_tokenizer.txt", terms_all, fmt='%s')

#######################################################################

############# Stop words Built for the tweet contends ##############################
import nltk
import string
from sklearn.feature_extraction import text
punctuation = list(string.punctuation)
tweet_stop_words = ['rt','via','RT','\\x80', 'I','i',':)',';)',':/',':(']
stop_words = text.ENGLISH_STOP_WORDS
print len(stop_words)
punctuation
stop_words = stop_words.union(punctuation)
print len(stop_words)
stop_words = stop_words.union(tweet_stop_words)
stop_words
print len(stop_words)   
################################################################################
    
######################## stop words & Data cleansing and writing the tweet contents#######
########################### in a file  ################################################
tweet_after_stop_words=[]
for terms in terms_all:
    tweet_after_stop_words.append([i for i in terms if i not in stop_words])
print tweet_after_stop_words
np.savetxt(r"C:\\Python_Eclipse\\jan04_tweets\\tweets_head_1000_stop_words.txt", terms_all, fmt='%s')
######################################################################################

############ Data Cleaning - Identify the language ##################################
############# Detect Language #######################################################
tweet_contents_str=[]
tweetcontents=[]
users_start_with_at=[]
hashtags=[]
web_links=[]
tweet_list=[]
import time 
t0 = time.time()
i = 1
for idx, val in enumerate(tweet_after_stop_words):
    tweetcontents=[]
    for term in val:
        if term.startswith('@'):
            users_start_with_at.append(term)
            continue
        if term.startswith('#'):
             hashtags.append(term)
             continue
        if term.startswith('http'):
             web_links.append(term)
             continue
        if term.startswith('.'):
             continue
        tweetcontents.append(term.encode('ascii','ignore'))
        tweet_list = filter(None, tweetcontents)
    str = ' '.join(tweet_list)
    tweet_contents_str.append(str)
    i=i+1
    if(i==20):
        break
print tweet_contents_str
    #tweet_list.append(filter(None, tweet_contents_str))
print time.time() - t0

#tweet_list = filter(None, tweetcontents)

for text in tweet_contents_str:
    print text
    break
######################################################################################
for i in range(1,30):
    text_file = open("C:\\Python_Eclipse\\jan04_tweets\\tweetList.txt", "a")
    text_file.write("%s" % tweet_contents_str)
    text_file.close()


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = tweet_list
vectorizer = TfidfVectorizer(min_df=2)
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
vect_tweets=  dict(zip(vectorizer.get_feature_names(), idf))
for vect in vect_tweets:
    print vect
    break
print vect_tweets  
  
  
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer  = TfidfVectorizer()
tf_transform = tfidf_vectorizer.fit_transform(tweet_list)
idf = tf_transform.idf_
vect_tweets=  dict(zip(tf_transform.get_feature_names(), idf))
print vect_tweets

terms = tfidf_vectorizer.get_feature_names()
print terms

##################### K means #########################

from sklearn.cluster import KMeans
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
%time km.fit(tf_transform)
clusters = km.labels_.tolist()

from sklearn.externals import joblib

#uncomment the below to save your model 
#since I've already run my model I am loading from the pickle
joblib.dump(km,  'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

for c in clusters:
    print c

##########################################################

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
with open('C:\\Python_Eclipse\\jan04_tweets\\tweets_head_1000_terms_only.txt', 'r') as tweetdocfile:
   synopses =  tweetdocfile.read()
print synopses



#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
print synopses

tfidf_vectorizer  = TfidfVectorizer()
tf_transform = tfidf_vectorizer.fit_transform(synopses)

print tf_transform
idf = tf_transform.idf_
vect_tweets=  dict(zip(tf_transform.get_feature_names(), idf))

tfidf_vectorizer = TfidfVectorizer(min_df=1)
tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)


tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses

print(tfidf_matrix.shape)

with open('C:\\Python_Eclipse\\jan10_tweets\\test_tweets1.txt', 'r') as tweetdocfile:
   synopses =  tweetdocfile.read()
   
print synopses

result = ''.join([i for i in synopses if not i.isdigit()])
print result

text_file = open("C:\\Python_Eclipse\\jan10_tweets\\test_tweets_text.txt", "w")
text_file.write("%s" % result) 



################################ SCI-KIT TOPIC MODELING ############################

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


print("Loading dataset...")
t0 = time()
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
type(dataset.data)
file = open("C:\\Python_Eclipse\\jan04_tweets\\test.txt", "r")
#data_samples = [line.split(',') for line in file.readlines()]
data_samples = file.read().split(',')
type(data_samples)
print("done in %0.3fs." % (time() - t0))

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(min_df=1, #max_features=n_features,
                                   )
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

print("Extracting tf features for LDA...")  
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model with tf-idf features,"
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
#exit()
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)




from nltk import bigrams 
terms_bigram = bigrams(tweet_after_stop_words)   
print terms_bigram
for term in terms_bigram:
    print term
terms_bigram_with_count = bigrams(tweet)   
for term_bigram in terms_bigram_with_count:
    print terms_bigram

################# Bigrams ########################################
from nltk import bigrams 
terms_bigram = bigrams(count_all)   
print terms_bigram
for term in terms_bigram:
        print term
##################### ############# ###################