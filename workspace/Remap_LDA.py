# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:44:10 2016

@author: ekardee
"""

print "Deepak ROCK STAR"

import pandas as pd
import numpy as np
from collections import defaultdict
data = defaultdict(list)
filename_jan04 = "C:\\ISB\\BigTapp\\TwitterBigTap\\data_jan_4_2015.csv"
tweetFields = ['text','user_lang']
df = pd.read_csv(filename_jan04, low_memory=False,usecols=tweetFields)
for row in df.itertuples():
    data[row[2]].append(row [1])

df5 = df.head(n=100000)
data5 = defaultdict(list)
for row in df5.itertuples():
    data5[row[2]].append(row [1])
#print data5.keys()
tweets_en = data5['en']

tweets_en = data['en']

import operator
from collections import Counter
tweets_en_tok = []
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
for text in tweets_en:
    tweets_en_tok.append(tknzr.tokenize(text))


from sklearn.feature_extraction import text, stop_words
import nltk
import string
punctuation = list(string.punctuation)
tweet_stop_words = ['rt','via','RT','\\x80', 'I','i',':)',';)',':/',':(',':3','<3']
stop_words = text.ENGLISH_STOP_WORDS
stop_words = stop_words.union(punctuation)
stop_words = stop_words.union(tweet_stop_words)
tweets_stop_words=[]
for terms in tweets_en_tok:
    tweets_stop_words.append([t for t in terms if t not in stop_words])

users_start_with_at=[]
hashtags=[]
web_links=[]
tweet_list=[]
import time 
t0 = time.time()
for idx, val in enumerate(tweets_stop_words):
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
    tweet_list.append(filter(None, tweetcontents))
#tweet_list = filter(None, tweetcontents)
print tweet_list
tweet_str_list=[]
for tweets in tweet_list:
    str = ' '.join(tweets)
    tweet_str_list.append(str)


joinstr = ' '.join(tweet_str_list)
text_file = open("C:\\Python_Eclipse\\jan04_tweets\\test_jan15.txt", "w")
text_file.write("%s" % joinstr)
text_file.close()




print tweet_str_list


data_samples = tweet_str_list

print data_samples
from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

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
# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, #max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
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

