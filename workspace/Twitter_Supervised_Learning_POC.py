# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:44:58 2016

@author: ekardee
"""

import pandas as pd
import numpy as np
from collections import defaultdict
data = defaultdict(list)
filename_jan04 = "C:\\ISB\\BigTapp\\TwitterBigTap\\data_jan_4_2015.csv"
tweetFields = ['text','user_lang']
df = pd.read_csv(filename_jan04, low_memory=False,usecols=tweetFields)
for row in df.itertuples():
    data[row[2]].append(row [1])

################ POC of the tweeet data with 20 tweets ###############################################
df5 = df.head(n=100000)
data5 = defaultdict(list)
for row in df5.itertuples():
    data5[row[2]].append(row [1])
tweets_en = data5['en']

print data5.keys()
print tweets_en

import operator
from collections import Counter
terms_all = []
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
for text in tweets_en:
    try:
       tokenized_text= tknzr.tokenize(text)
    except Exception:
        print "Exception thrown in python in text" , text
        pass  # or you could use 'continue'
    terms_all.append(tokenized_text)

######################################################################################

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

#####################################################################################

########################## Hashtags as keys and value as 
hashtag_with_tweets = defaultdict(list)
hashtag='#'
for idx, val in enumerate(tweet_after_stop_words):
     if any(hashtag in s for s in val):
        for s in val:
            if hashtag in s:
                hashtag_with_tweets[s].append(val)

print hashtag_with_tweets.keys()
print len(hashtag_with_tweets.keys())

np.savetxt(r"C:\\Python_Eclipse\\jan04_tweets\\testtest.txt", hashtag_with_tweets.keys(), fmt='%s')
print hashtag_with_tweets.keys()
print len(hashtag_with_tweets.keys())
hashtag_with_tweets.items()

lengths = {key:len(value) for key,value in hashtag_with_tweets.iteritems()}


hashtag_with_more_tweets =[]
for key,value in hashtag_with_tweets.iteritems():
    if len (value) > 100:
        hashtag_with_more_tweets.append(key)

print hashtag_with_more_tweets

tweetList = hashtag_with_tweets['#Samsung']
print tweetList

tweetList =hashtag_with_tweets['#ipad']

for tweet in tweetList:
    print tweet

tweet_contents_str=[]
tweetcontents=[]
users_start_with_at=[]
hashtags=[]
web_links=[]
tweet_list=[]
import time 
t0 = time.time()
i = 1
for idx, val in enumerate(tweetList):
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
print time.time() - t0

print tweet_contents_str

str = ' '.join(tweet_contents_str)
#text_file = open("C:\\Python_Eclipse\\jan04_tweets\\hashtag_samsung.txt", "w")
text_file = open("C:\\Python_Eclipse\\jan04_tweets\\hashtag_ipad.txt", "w")
text_file.write("%s" % tweet_contents_str)
text_file.close()