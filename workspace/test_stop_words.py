# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:49:23 2015

@author: ekardee
"""
from nltk.corpus import stopwords
stop = stopwords.words('english')
sentence = "this is a foo bar sentence"
print [i for i in sentence.split() if i not in stop]
['foo', 'bar', 'sentence']


import nltk
text = "Elvis Aaron Presley was an American singer and actor. Born in Tupelo, Mississippi, when Presley was 13 years old he and his family relocated to Memphis, Tennessee."

chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))

print chunks

import nltk
import string
from sklearn.feature_extraction import text
punctuation = list(string.punctuation)
tweet_stop_words = ['rt','via']
stop_words = text.ENGLISH_STOP_WORDS
print len(stop_words)
punctuation
stop_words = stop_words.union(punctuation)
print len(stop_words)
stop_words = stop_words.union(tweet_stop_words)
stop_words
print len(stop_words)

import operator
from collections import Counter
text = "Elvis Aaron Presley was an American singer and actor. Born in Tupelo, Mississippi, when Presley was 13 years old he and his family relocated to Memphis, Tennessee."
count_all = Counter()
terms_all = preprocess(text)
print terms_all
term_with_counters = Counter(terms_all)
print term_with_counters
tweet_after_stop_words= [i for i in term_with_counters if i not in stop]
print tweet_after_stop_words
 from nltk import bigrams 
terms_bigram = bigrams(tweet_after_stop_words)   
print terms_bigram
for term in terms_bigram:
    print term


print(count_all.most_common(5))

terms_stop = [term for term in preprocess(text) if term not in stop_words]
print terms_stop
terms_single = set(terms_all)
# Count hashtags only
terms_hash = [term for term in preprocess(tweet['text']) 
              if term.startswith('#')]
# Count terms only (no hashtags, no mentions)
terms_only = [term for term in preprocess(tweet['text']) 
if term not in stop and
              not term.startswith(('#', '@'))] 
              # mind the ((double brackets))
              # startswith() takes a tuple (not a list) if 
              # we pass a list of inputs

from nltk.corpus import stopwords
import string
 
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']
print stop