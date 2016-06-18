# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 23:57:34 2015

@author: ekardee
"""

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