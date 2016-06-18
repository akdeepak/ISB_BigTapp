# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 21:56:43 2015

@author: ekardee
"""

import goslate
gs = goslate.Goslate()
print gs.translate('hello world', 'de')

print "deepak"


from nltk.tokenize import word_tokenize
 
tweet = 'RT @marcobonzanini: just an example! :D http://example.com #NLP'
print(word_tokenize(tweet))


 
tweet = "RT @marcobonzanini: just an example! :D http://example.com #NLP"
print(preprocess(tweet))


import operator 
import json
from collections import Counter
count_all = Counter()
# Create a list with all the terms
terms_all = [term for term in preprocess(tweet)]
# Update the counter
count_all.update(terms_all)
# Print the first 5 most frequent words
print(count_all.most_common(5))

from nltk.corpus import stopwords
import string
 
punctuation = list(string.punctuation)
print punctuation
stop = stopwords.words('english') + punctuation + ['rt', 'via']

terms_stop = [term for term in preprocess(tweet) if term not in stop]
print terms_stop

# ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com', '#NLP']
