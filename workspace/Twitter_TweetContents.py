# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 23:20:46 2015

@author: ekardee
"""
import pandas as pd
import numpy as np
from collections import defaultdict
data = defaultdict(list)
filename_jan04 = "C:\\ISB\\BigTapp\\TwitterBigTap\\data_2015-01-10.csv"
tweetFields = ['text','user_lang']
CHUNKSIZE  = 10 ** 6
print CHUNKSIZE
df = pd.read_csv(filename_jan04,error_bad_lines=False, low_memory=False,usecols=tweetFields,chunksize = CHUNKSIZE)
i=1
for chunk in df:
    for row in chunk.itertuples():
        data[row[2]].append(row [1])   
    np.savetxt(r"C:\\Python_Eclipse\\jan10_tweets\\tweets%s.txt"%i, data['en'], fmt='%s')
 

print len(data.keys())
lst_keys= data.keys()
valid_keys=[]
for key in lst_keys:
     value = data[key]
     if len(value) >= 20:
         valid_keys.append(key)
print valid_keys
print len(valid_keys)

for key in valid_keys:
    print key

################### word Tokenize with tweet tokenizer ############################################
tweets_en = data['en']
import operator
from collections import Counter
tweets_en_tok = []
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
for text in tweets_en:
    tweets_en_tok.append(tknzr.tokenize(text))

######################################################################################

############# Stop words Built for the tweet contends ##############################
import nltk
import string
from sklearn.feature_extraction import text
punctuation = list(string.punctuation)
tweet_stop_words = ['rt','via','RT','\\x80']
stop_words = text.ENGLISH_STOP_WORDS
punctuation
stop_words = stop_words.union(punctuation)
stop_words = stop_words.union(tweet_stop_words)
stop_words
print len(stop_words)
################################################################################

######################## stop words ###########################################
tweets_stop_words=[]
for terms in tweets_en_tok:
    print terms
    break
    tweets_stop_words.append([i for i in terms if i not in stop_words])

tweetcontents=[]
users_start_with_at=[]
hashtags=[]
web_links=[]
for idx, val in enumerate(tweets_stop_words):
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

############ Data Cleaning - Identify the language ##################################
############# Detect Language #######################################################
from langdetect import detect
tweets_final=[]
for tweet in tweet_list:
    lang = detect(tweet)
    if lang != 'en':
        print "Detected Language : " ,lang
        data[lang].append(tweet)
    elif lang == 'en':
        tweets_final.append(tweet)
##############################################################################################################

str = ' '.join(tweet_list)
text_file = open("C:\\Python_Eclipse\\jan09_tweets\\test_en.txt", "a")
text_file.write("%s" % str)
text_file.close()

import lda
import lda.datasets
import numpy as np
import textmining

with open('C:\\Python_Eclipse\\jan09_tweets\\test_only.txt', 'r') as myfile:
    doc=myfile.read()

   
tdm = textmining.TermDocumentMatrix()
tdm.add_doc(doc)
temp = list(tdm.rows(cutoff=1))
vocab = tuple(temp[0])
X = np.array(temp[1:])
print("\n** Output produced by the textmining package...")
print("* The 'document-term' matrix")
print("type(X): {}".format(type(X)))
print("shape: {}".format(X.shape))
print("X:", X)

model = lda.LDA(n_topics=20, n_iter=1000, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available

topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
doc_topic = model.doc_topic_
for i in range(10):
    print(" (top topic: {})".format(doc_topic[i].argmax()))

############ Data Cleaning - Identify the language ##################################
############# Detect Language #######################################################
from langdetect import detect
for tweet in tweet_list:
     print tweet
     lang = detect(tweet)
     if lang != 'en':
         print "Detected Language : " ,lang
         data[lang].append(tweet)
######################################################################################
 
lang = detect('2')




from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
tfidf_matrix =  tf.fit_transform(tweet_list)
feature_names = tf.get_feature_names() 
len(feature_names)
 feature_names[50:70]
 


############################# LDA from 10 GB of File ##############
##################################################################
####################################################################
import lda
import lda.datasets
import numpy as np
import textmining

doc=[]
j=1
for i in range (1,8):
    with open('C:\\Python_Eclipse\\jan10_tweets\\tweets_document%s.txt'%j, 'r') as tweetdocfile:
        doc.append(tweetdocfile.read())
        j=j+2
        print j
        
tdm = textmining.TermDocumentMatrix()
for d in doc:
    tdm.add_doc(d)
temp = list(tdm.rows(cutoff=1))
vocab = tuple(temp[0])
X = np.array(temp[1:])
print("\n** Output produced by the textmining package...")
print("* The 'document-term' matrix")
print("type(X): {}".format(type(X)))
print("shape: {}".format(X.shape))
print("X:", X)

model = lda.LDA(n_topics=20, n_iter=1000, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available

topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
doc_topic = model.doc_topic_
for i in range(10):
    print(" (top topic: {})".format(doc_topic[i].argmax()))


#######################################################################





