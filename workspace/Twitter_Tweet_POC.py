# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 22:26:48 2015

@author: ekardee
"""
print "deepak"
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
df5 = df.head(n=100)
data5 = defaultdict(list)
for row in df5.itertuples():
    data5[row[2]].append(row [1])
print data5.keys()
tweets_en = data5['en']

print tweets_en
print tweets_en
np.savetxt(r"C:\\Python_Eclipse\\jan04_tweets\\tweets_head_1000.txt", tweets_en, fmt='%s')

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
print terms_all
#######################################################################

for text in terms_all:
    print text
    break

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


###################### Lemmatizer ################################################
tweet_after_lemma=[]
import nltk
lemma = nltk.wordnet.WordNetLemmatizer()
for words in tweet_after_stop_words:
    print lemma.lemmatize(words)
    tweet_after_lemma.append(lemma.lemmatize(words))    
print tweet_after_lemma
np.savetxt(r"C:\\Python_Eclipse\\jan04_tweets\\tweets_head_1000_terms_lemma.txt", tweet_list, fmt='%s')
#####################################################################################
  
    
############ Data Cleaning -  ##################################
############# Detect Language #######################################################

tweetcontents=[]
users_start_with_at=[]
hashtags=[]
web_links=[]
tweet_list=[]
import time 
t0 = time.time()
for idx, val in enumerate(tweet_after_stop_words):
    print idx ,val
    break
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
print time.time() - t0
#tweet_list = filter(None, tweetcontents)
print tweet_list

print tweetcontents

######################################################################################
for text in tweet_list:
    print text
    break

###################### Lemmatizer ################################################
tweet_after_lemma=[]
tweets_lemma_list=[]
import nltk
lemma = nltk.wordnet.WordNetLemmatizer()
for words in tweet_list:
    for w in  words:
        tweet_after_lemma.append(lemma.lemmatize(w))
    tweets_lemma_list.append(tweet_after_lemma)
    
print tweets_lemma_list
######################################################################################
###################### Remove integers for the R code Interpretation ###############
    #################################################################################
tweet_remove_numerics=[]
for tweet in tweet_list:
    result = ' '.join([i for i in tweet if not i.isdigit()])
    tweet_remove_numerics.append(result)
###################################################################################

############### Detect Language #####################################################
###################################################################################
tweets_after_lang = []
from langdetect import detect
for tweet in tweet_remove_numerics:
    lang = detect(tweet)
    print lang
    break
    if lang != 'en':
        print "Detected Language : " ,lang
        data5[lang].append(tweet)
        continue
    tweets_after_lang.append(tweet)

###################################################################################



text_file = open("C:\\Python_Eclipse\\jan04_tweets\\twitterdata.txt", "w")
text_file.write("%s" % tweet_remove_numerics)
text_file.close()


############# remove integers in the words ##########################################


###################### convert lemmatizer tokenized words into document #############
str = ' '.join(tweet_after_lemma)
print str
text_file = open("C:\\Python_Eclipse\\jan04_tweets\\tweets_head_1000_terms_only.txt", "w")
text_file.write("%s" % str)
text_file.close()

######################################################################################

from langdetect import detect
l= 1
for tweet in tweet_list:
    lang = detect(tweet)
    print lang
    break
    if lang != 'en':
        print "Detected Language : " ,lang
        data5[lang].append(tweet)
    if(l == 10):
        break
    l=l+1   
        

lang = detect("Ein, zwei, drei, vier")
print lang
if lang != 'en':
    print lang
print "language s " , lang    
###################################################################################


text_file = open("C:\\Python_Eclipse\\jan04_tweets\\tweets_en_only.txt", "w")
text_file.write("%s" % str)
text_file.close()


###############################################################################


##################################################################################
####################### TF - IDF Word count #####################################

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = tweet_after_lemma
print corpus
vectorizer = TfidfVectorizer(min_df=10)
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
vect_tweets=  dict(zip(vectorizer.get_feature_names(), idf))
print vect_tweets



############################ LDA #################################################

import lda
import lda.datasets
import numpy as np
import textmining

with open('C:\\Python_Eclipse\\jan04_tweets\\tweets_head_1000_terms_only.txt', 'r') as myfile:
    data=myfile.read()
    
tdm = textmining.TermDocumentMatrix()
tdm.add_doc(data)
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
    
###################### Lemmatizer ################################################
tweet_after_lemma=[]
import nltk
lemma = nltk.wordnet.WordNetLemmatizer()
for words in tweet_list:
    tweet_after_lemma.append(lemma.lemmatize(words))

print tweet_after_lemma
np.savetxt(r"C:\\Python_Eclipse\\jan04_tweets\\tweets_head_1000_terms_lemma.txt", tweet_list, fmt='%s')
#####################################################################################

print tweetcontents
    if term not in stop_words and  not term.startswith(('#', '@','\x'))] 
terms_only_list.append(terms_only)
    
sentiment_chars='\u'
for idx, val in enumerate(tweet_after_stop_words):
     print idx,val
     if val.startwith:
         
         
termsLen = len(terms_all)
for x in xrange(1, termsLen):
    print termsLen[x]

##############  Iterate the List of tweets to find whether it contains hashtags ######
################# group them by specific hashtags ################################
############ I consider it as a SUPERVISED LEARNING ##############################

hashtag_with_tweets = defaultdict(list)
hashtag='#'
for idx, val in enumerate(tweet_after_stop_words):
     if any(hashtag in s for s in val):
        for s in val:
            if hashtag in s:
                print s, '\n'
                hashtag_with_tweets[s].append(val)
print hashtag_with_tweets
np.savetxt(r"C:\\Python_Eclipse\\jan04_tweets\\tweets_head_1000_hashtag_with_tweets.txt", hashtag_with_tweets, fmt='%s')

##################################################################################
###################################################################################

import numpy as np
import lda
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(tweet_after_stop_words)

######################### Counter - word count ####################
for tweet in tweet_after_stop_words :
   count_all.update(tweet)
print count_all
tweetforstem = count_all
print tweetforstem
##################################################################

################# Best stemming is achieved by word Lemmatizer ######
import nltk
lemma = nltk.wordnet.WordNetLemmatizer()
lemma.lemmatize('article')
lemma.lemmatize('leaves')
###############################################################
################ Port Stemmmer #######################################
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
stemmedwords=[]
ps = PorterStemmer()
for w in count_all :
    stemmedwords.append(ps.stem(w))
print stemmedwords
####################################################################
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
chunkParser = nltk.RegexpParser(chunkGram)
for stemw in stemmedwords:
    chunked = chunkParser.parse(stemw)
    chunked.draw() 


# Count terms only once, equivalent to Document Frequency
terms_single = set(terms_all)

# Count hashtags only
terms_hash_list=[]
for text in tweets_en:
    terms_hash = [term for term in preprocess(text) 
    if term.startswith('@')]
    terms_hash_list.append(terms_hash)    
print terms_hash_list

terms_only_list=[]
# Count terms only (no hashtags, no mentions)
for text in tweets_en:
    terms_only = [term for term in preprocess(text) 
              if term not in stop_words and
              not term.startswith(('#', '@','\x'))] 
    terms_only_list.append(terms_only)
print terms_only_list

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
terms_only_list[0]
for w in terms_only_list[0]:
   print(ps.stem(terms_only))

#term_with_counters = Counter(dict(terms_all))
#print term_with_counters
################# Bigrams ########################################
from nltk import bigrams 
terms_bigram = bigrams(count_all)   
print terms_bigram
for term in terms_bigram:
        print term
##################### ############# ###################
        
from nltk import bigrams 
terms_bigram = bigrams(tweet_after_stop_words)   
print terms_bigram
for term in terms_bigram:
    print term
terms_bigram_with_count = bigrams(tweet)   
for term_bigram in terms_bigram_with_count:
    print terms_bigram

title = u"Klüft skräms inför på fédéral électoral große"
import unicodedata
unicodedata.normalize('NFKD', title).encode('ascii','ignore')
print title

s='This is some  text that has to be cleaned! it annoying!'
print (word_tokenize(s))
print(s.decode('unicode_escape').encode('ascii','ignore'))

for terms in terms_all:
    ''.join(terms)
    print terms
    print(terms.decode('unicode_escape').encode('ascii','ignore'))
for i in 1:
        print tweets_en[0]
        print str(tweets_en[0])
tweets_en = data5['en']
import unicodedata
title
unicodedata.normalize('NFKD', tweets_en[0]).encode('ascii','ignore')
print word_tokenize(str(tweets_en[0]))


title = tweets_en[0]
type(title)
print title
import unicodedata
unicodedata.normalize('NFKD', title).encode('ascii','ignore')
'Kluft skrams infor pa federal electoral groe'


###################  Pre process  ################################################
import re
 
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
   # emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
#emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    print s
    #if lowercase:
        #tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
############################################################################################

####################### Word tokenizer & Preprocess ##############################
from nltk.tokenize import word_tokenize
for text in tweets_en:
    #terms_all.append(preprocess(text))
    terms_all.append(word_tokenize(text))
    #terms_all.append(word_tokenize(text.decode('utf8')))
print terms_all[0]
################################################################################

    
    

############# Stop words Built for the tweet contends ##############################
import nltk
import string
from sklearn.feature_extraction import text
punctuation = list(string.punctuation)
tweet_stop_words = ['rt','via','RT','\\x80']
stop_words = text.ENGLISH_STOP_WORDS
print len(stop_words)
punctuation
stop_words = stop_words.union(punctuation)
print len(stop_words)
stop_words = stop_words.union(tweet_stop_words)
stop_words
print len(stop_words)   
################################################################################

######################### Languages ##################################
import pycountry
bn =  pycountry.languages.get(iso639_1_code='es')
bn.name
print pycountry.countries.get(code ='en')
print pycountry.countries.get(alpha2='DE')
####################################################################









##################### TF-IDF vectorizer Example #################################

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ['pfft like need train', 'The bath bomb need', 'Mike Huckabee slot open Good time agents make bid', 'Education Dispute Top Story 2014 Insights West Education concern 2014 poll', 'remember little things taught turned man today', 'Good morning everybody Have wonderful day', 'Only states redy available info quality physician care finds', "think I'm gonna guy doesn't allow people wear shoes home", "TY-I don't watch", 'Rumor bout hospice Not True Airball Swing miss continue treatment C missed work Hospice No', "If doesn't don't heart RIP Stuart Scott Merica", 'Congress caught web lies U make jokers write articles Qning incident present same.People turn', 'A tone aiming oust government unacceptable Criticisms welcome continuity Turkish government e', 'S o chics nagging rships planet ones Coasks', "I'm dead inside I'm basically portable graveyard", 'bc currently listening ok love', 'Shit', 'finally thing love Softball', "Winter break ending I'm waiting spring break",'pfft like need train', 'The bath bomb need', 'Mike Huckabee slot open Good time agents make bid', 'Education Dispute Top Story 2014 Insights West Education concern 2014 poll', 'remember little things taught turned man today', 'Good morning everybody Have wonderful day', 'Only states redy available info quality physician care finds', "think I'm gonna guy doesn't allow people wear shoes home", "TY-I don't watch", 'Rumor bout hospice Not True Airball Swing miss continue treatment C missed work Hospice No', "If doesn't don't heart RIP Stuart Scott Merica", 'Congress caught web lies U make jokers write articles Qning incident present same.People turn', 'A tone aiming oust government unacceptable Criticisms welcome continuity Turkish government e', 'S o chics nagging rships planet ones Coasks', "I'm dead inside I'm basically portable graveyard", 'bc currently listening ok love', 'Shit', 'finally thing love Softball', "Winter break ending I'm waiting spring break"]
vectorizer = TfidfVectorizer(min_df=2)
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
vect_tweets=  dict(zip(vectorizer.get_feature_names(), idf))
print vect_tweets
print "deepak"


####################################################################################
####################################################################################

import json
import numpy as np

def load_R_model(filename):
    with open(filename, 'r') as j:
        data_input = json.load(j)
    data = {'topic_term_dists': data_input['phi'], 
            'doc_topic_dists': data_input['theta'],
            'doc_lengths': data_input['doc.length'],
            'vocab': data_input['vocab'],
            'term_frequency': data_input['term.frequency']}
    return data

movies_model_data = load_R_model('data/movie_reviews_input.json')

print('Topic-Term shape: %s' % str(np.array(movies_model_data['topic_term_dists']).shape))
print('Doc-Topic shape: %s' % str(np.array(movies_model_data['doc_topic_dists']).shape))

#######################################################################################
#######################################################################################













import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import unicodedata
import nltk 
import StringIO
from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


TweetsFile = open('C:\\Python_Eclipse\\jan10_demo\\newgroupdata.txt','r+')
yourResult = [line.split(',') for line in TweetsFile.readlines()]
count_vect = CountVectorizer(input="file")
docs_new = [ StringIO.StringIO(x) for x in yourResult ]
X_train_counts = count_vect.fit_transform(docs_new)
vocab = count_vect.get_feature_names()
print X_train_counts.shape


print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, #max_features=n_features,
                                   stop_words='english')
                                   
tfidf = tfidf_vectorizer.fit_transform(yourResult)





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