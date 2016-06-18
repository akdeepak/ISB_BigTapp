# -*- coding: utf-8 -*-
"""
Created on Sun Jan 03 20:06:38 2016

@author: ekardee
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 22:26:48 2015

@author: ekardee
"""
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


print "deepak"
import pandas as pd
import numpy as np
from collections import defaultdict
import operator
from collections import Counter
from nltk.tokenize import TweetTokenizer
import nltk

filename_jan10 = "C:\\ISB\\BigTapp\\TwitterBigTap\\data_2015-01-10.csv"
tweetFields = ['text','user_lang']
CHUNKSIZE = 10 ** 6
df = pd.read_csv(filename_jan10,error_bad_lines=False, low_memory=False,usecols=tweetFields,chunksize=CHUNKSIZE)
data = defaultdict(list)
masterdata= defaultdict(list)
i=1
for chunk in df:
    data = defaultdict(list)
    for row in chunk.itertuples():
        data[row[2]].append(row [1])
    tweets_en = data['en']
    terms_all = []
    tknzr = TweetTokenizer()
    for text in tweets_en:
        terms_all.append(tknzr.tokenize(text))
    print " tokenized"
    tweet_after_stop_words=[]
    for terms in terms_all:
        tweet_after_stop_words.append([i for i in terms if i not in stop_words])
    print "Stop Words"
    tweetcontents=[]
    users_start_with_at=[]
    hashtags=[]
    web_links=[]
    for idx, val in enumerate(tweet_after_stop_words):
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
    print "Data Cleansing"
    tweet_after_lemma=[]
    lemma = nltk.wordnet.WordNetLemmatizer()
    for words in tweet_after_stop_words:
        tweet_after_lemma.append(lemma.lemmatize(words))
    print "Lemmatization "
    str = ' '.join(tweet_after_lemma)
    text_file = open("C:\\Python_Eclipse\\jan10_tweets\\tweetsasdoc%s.txt"%i, "w")
    text_file.write("%s" % str)
    text_file.close()    
    i=i+1    
    if i == 3:
        break
   
    np.savetxt(r"C:\\Python_Eclipse\\jan10_tweets\\tweets%s.txt"%i, data['en'], fmt='%s')
    if i == 5 :
        print i
        break

for key, value in masterdata.iteritems():
    print value
    break



################ POC of the tweeet data with 20 tweets ###############################################
df5 = df.head(n=10000)
data5 = defaultdict(list)
for row in df5.itertuples():
    data5[row[2]].append(row [1])
print data5
print data5.keys()
tweets_en = data5['en']
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
    
######################################################################################
############ Data Cleaning - Identify the language ##################################
############# Detect Language #######################################################


from langdetect import detect
for tweet in tweets_en:
    lang = detect(tweet)
    print lang
    if lang != 'en':
        print "Detected Language : " ,lang
        data5[lang].append(tweet)

lang = detect("pptf, like, you")
print lang
lang = detect("Ein, zwei, drei, vier")
print lang
if lang != 'en':
    print lang
print "language s " , lang    
###################################################################################

###############################################################################

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



######################## stop words & Data cleansing and writing the tweet contents#######
########################### in a file  ################################################
tweet_after_stop_words=[]
for terms in terms_all:
    tweet_after_stop_words.append([i for i in terms if i not in stop_words])
print tweet_after_stop_words
np.savetxt(r"C:\\Python_Eclipse\\jan04_tweets\\tweets_head_1000_stop_words.txt", terms_all, fmt='%s')


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




tweetcontents=[]
users_start_with_at=[]
hashtags=[]
web_links=[]
for idx, val in enumerate(tweet_after_stop_words):
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
print tweet_list

######################################################################################
###################### Lemmatizer ################################################
tweet_after_lemma=[]
import nltk
lemma = nltk.wordnet.WordNetLemmatizer()
for words in tweet_list:
    tweet_after_lemma.append(lemma.lemmatize(words))
print tweet_after_lemma
######################################################################################

######################################################################################
###################### convert lemmatizer tokenized words into document #############
str = ' '.join(tweet_after_lemma)
print str
text_file = open("C:\\Python_Eclipse\\jan04_tweets\\tweets_head_1000_terms_only.txt", "w")
text_file.write("%s" % str)
text_file.close()

######################################################################################
###################### Lemmatizer ################################################
tweet_after_lemma=[]
import nltk
lemma = nltk.wordnet.WordNetLemmatizer()
for words in tweet_list:
    tweet_after_lemma.append(lemma.lemmatize(words))

print tweet_after_lemma
np.savetxt(r"C:\\Python_Eclipse\\jan04_tweets\\tweets_head_1000_terms_lemma.txt", tweet_list, fmt='%s')
#####################################################################################


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

################# stop words ########################################
for terms in terms_all:
    tweet_after_stop_words.append([i for i in terms if i not in stop_words])
print tweet_after_stop_words
########################################################################


##################################################################################
###################################################################################

################# stop words ########################################
for terms in terms_all:
    tweet_after_stop_words.append([i for i in terms if i not in stop_words])
print tweet_after_stop_words
########################################################################

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
corpus = ["This is very strange",
          "This is very nice"]
vectorizer = TfidfVectorizer(min_df=2)
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
vect_tweets=  dict(zip(vectorizer.get_feature_names(), idf))
print vect_tweets


