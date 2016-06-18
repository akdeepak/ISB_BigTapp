# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 16:24:54 2015

@author: ekardee
"""

print "Deepak"

#######################  EXTRACT DATA AND STARTED TO APPLY NLTK PACKAGE ##################
####################### Save the extracted tweet into the Text file #####################
import pandas as pd
import numpy as np
from collections import defaultdict
data = defaultdict(list)
filename_jan04 = "C:\\ISB\\BigTapp\\TwitterBigTap\\data_jan_4_2015.csv"
tweetFields = ['text','user_lang']
df = pd.read_csv(filename_jan04,error_bad_lines=False, low_memory=False,usecols=tweetFields)
for row in df.itertuples():
    data[row[2]].append(row [1])

tweets_en = data['en']
#print tweets_en

import operator
from collections import Counter
terms_all = []
tweet_after_stop_words = []
count_all = Counter()
for text in tweets_en:
    terms_all.append(preprocess(text))
for terms in terms_all:
    tweet_after_stop_words.append([i for i in terms if i not in stop_words])

for tweet in tweet_after_stop_words :
    count_all.update(tweet)
    
from nltk import bigrams 
terms_bigram = bigrams(tweet_after_stop_words)   
for term in terms_bigram:
    terms_bigram_with_count = bigrams(term)   
print terms_bigram
np.savetxt(r"C:\\Python_Eclipse\\jan04_tweets\\tweets1_bigrams.txt", terms_bigram_with_count, fmt='%s')



tweets_es = data5['es']
tweets_tr = data5['tr']
for en in tweets_en:
    print en
#print data.keys() 
## store the keys in text file 
np.savetxt(r"C:\\Python_Eclipse\\jan04_tweets\\tweets1_keys.txt", data.keys(), fmt='%s')
####
eng_Tweet_text = data['en']
np.savetxt(r"C:\\Python_Eclipse\\jan04_tweets\\tweets1.txt", data['en'], fmt='%s')
##########################################################################################


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

###################  Pre process  ################################################
import re
 
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
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
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
############################################################################################

####################Languate Translator ---- Translated the tweet contents #############
import goslate
gs = goslate.Goslate()
sw = "Är det de här bladen du jobbat med?Har du möjlighet att titta på detta? Fan vet vad han menar med baremetal"
print gs.translate(sw,'en')

for t in tweets_es:
    print t
    print gs.translate(t, 'en')
#################################################################################
    
tweets['text'] = map(lambda tweet: tweet['text'], tweets_data)
tweets['lang'] = map(lambda tweet: tweet['lang'], tweets_data)
tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data)


import HTMLParser
htmpParser = HTMLParser.HTMLParser()

import nltk
nltk.download()
from nltk.tokenize import word_tokenize
tweet = 'RT @marcobonzanini: just an example! :D http://example.com #NLP'
print(word_tokenize(tweet))

    
    
 ### working on list
 default_data = {
            'item1': 1,
            'item2': 2,
}

default_data.update({'item1': 3})

default_data
#################### LOAD TWEETS of 10 gb file into separate  text file with tweets only
####################### for analyzing and data cleansing ########################

import pandas as pd
import numpy as np
filename_jan10 = "C:\\ISB\\BigTapp\\TwitterBigTap\\data_2015-01-09.csv"
tweetFields = ['text']
CHUNKSIZE  = 10 ** 6
CHUNKSIZE
df = pd.read_csv(filename_jan10,error_bad_lines=False, low_memory=False,chunksize = CHUNKSIZE,usecols=tweetFields )
i=1jn
for data in df:
 np.savetxt(r"C:\\Python_Eclipse\\jan09_tweets\\tweets%s.txt"%i, data, fmt='%s')
 i=i+1


#############################################################33

###### LOAD TWEETS FOR 10 GB ######################

import pandas as pd
filename = "C:\\ISB\\BigTapp\\TwitterBigTap\\data_jan_4_2015.csv"
file_jan10 = 'C:\\ISB\\BigTapp\\TwitterBigTap\\data_2015-01-10.csv'
# Read the file
CHUNKSIZE  = 10 ** 6
CHUNKSIZE 
tweetFields = ['user_id_str','text']
text_file = open("C:\\Python_Eclipse\\test.txt", "w")
datareader = pd.read_csv(file_jan10,error_bad_lines=False, low_memory=False,chunksize = CHUNKSIZE,usecols=tweetFields )
for dr in datareader:
     text_file.write("%s" % reader)
     
text_file.close()
########################################################

########## LOAD TWEETS FOR THE ROW SIZE : 1048575 #######
import pandas as pd

filename_jan10 = "C:\\ISB\\BigTapp\\TwitterBigTap\\data_jan_4_2015.csv"
tweetFields = ['user_id_str','text']
CHUNKSIZE = 50000
text_file = open("C:\\Python_Eclipse\\test.txt", "w")
datareader = pd.read_csv(filename_jan10,error_bad_lines=False, low_memory=False,chunksize= CHUNKSIZE, usecols=tweetFields )
for dr in datareader:
    text_file.write("%s" % dr)   

text_file.close()


##########################################################


######################
import pandas as pd
import numpy as np
fields = ['NAME']
data =pd.read_csv("c:\\ISB\\test.csv",usecols=fields,chunksize=2)
#text_file = open("C:\\Python_Eclipse\\test.txt", "w")
i=1
print("C:\\Python_Eclipse\\test",i)
for reader in data:
    np.savetxt(r"C:\\Python_Eclipse\\test",i, reader, fmt='%s')
    i+=1  

##################################


from pandas import DataFrame,read_csv
fields = ['NAME']
f =pandas.read_csv("c:\\ISB\\test.csv",usecols=fields,chunksize=2)
alist = [];
saved = DataFrame()
for reader in f:
    submission = DataFrame(reader)    
    newvar = alist.append(submission)
    print newvar
    DataFrame.append(reader,ignore_index=True)
submission.to_csv("c:\\ISB\\testOuput.txt",index=False)

text_file = open("C:\\Python_Eclipse\\test.txt", "w")
    
text_file.close()


import pandas
fields = ['NAME']
f =pandas.read_csv("c:\\ISB\\test.csv",usecols=fields,chunksize =2)
for reader in f:
    print reader
    
import pandas
fields = ['NAME']
f =pandas.read_csv("c:\\ISB\\test.csv",usecols=fields,chunksize =2)
print f
for reader in f:
    print reader
    pandas
     pandas.DataFrame.to_csv("C:\\test_spyder\\test.txt")
    #print reader        
    #pandas.to_csv(r'c:\data\pandas.txt', header=None, index=None, sep=' ', mode='a')

# using panda library to Load the complete rows of 1048575 ( 1 M ) records i.e. 7 GB data of csv file
tweetFields = ['user_id_str','text']
io = pandas.read_csv('C:\\ISB\\BigTapp\\TwitterBigTap\\data_jan_4_2015.csv',low_memory=False,usecols=tweetFields)



io.text
len(io.text)
len(io)

import pandas
tweetFields = ['user_id_str','text']
data_jan10 = pandas.read_csv('C:\\ISB\\BigTapp\\TwitterBigTap\\data_2015-01-10.csv',error_bad_lines=False,low_memory=False,usecols=tweetFields)

data_jan10


data_jan10.text
len(data_jan10)


chunksize = 10 ** 6
chunksize
filename = 'C:\\ISB\\BigTapp\\TwitterBigTap\\data_jan_4_2015.csv'
filename
    print(chunk)






import pandas as pd
filename = "C:\\ISB\\BigTapp\\TwitterBigTap\\data_jan_4_2015.csv"
file_jan10 = 'C:\\ISB\\BigTapp\\TwitterBigTap\\data_2015-01-10.csv'
# Read the file
CHUNKSIZE  = 10 ** 6
CHUNKSIZE 
tweetFields = ['user_id_str','text']
datareader = pd.read_csv(file_jan10,error_bad_lines=False, low_memory=False,chunksize = CHUNKSIZE,usecols=tweetFields )
for dr in datareader:
        len(dr)

print "There are %d rows of data"%(result)
# Output the number of rows
print("Total rows: {0}".format(len(data)))
# See which headers are available
print(list(data))

print(" Tweets -----------" + data.text)




###########  CONCAT  Example  ######################################


import pandas as pd
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                       'B': ['B0', 'B1', 'B2', 'B3'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']},
                       index=[0, 1, 2, 3])

result = pd.concat(df1,join='inner')
pieces = {'x': df1}
result = pd.concat(pieces)
result
######################################################################



 ########### Test program to write into files by dynamically creating a file name #####
import pandas as pd
import numpy as np
fields = ['NAME']
df1 =pd.read_csv("c:\\ISB\\test.csv",usecols=fields,chunksize=2)
i=1
for data in df1:
 print data
 np.savetxt(r"C:\\Python_Eclipse\\test%s.txt"%i, data, fmt='%s')
 i=i+1
###########################################################################


################ Program to load the tweets of 1048575 size into separate files #####


import pandas as pd
import numpy as np
filename_jan4 = "C:\\ISB\\BigTapp\\TwitterBigTap\\data_jan_4_2015.csv"
tweetFields = ['user_id_str','text']
CHUNKSIZE = 100000
text_file = open("C:\\Python_Eclipse\\test.txt", "w")
df = pd.read_csv(filename_jan4,error_bad_lines=False, low_memory=False,chunksize= CHUNKSIZE, usecols=tweetFields )
i=1
for data in df:
 print data
 np.savetxt(r"C:\\Python_Eclipse\\jan4_tweets%s.txt"%i, data, fmt='%s')
 i=i+1
####################################################################################