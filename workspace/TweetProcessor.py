# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 19:30:48 2015

@author: ekardee
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 23:20:46 2015

@author: ekardee
"""
import pandas as pd
import nltk
import string
from sklearn.feature_extraction import text

class TweetProcessor:
   'Common base class for all employees'
   empCount = 0
   tweetFields = ['text','user_lang']
   CHUNKSIZE  = 10 ** 6
   TextFileReader 
   def __init__(self, name):
      self.filename = name
      TweetProcessor.empCount += 1
      
   def load_csv(name):
       self.filename = name
       df = pd.read_csv(self.filename,error_bad_lines=False, low_memory=False,usecols=tweetFields,chunksize = CHUNKSIZE)
          
   def processTweet(self):
       masterdata = defaultdict(list)
        data = defaultdict(list)
        for chunk in df:
            for row in chunk.itertuples():
                data[row[2]].append(row [1])
                np.savetxt(r"C:\\Python_Eclipse\\jan09_tweets\\tweets%s.txt"%i, data['en'], fmt='%s')
            break        
        
   def displayCount(self):
     print "Total Employee %d" % TweetProcessor.empCount

   def displayEmployee(self):
      print "Name : ", self.name

    def stopwords(stopwordsList):
        punctuation = list(string.punctuation)
        tweet_stop_words = ['rt','via','RT','\\x80']
        stop_words = text.ENGLISH_STOP_WORDS
        punctuation
        stop_words = stop_words.union(punctuation)
        stop_words = stop_words.union(tweet_stop_words)
        stop_words
        print len(stop_words)
        
        
        
tp = TweetProcessor("C:\\ISB\\BigTapp\\TwitterBigTap\\data_2015-01-10.csv")
tp.load
emp1.processTweet()

print "Total Employee %d" % Employee.empCount



print data[0]

print tweet_after_stop_words

for tweet in tweet_after_stop_words :
    count_all.update(tweet)