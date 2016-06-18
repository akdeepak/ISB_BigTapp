# -*- coding: utf-8 -*-
"""
Created on Sat Jan 09 17:09:58 2016

@author: ekardee
"""

import pandas as pd
import numpy as np
from collections import defaultdict
data = defaultdict(list)
filename_jan04 = "C:\\ISB\\BigTapp\\TwitterBigTap\\data_jan_4_2015.csv"
tweetFields = ['text','user_lang']
df = pd.read_csv(filename_jan04,error_bad_lines=False, low_memory=False,usecols=tweetFields)
for row in df.itertuples():
    data[row[2]].append(row [1])
    

df5 = df.head(n=10000)
data5 = defaultdict(list)
for row in df5.itertuples():
    data5[row[2]].append(row [1])
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
    