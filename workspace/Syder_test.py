# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 16:24:54 2015

@author: ekardee
"""

import pandas

fields = ['NAME']

f =pandas.read_csv("c:\\ISB\\test.csv")
print f
f =pandas.read_csv("c:\\ISB\\test.csv",usecols=fields)
print f

tweetFields = ['user_id_str','text']
io = pandas.read_csv('C:\\ISB\\BigTapp\\TwitterBigTap\\data_jan_4_2015.csv',low_memory=False,usecols=tweetFields)
for data in io:
    print io$text
len(io)