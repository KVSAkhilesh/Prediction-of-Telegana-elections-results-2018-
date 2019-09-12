#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:21:15 2018

@author: navyarao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 23:42:44 2018

@author: navyarao
"""

import tweepy
import csv
import pandas as pd
####input your credentials here
#consumer_key = 'j2wuJV2vnMqx5dCXWHxxUQFtB'
#consumer_secret = 'ssWXFkdHQdT9ZXcTO5HX6XfrWIq2VBXa1ZCYqxY6FfhHVvAQgO'
#access_token = '1071035302297317378-Xu5VAUfrrXNXA461bvtXtaQoeElvyL'
#access_token_secret = '2PAC9qYablFnS1Vu78fw16GHgfHxnKkZYiJSbo0mmh7It'

consumer_key = 'j2wuJV2vnMqx5dCXWHxxUQFtB'

consumer_secret = 'ssWXFkdHQdT9ZXcTO5HX6XfrWIq2VBXa1ZCYqxY6FfhHVvAQgO'

access_token = '1071035302297317378-Xu5VAUfrrXNXA461bvtXtaQoeElvyL'

access_token_secret = '2PAC9qYablFnS1Vu78fw16GHgfHxnKkZYiJSbo0mmh7It'



auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
#####United Airlines
# Open/Create a file to append data
csvFile = open('C:/Users/Akhilesh/Downloads/Practicum/tpol.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

#for tweet in tweepy.Cursor(api.search,q="#TRSParty",count=10,
#                           lang="en",
#                           since="2017-04-03",untill="2018-12-06").items():
#    print (tweet.created_at, tweet.text)
#    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
    
import time
from time import sleep    

tweek=[]
IDS=[]
Date=[]
Location=[]
for tweet_info in tweepy.Cursor(api.search, q="#RRRTitle",include_entities=True,
                       monitor_rate_limit=True, 
                       wait_on_rate_limit=True,
                       wait_on_rate_limit_notify = True,
                       retry_count = 5, #retry 5 times
                       retry_delay = 5, lang = 'en', tweet_mode='extended').items(5000):
    #print(tweet_info)
    if 'retweeted_status' in dir(tweet_info):
        #print(tweet_info)
        date=tweet_info.retweeted_status.created_at
        Date.append(date)
        ids=tweet_info.retweeted_status.id_str
        IDS.append(ids)
        locate=tweet_info.author.location
        Location.append(locate)
        #print(tweet_info)
        tweet=tweet_info.retweeted_status.full_text
        tweek.append(tweet)
        time.sleep(5)
    else:
        date=tweet_info.created_at
        Date.append(date)
        ids=tweet_info.id_str
        IDS.append(ids)
        locate=tweet_info.author.location
        Location.append(locate)
        tweet=tweet_info.full_text
        tweek.append(tweet)
        time.sleep(5)
        

        
full_tweets_telangana_TRSparty= pd.DataFrame({'tweets':tweek})
Date_TRSparty=pd.DataFrame({'Date':Date})
Location_TRSparty=pd.DataFrame({'Location':Location})




full_tweets_telangana_TRSparty.to_csv("C:/Users/Akhilesh/Downloads/Practicum/TelanganaPolls.csv")

n=pd.unique(full_tweets_telangana_TRSparty['tweets'].values.ravel('K'))

frames = [full_tweets_telangana_TRSparty,Date_TRSparty,Location_TRSparty]
TRSparty = pd.concat(frames)

TRSparty.to_csv("C:/Users/Akhilesh/Downloads/Practicum/TelanganaPolls.csv")




#INCTelangana


#TRSParty
#mahakutami