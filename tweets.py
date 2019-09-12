#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:30:27 2018

@author: navyarao
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk
nltk.download('wordnet')# Gettnig rid of unnecessary warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import emoji
import string
import time

#%matplotlib inline

trs1=pd.read_csv("/Users/navyarao/Downloads/scraped _data/final_VoteForKCR.csv")
trs2=pd.read_csv("/Users/navyarao/Downloads/scraped _data/VoteForTRS.csv",encoding='cp1252')
trs3=pd.read_csv("/Users/navyarao/Downloads/scraped _data/Selenium/voteforcar.csv")
trs4=pd.read_csv("/Users/navyarao/Downloads/scraped _data/Selenium/TelanganaWithKCR.csv") 
#trs5=pd.read_csv("/Users/navyarao/Downloads/scraped _data/Selenium/PhirEkBaarKCR.csv")
trs6=pd.read_csv("/Users/navyarao/Downloads/scraped _data/final_TRSParty.csv")
trs7=pd.read_csv("/Users/navyarao/Downloads/scraped _data/Selenium/kcr.csv")
trs8=pd.read_csv("/Users/navyarao/Downloads/#HarishRao.csv")
trs9=pd.read_csv("/Users/navyarao/Downloads/scraped _data/Selenium/KTR.csv")

#trs3=pd.read_csv("/Users/navyarao/Downloads/scraped _data/VoteForCAR.csv")
#trs4=pd.read_csv("/Users/navyarao/Downloads/scraped _data/VoteForTRS.csv")

import datetime


#trs1,trs2,trs6 belongs to API and trs3,4,5 belongs to selenium group.
final_trs_api=pd.concat([trs1,trs2,trs6],axis=0)

#167 212

final_trs_sele=pd.concat([trs3,trs4,trs7,trs8,trs9],axis=0)
final_trs_sub=pd.DataFrame(final_trs_sele[['created_at','text']])
final_trs_sub['Location']='None'
final_trs_sub=final_trs_sub.rename(columns={"created_at":"Date", "text":"tweets"})
final_trs_sub = final_trs_sub[['tweets','Date','Location']]

dates=[]
for i in range(1,len(final_trs_sub)):
    your_dt=datetime.datetime.fromtimestamp(int(final_trs_sub.iloc[i,1])/1000)
    your_dt=your_dt.strftime("%Y-%m-%d %H:%M:%S")
    dates.append(your_dt)
    
dates=pd.DataFrame(dates,columns=['Date'])

final_trs_sub['Date']=dates['Date']


#final_trs_sub=final_trs_sub.rename(columns={"created_at":"Date", "text":"tweets"})


## BInd api and selenium

final_trs_api=final_trs_api[['tweets','Date','Location']]

final_trs=pd.concat([final_trs_api,final_trs_sub],axis=0)

final_trs['Date']=pd.to_datetime(final_trs['Date'])

final_trs=final_trs.loc[final_trs['Date'] < '2018-12-06 18:58:22']

final_trs = final_trs[final_trs.tweets!= 'congartulations']
final_trs = final_trs[final_trs.tweets!= 'swearinginceremony']
final_trs = final_trs[final_trs.tweets!= 'oathceremony']

final_trs = final_trs[final_trs.tweets!= 'Hearty congartulations']


kutami1=pd.read_csv("/Users/navyarao/Downloads/scraped _data/final_mahakutami.csv")
kutami2=pd.read_csv("/Users/navyarao/Downloads/scraped _data/Prajakutami.csv")
kutami3=pd.read_csv("/Users/navyarao/Downloads/scraped _data/VoteForMahaKutami.csv")
kutami4=pd.read_csv("/Users/navyarao/Downloads/scraped _data/Mahakutami.csv")



#final_trs=pd.concat([trs1,trs2,trs3,trs4,trs5,trs6],axis=0)

#final_trs['party']="trs"

final_kutami=pd.concat([kutami1,kutami2,kutami3,kutami4],axis=0)

#inal_kutami['party']="kutami"

#count_part=pd.DataFrame()
#count_part['party']=final_trs['party']
#

#import numpy as np
#count_part=pd.concat([count_part,final_kutami['party']],axis=0)

final_trs_depuclicate = final_trs.drop_duplicates()


final_kutami_depuclicate = final_kutami.drop_duplicates()


#fin_text=[]
#for st in final_trs_depuclicate['tweets']:
#    new=re.sub('[^A-Za-z0-9]+', ' ', str(st))
#    fin_text.append(new)


####### REMOVING ROWS CONTAINING BELOW WORDS



translator = str.maketrans('', '', string.punctuation)




######################################################## KUTAMI
def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text

words = set(nltk.corpus.words.words())

fin_text_trs=[]
for st in final_kutami_depuclicate['tweets']:
    no_hash=" ".join(filter(lambda x:x[0]!='#', st.split()))
    sent=" ".join(w for w in nltk.wordpunct_tokenize(no_hash)if w.lower() in words or not w.isalpha())
    if len(sent)>0:
        translator = str.maketrans('', '', string.punctuation)
        text=no_hash.translate(translator)
        emo_free=give_emoji_free_text(text)
        fin_text_trs.append(emo_free)




all_sent=[]
for sen in fin_text_trs:
    sent=re.sub('[0-9]+', '', sen)
    all_sent.append(sent)



#123 to 644
clean_text_kutami=pd.DataFrame(all_sent,columns=['tweets'])

clean_text_kutami=clean_text_kutami[~clean_text_kutami.isin(['nan']).any(axis=1)]

clean_text_kutami=clean_text_kutami['tweets'].str.lower()

clean_text_kutami=pd.DataFrame(clean_text_kutami)


final_kutami_depuclicate['tweets']=clean_text_kutami['tweets']

final_kutami_depuclicate = final_kutami_depuclicate[final_kutami_depuclicate.tweets!= 'congartulations']
final_kutami_depuclicate = final_kutami_depuclicate[final_kutami_depuclicate.tweets!= 'swearinginceremony']
final_kutami_depuclicate = final_kutami_depuclicate[final_kutami_depuclicate.tweets!= 'oathceremony']

sep=' '



final_kutami_depuclicate['Dates'] = final_kutami_depuclicate['Date'].str.split(' ').str[0]

final_kutami_depuclicate=final_kutami_depuclicate[['tweets','Location','Dates']]


final_kutami_depuclicate = final_kutami_depuclicate[final_kutami_depuclicate.tweets!= 'swearinginceremony']
final_kutami_depuclicate = final_kutami_depuclicate[final_kutami_depuclicate.tweets!= 'oathceremony']

final_kutami_depuclicate['Dates']=pd.to_datetime(final_kutami_depuclicate['Dates'])
final_kutami_depuclicate=final_kutami_depuclicate.loc[final_kutami_depuclicate['Dates'] < '2018-12-06 18:58:22']




#from datetime import datetime
#polarity=[]
#for i in range(0,len(final_kutami_depuclicate['tweets'])):
#    pola=list(TextBlob(final_kutami_depuclicate.iloc[i,0]).sentiment)
#    polarity.append(pola)


#polari = pd.DataFrame(polarity, columns=['polarity','subjectivity'])

#polari = polari.reset_index(drop=True)
#final_kutami_depuclicate = final_kutami_depuclicate.reset_index(drop=True)

#final_kutami_depuclicate=pd.concat([final_kutami_depuclicate,polari],axis=1)


#################################### TRS

all_sent=[]
for sen in final_trs_depuclicate['tweets']:
    sent=re.sub('[0-9]+', '', str(sen))
    all_sent.append(sent)

words = set(nltk.corpus.words.words())

fin_text_trs=[]
for st in all_sent:
    no_hash=" ".join(filter(lambda x:x[0]!='#', st.split()))
    sent=" ".join(w for w in nltk.wordpunct_tokenize(no_hash) if w.lower() in words or not w.isalpha())
    if len(sent)>0:
        translator = str.maketrans('', '', string.punctuation)
        text=no_hash.translate(translator)
        emo_free=give_emoji_free_text(text)
        fin_text_trs.append(emo_free)




clean_text_trs=pd.DataFrame(fin_text_trs,columns=['tweets'])
clean_text_trs=clean_text_trs.dropna(how='all')    #to drop if all values in the row are nan


final_trs_depuclicate['tweets']=clean_text_trs['tweets']

final_trs_depuclicate=final_trs_depuclicate.dropna()

#final_trs_depuclicate=final_trs_depuclicate[~final_trs_depuclicate.isin(['nan']).any(axis=1)]

#from datetime import datetime
#polarity=[]
#for i in range(0,len(final_trs_depuclicate['tweets'])):
#    pola=list(TextBlob(final_trs_depuclicate.iloc[i,0]).sentiment)
#    polarity.append(pola)
#
#
#polari_trs = pd.DataFrame(polarity, columns=['polarity','subjectivity'])
#
#polari_trs = polari_trs.reset_index(drop=True)
#final_trs_depuclicate = final_trs_depuclicate.reset_index(drop=True)
#
#final_trs_depuclicate=pd.concat([final_trs_depuclicate,polari_trs],axis=1)
#






from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

from nltk import tokenize

sid = SentimentIntensityAnalyzer()

final_trs_depuclicate['sentiment_compound_polarity']=final_trs_depuclicate.tweets.apply(lambda x:sid.polarity_scores(x)['compound'])
final_trs_depuclicate['sentiment_neutral']=final_trs_depuclicate.tweets.apply(lambda x:sid.polarity_scores(x)['neu'])
final_trs_depuclicate['sentiment_negative']=final_trs_depuclicate.tweets.apply(lambda x:sid.polarity_scores(x)['neg'])
final_trs_depuclicate['sentiment_pos']=final_trs_depuclicate.tweets.apply(lambda x:sid.polarity_scores(x)['pos'])
final_trs_depuclicate['sentiment_type']=''
final_trs_depuclicate.loc[final_trs_depuclicate.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
final_trs_depuclicate.loc[final_trs_depuclicate.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
final_trs_depuclicate.loc[final_trs_depuclicate.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'
final_trs_depuclicate.head()

final_trs_depuclicate.sentiment_type.value_counts().plot(kind='bar',title="sentiment analysis")





final_kutami_depuclicate['sentiment_compound_polarity']=final_kutami_depuclicate.tweets.apply(lambda x:sid.polarity_scores(x)['compound'])
final_kutami_depuclicate['sentiment_neutral']=final_kutami_depuclicate.tweets.apply(lambda x:sid.polarity_scores(x)['neu'])
final_kutami_depuclicate['sentiment_negative']=final_kutami_depuclicate.tweets.apply(lambda x:sid.polarity_scores(x)['neg'])
final_kutami_depuclicate['sentiment_pos']=final_kutami_depuclicate.tweets.apply(lambda x:sid.polarity_scores(x)['pos'])
final_kutami_depuclicate['sentiment_type']=''
final_kutami_depuclicate.loc[final_kutami_depuclicate.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
final_kutami_depuclicate.loc[final_kutami_depuclicate.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
final_kutami_depuclicate.loc[final_kutami_depuclicate.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'
final_kutami_depuclicate.head()

final_kutami_depuclicate.sentiment_type.value_counts().plot(kind='bar',title="sentiment analysis")



#############################

trs_negative=final_trs_depuclicate.loc[final_trs_depuclicate['sentiment_type'] == 'NEGATIVE']

trs_positive=final_trs_depuclicate.loc[final_trs_depuclicate['sentiment_type'] == 'POSITIVE']


kutami_negative=final_kutami_depuclicate.loc[final_kutami_depuclicate['sentiment_type'] == 'NEGATIVE']

kutami_positive=final_kutami_depuclicate.loc[final_kutami_depuclicate['sentiment_type'] == 'POSITIVE']





#to drop if all values in the row are nan


#final_trs_depuclicate['tweets'] = final_trs_depuclicate[final_trs_depuclicate.tweets!= 'congartulations']
#final_trs_depuclicate['tweets'] = final_trs_depuclicate[final_trs_depuclicate.tweets!= 'swearinginceremony']
#final_trs_depuclicate ['tweets']= final_trs_depuclicate[final_trs_depuclicate.tweets!= 'oathceremony']

##############################################  TOPIC MODELLING ########################################



from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
stop.update(['trs','kcr','ktr','ktrtrs','chusii','take','ga','get','medium ','r','t','goppa','vallantha','sir','anukunna','telangana','election','trsparty','revanth','galla','trspartyonline','side','jai','reddy','way','election','revanthreddy','vote','voteforcar','telangana','election','telanganaelections','people','election','one','amp','medium'])
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

#########################################  topic modelling for trs_negative

doc_clean = [clean(doc).split() for doc in trs_negative['tweets']]

#new_tweet=final_kutami_depuclicate['tweets']
#stop_free_tweets = " ".join([i for i in new_tweet.iloc[i].lower().split() if i not in stopwords])


# Importing Gensim

import gensim
from gensim import corpora
from gensim.corpora import Dictionary


dictionary = Dictionary(doc_clean)  # initialize a Dictionary

dt_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


# Creating the object for LDA model using gensim library


import gensim
NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(dt_matrix, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

#########################################  topic modelling for kutami_negative

doc_clean_kutami_negative = [clean(doc).split() for doc in kutami_negative['tweets']]

#new_tweet=final_kutami_depuclicate['tweets']
#stop_free_tweets = " ".join([i for i in new_tweet.iloc[i].lower().split() if i not in stopwords])


# Importing Gensim

import gensim
from gensim import corpora
from gensim.corpora import Dictionary


dictionary = Dictionary(doc_clean_kutami_negative)  # initialize a Dictionary

dt_matrix = [dictionary.doc2bow(doc) for doc in doc_clean_kutami_negative]


# Creating the object for LDA model using gensim library


import gensim
NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(dt_matrix, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)



final_kutami_depuclicate['Dates']=pd.to_datetime(final_kutami_depuclicate['Dates'])



final_kutami_depuclicate.reset_index(drop=True, inplace=True)

final_kutami_depuclicate1 = final_kutami_depuclicate.groupby(['Dates'])['sentiment_compound_polarity'].agg(['sum', 'idxmax'])



final_kutami_depuclicate1['Date'] = final_kutami_depuclicate1.index


plt.plot_date(x=final_kutami_depuclicate1['Date'], y=final_kutami_depuclicate1['sum'], fmt="r-")
plt.title("Pageessions on example.com")
plt.ylabel("Page impressions")
plt.grid(True)
plt.show()





from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
  
# Reads 'Youtube04-Eminem.csv' file  

  
comment_words = ' '
stopwords = set(STOPWORDS) 
# iterate through the csv file 




final_kutami_depuclicate['tweets']=new_tweet

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


show_wordcloud(trs_positive['tweets'])


#comment_words = ' '
#stopwords = set(STOPWORDS) 
#stopwords.update(['will','night','httpstcosntnibyqh','ok'])
## iterate through the csv file 
#for val in trs_positive.tweets: 
#      
#    # typecaste each val to string 
#    val = str(val) 
#  
#    # split the value 
#    tokens = val.split() 
#      
#    # Converts each token into lowercase 
#    for i in range(len(tokens)): 
#        tokens[i] = tokens[i].lower() 
#          
#    for words in tokens: 
#        comment_words = comment_words + words + ' '
#  
#  
#wordcloud = WordCloud(width = 800, height = 800, 
#                background_color ='white', 
#                stopwords = stopwords, 
#                min_font_size = 10).generate(comment_words) 
#  
## plot the WordCloud image                        
#plt.figure(figsize = (8, 8), facecolor = None) 
#plt.imshow(wordcloud) 
#plt.axis("off") 
#plt.tight_layout(pad = 0) 
#  
#plt.show() 






import nltk
words = set(nltk.corpus.words.words())

sent = "andharu dongale appudu"
" ".join(w for w in nltk.wordpunct_tokenize(sent) \
         if w.lower() in words or not w.isalpha())











