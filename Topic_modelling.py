#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 23:40:16 2018

@author: navyarao
"""

import spacy
spacy_tok = spacy.load('en')
spacy.load('en')
from spacy.lang.en import English
parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    tokens = nltk.wordpunct_tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

url_free_trs_positive=[]
for i in range(0,len(trs_positive['tweets'])):
    new_txt=" ".join(filter(lambda x:x[0:4]!='http', trs_positive['tweets'].iloc[i].split()))
    url_free_trs_positive.append(new_txt)



import random
text_data = []

for line in range(0,len(url_free_trs_positive)):
    tokens = prepare_text_for_lda(url_free_trs_positive[line])
    if random.random() > .99:
        print(tokens)
        text_data.append(tokens)

                      
from gensim import corpora
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')



import gensim
NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
#ldamodel.save('/Users/navyarao/Downloads/Sentiment Analysis Practicum/model5.gensim')
model =  models.LdaModel.load('model5.gensim')
topics = model.print_topics(num_words=4)
for topic in topics:
    print(topic)






######################################3 TRS negative



url_free_trs_negative=[]
for i in range(0,len(trs_negative['tweets'])):
    new_txt=" ".join(filter(lambda x:x[0:4]!='http', trs_negative['tweets'].iloc[i].split()))
    url_free_trs_negative.append(new_txt)



import random
text_data = []

for line in range(0,len(url_free_trs_negative)):
    tokens = prepare_text_for_lda(url_free_trs_negative[line])
    if random.random() > .99:
        print(tokens)
        text_data.append(tokens)

                      
from gensim import corpora
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')



import gensim
NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
#ldamodel.save('/Users/navyarao/Downloads/Sentiment Analysis Practicum/model5_trsneg.gensim')
model =  models.LdaModel.load('model5_trsneg.gensim')
topics = model.print_topics(num_words=4)
for topic in topics:
    print(topic)



######################################3 kutami negative



url_free_kutami_negative=[]
for i in range(0,len(kutami_negative['tweets'])):
    new_txt=" ".join(filter(lambda x:x[0:4]!='http', kutami_negative['tweets'].iloc[i].split()))
    url_free_kutami_negative.append(new_txt)



import random
text_data = []

for line in range(0,len(url_free_kutami_negative)):
    tokens = prepare_text_for_lda(url_free_kutami_negative[line])
    if random.random() > .99:
        print(tokens)
        text_data.append(tokens)

                      
from gensim import corpora
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')



import gensim
NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
#ldamodel.save('/Users/navyarao/Downloads/Sentiment Analysis Practicum/model5_kutneg.gensim')
model =  models.LdaModel.load('model5_kutneg.gensim')
topics = models.print_topics(num_words=4)
for topic in topics:
    print(topic)












######################################3 kutami negative



url_free_kutami_positive=[]
for i in range(0,len(kutami_positive['tweets'])):
    new_txt=" ".join(filter(lambda x:x[0:4]!='http', kutami_positive['tweets'].iloc[i].split()))
    url_free_kutami_positive.append(new_txt)



import random
text_data = []

for line in range(0,len(url_free_kutami_positive)):
    tokens = prepare_text_for_lda(url_free_kutami_positive[line])
    if random.random() > .99:
        print(tokens)
        text_data.append(tokens)

                      
from gensim import corpora
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')



import gensim
NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
#ldamodel.save('/Users/navyarao/Downloads/Sentiment Analysis Practicum/model5_kutneg.gensim')
model =  models.LdaModel.load('model5_kutneg.gensim')
topics = models.print_topics(num_words=4)
for topic in topics:
    print(topic)















