
# coding: utf-8

# Import statements

# In[ ]:


#import statements
import csv
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import PunktSentenceTokenizer
import pickle
from nltk.stem import *
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
import nltk.text


# # Relative locations of datasets
# Give the number of rows (examples) wished to extract

# In[22]:


AP_NEWS_DIR = 'datasets/apnews'  
ALL_NEWS_DIR = '~/.kaggle/datasets/snapcrack/all-the-news' #https://www.kaggle.com/snapcrack/all-the-news/data
NUM_ROWS_EXTRACT = 5000


# Loading Punctuation for english language. You might need to dowload using nltk.download()

# In[23]:


english_sent_tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')


# __Description__:<br>
# This method removes the location from the beginning of the news article.<br> 
# __Input__ :<br>
# News Article Content<br> 
# __Output__:<br>
# Article with location removed<br>
# __Example__:<br>
# Washington - President announces ..." is changed to "President announces ..."

# In[24]:


def remove_location_from_news(text):
    ts = text.split('—')
    if(len(ts[0])< 35):
        #print(ts[0])
        return '—'.join(ts[1:])
    return text
        


# __Description__:<br>
# This method splits the article into meaningful sentences, lemmatizes the words and removes puntuation from the input text.<br>
# __Input__ :<br>
# >text: News Article Content or title<br>
# use_lemmatizer: Whether to use lemmatizer or not<br>
# use_stemmer : Whether to use lemmatizer or not<br>
# interval: Number of rows(interval) to print status of processing
# 
# __Output__:<br>
# Processed News Article Content or title<br>

# In[25]:


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
def tokenize_text(text, use_lemmatizer = True, use_stemmer = False,interval=1000):
    global error_count, run_count
    text = remove_location_from_news(text)
    run_count+=1
    if(run_count%interval==1):
        print(run_count)
    try:
        #print(text)
        sent_l = english_sent_tokenizer.tokenize(text)
        if(use_lemmatizer):
            sent_l = [' '.join([w_al for w_al in [lemmatizer.lemmatize(w) for w in nltk.wordpunct_tokenize(sent)] if w_al.isalnum()]) for sent in sent_l]
        if(use_stemmer):
            sent_l = [' '.join([w_al for w_al in [stemmer.stem(w) for w in nltk.wordpunct_tokenize(sent)] if w_al.isalnum()]) for sent in sent_l]
        #print(sent_l)
        return sent_l
    except Exception as e:
        print(e)
        print("Couldn't tokenize :")
        error_count+=1
        #print((text))
        return [text]


# __Description__:<br>
# This method is a wrapper that preprocesses the title and content of the news dataframe.<br> 
# __Input__ :<br>
# News data frame with 'content' and 'title' columns<br> 
# __Output__:<br>
# Processed News data frame with 'content' and 'title' columns<br>

# In[26]:


def parse_dataframe(df):
    df['content']=df['content'].apply(lambda x : tokenize_text(x))
    df['title']=df['title'].apply(lambda x : tokenize_text(x))
    return df


# **Description:**<br>
# This method converts a dataframe into a pickle format<br>
# **Input:**<br>
# Dataframe<br>
# **Output:**<br>
# Dumped pickle file containing heads, desc, and keywords (not used when training)<br>

# In[27]:


def tuple2pickle(df,nr):
    heads, desc = [], []
    for index, row in df.iterrows():
        if(len(row['title'])>=1):
            heads.append(row['title'])
            desc.append(row['content'])
    with open('pickles/all-the-news_'+nr+'.pickle', 'wb') as f:
        pickle.dump([heads, desc, None], f, pickle.HIGHEST_PROTOCOL)
    print('Extracting rows into ', 'pickles/all-the-news_'+nr+'.pickle')


# __Description__:<br>
# This method loads, cleans, preprocesses and returns the "all the news" dataset as a dataframe with 'content' and 'title' columns.<br> 
# __Input__ :<br>
# >partial: Whether to load partial data or complete data<br>
# rows: Number of rows to be processed if partial is True
# 
# __Output__:<br>
# Processed News data frame with 'content' and 'title' columns<br>

# In[28]:


error_count = 0
run_count = 0 
def get_all_news_df(partial= True,rows = 5000):
    global error_count, run_count
    error_count = 0
    run_count = 0 
    df = pd.read_csv(ALL_NEWS_DIR+'/articles1.csv')
    df = df.append(pd.read_csv(ALL_NEWS_DIR+'/articles2.csv'))
    df = df.append(pd.read_csv(ALL_NEWS_DIR+'/articles3.csv'))
    df2 = df[['title','content']]
    if(partial):
        df3 = parse_dataframe(df2.head(rows))
        tuple2pickle(df3,str(rows))
    else: 
        df3 = parse_dataframe(df2)
        tuple2pickle(df3,'all')
    return df3


# In[29]:


df3 = get_all_news_df(True, NUM_ROWS_EXTRACT)


# In[30]:


df3.head()

