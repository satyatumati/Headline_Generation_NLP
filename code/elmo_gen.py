
# coding: utf-8

# In[12]:


from __future__ import print_function
# from allennlp.modules.elmo import _ElmoBiLm
from allennlp.modules.elmo import Elmo
import torch.nn as nn
import torch
from torch import FloatTensor,LongTensor,ByteTensor, Tensor
import torch.nn.functional as f
from collections import OrderedDict
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper
from torch.autograd import Variable
import nltk,os
import numpy as np
import argparse
import pickle
from allennlp.modules.elmo import Elmo
#from nn_layer import EmbeddingLayer, Encoder
from allennlp.commands.elmo import ElmoEmbedder

def get_sentences(nr):
    df = pickle.load(open('pickles/json_news_'+str(nr)+'.pickle', 'rb'))
    heads = df['title'].tolist()
    desc = df['content'].tolist()
    print(len(heads), " news is loaded!")
    
    sentences = []
    for h, d in zip(heads, desc):
      sentences.append(h[0])
      cnt = 0
      for l in d:
        sentences.append(l)
        cnt += len(l)
        if cnt > 50:
          break
    print(len(sentences), " sentences is loaded.")
    return sentences


# In[17]:




def elmo_sent_mapper(sentence, max_length, pad_token="~"):
    word_list = []
    for i in range(max_length):
        word = sentence[i] if i < len(sentence) else pad_token
        word_list.append(ELMoCharacterMapper.convert_word_to_char_ids(word))
    return word_list

def get_batches(data, batch_size):
    batched_data = []
    for i in range(len(data)):
        if i % batch_size == 0:
            batched_data.append([data[i]])
        else:
            batched_data[len(batched_data) - 1].append(data[i])
    return batched_data

def batch_sentence_mapper(batch, maxl):
    return Variable(LongTensor([elmo_sent_mapper(sent,maxl) for sent in batch]))
    # return batch_to_ids(batch)

def store_batch_embeddings(sl, emb_red, num_rec, batch_size, max_sent_len):
    num_sent = len(sl)
    emb_red = emb_red.data.numpy()
    count = 0
    #print(sum([len(s) for s in sl] ))
    with open(embedding_file,'a+') as fil:
        for i in range(num_sent):
            for j in range(len(sl[i])):
                word_embedding = emb_red[i][j]
                fil.write('{0} {1}\n'.format(sl[i][j],' '.join(map(str,word_embedding))))
                count+=1
    return count

def get_elmo_embeddings(sl, num_rec, batch_size):
    if os.path.exists(embedding_file):
        print(embedding_file," already exists. Do you still want to proceed?")
        x = input("(y/n) ")
        if x=='y':
            print("Continuing..")
        else:
            return
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo_embedder = Elmo(options_file, weight_file,1)
    batched_data = get_batches(sl, batch_size)
    print("\t{0} sentences in {1} records and generated {2} batches each of {3} sentences".format(len(sl),num_rec,len(batched_data),batch_size))
    bno = 0
    wc  = 0
    for batch in batched_data:
        max_sent_len = max([len(s) for s in batch])
        mapped_sentences = batch_sentence_mapper(batch, max_sent_len)
        act = elmo_embedder(mapped_sentences)['elmo_representations']
        emb_red = act[0]
        cnt = store_batch_embeddings(batch, emb_red, num_rec,batch_size, max_sent_len)
        bno+=1
        wc +=cnt
        print("\t\tStored batch {0} [with {1} words]".format(bno,cnt))
    print("Generated embeddings for data with {0} words".format(wc)) 
    return 



# In[18]:


if __name__ == "__main__":
  num_records = 10000
  batch_size = 100

  print(" *********** Generating 1024 dimension embeddings for {0} news articles with batch size {1} *************".format(num_records,batch_size))

  embedding_file = 'elmo_embedding_nr-{0}_bsiz-{1}.txt'.format(num_records,batch_size)

  DIR = './pickles'
  nltk.download('punkt')
  english_sent_tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

  print('Getting sentence lists')

  sent_list=get_sentences(num_records)
  split_sent_list = [[w.lower() for w in s.split()] for s in sent_list]
  print('Splitting sentence lists and converting to lower case words')

  lengths=[len(s) for s in split_sent_list]
  wc = sum(lengths)
  print("\n\nNow, Generating embeddings for data with {0} words".format(wc))
  #plt.hist(lengths, bins=np.arange(min(lengths), max(lengths)+1))
  #plt.plot()

  get_elmo_embeddings(split_sent_list, num_records, batch_size)

