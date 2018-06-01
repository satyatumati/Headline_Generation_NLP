from sklearn.decomposition import TruncatedSVD
import numpy as np

def vector_sum(v1, v2):
  sum_v = []
  for n1, n2 in zip(v1, v2):
    sum_v.append(n1+n2)
  return sum_v

def accumulate_emb(data, embed_dict, word_cnt_dict):
  for line in data:
    word, emb = line[0], line[1:]
    if word in embed_dict:
      embed_dict[word] = vector_sum(embed_dict[word], [float(i) for i in emb])
    else:
      embed_dict[word] = [float(i) for i in emb]
    
    if word not in word_cnt_dict:
      word_cnt_dict[word] = 1
    else:
      word_cnt_dict[word] += 1

  return embed_dict, word_cnt_dict

def divide(l, a):
  for i in range(len(l)):
    l[i] /= a
  return l

def average(embed_dict, word_cnt_dict):
  for w in embed_dict:
    embed_dict[w] = divide(embed_dict[w], word_cnt_dict[w])
  return embed_dict

def save_emb_to_file(embed_dict, fname):
  with open(fname,'w') as fil:
    for w in embed_dict:
      emb = embed_dict[w]
      fil.write('{0} {1}\n'.format(w,' '.join(map(str,emb))))
  fil.close()
  return

if __name__ == "__main__":

  batch_size = 1000
  embed_dict = {}
  word_cnt_dict = {}

  bno = 0
  with open("new-elmo_embed_nr-10000_bsiz-100.txt", 'r') as f:
    data = []
    for line in f:
      data.append(line.split())
      if len(data) == batch_size:
        bno += 1
        if bno % 10 == 0:
          print("Processing batch ", bno)
        embed_dict, word_cnt_dict = accumulate_emb(data, embed_dict, word_cnt_dict)
        data = []
  f.close()
  print("Done reading file")
  embed_dict = average(embed_dict, word_cnt_dict)
  print("Done averaging embeddings")
  save_emb_to_file(embed_dict, "elmo_embedding_news1000.txt")
  print("Done saving embeddings to file")
  print("vocab size: ", len(word_cnt_dict))
  
  print("Doing SVD")
  X = []
  for w in embed_dict:
    X.append(embed_dict[w])

  svd = TruncatedSVD(n_components=300, n_iter=100, random_state=42)
  X = svd.fit_transform(X)
  print("Done dimension reduction") 
  
  reduce_dict = {}
  for emb, w in zip(X, embed_dict):
    reduce_dict[w] = emb 
  save_emb_to_file(reduce_dict, "elmo_10000_dim300.txt")
  print("Reduced embeddings saved to file")