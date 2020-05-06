import gensim
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


## TODO: Implement cosim
def cosim(v1, v2):
  ## return cosine similarity between v1 and v2
  dot_product = np.dot(v1, v2)
  norm_a = np.linalg.norm(v1)
  norm_b = np.linalg.norm(v2)
  return dot_product / (norm_a * norm_b)


def vecref(s):
  (word, srec) = s.split(' ', 1)
  return (word, np.fromstring(srec, sep=' '))

def ftvectors(fname):
  return { k:v for (k, v) in [vecref(s) for s in open(fname)] if len(v) > 1}

def mostSimilar(vec, vecDict):
  ## Use your cosim function from above
  mostSimilar = ''
  similarity = 0
  for k in vecDict.keys():
    similarity_temp = cosim(vec, vecDict[k])
    if similarity_temp >= similarity:
      similarity = similarity_temp
      mostSimilar = k
  return (mostSimilar, similarity)

def populate_language_links(lang_key, links):
  lang_links = []
  for link in links:
    if link[1] == lang_key and link[0] in envec.keys():
      lang_links.append(link)
  return lang_links

def evaluate_generic(list_links, lang_dict):
  accuracy = 0
  baselineAccuracy = 0
  for entry in list_links:
    if entry[0] in envec.keys():
      actual_word = mostSimilar(envec[entry[0]], lang_dict)
      if actual_word[0] == entry[2]:
        accuracy+=1
    if entry[0]== entry[2]:
      baselineAccuracy+=1
  accuracy = accuracy/len(list_links)
  baselineAccuracy = baselineAccuracy/len(list_links)
  return accuracy, baselineAccuracy

# !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/shakespeare_plays.txt
lines = [s.split() for s in open('shakespeare_plays.txt')]
model = Word2Vec(lines)
model.wv.most_similar(positive=['king','woman'], negative=['man'])
model.wv.most_similar(positive=['othello'])
#model.wv.most_similar(positive=['brutus'])
model.wv.similarity('othello', 'desdemona')
cosim(model.wv['othello'], model.wv['desdemona'])

# !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.en.vec
# !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.fr.vec
#
# # TODO: uncomment at least one of these
# # !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.ar.vec
# # !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.de.vec
# # !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.pt.vec
# # !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.ru.vec
# !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.es.vec
# # !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.vi.vec
# # !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.zh.vec

envec = ftvectors('30k.en.vec')
frvec = ftvectors('30k.fr.vec')

# TODO: load vectors for one more language, such as zhvec
# arvec = ftvectors('30k.ar.vec')
# devec = ftvectors('30k.de.vec')
# ptvec = ftvectors('30k.pt.vec')
# ruvec = ftvectors('30k.ru.vec')
esvec = ftvectors('30k.es.vec')
# vivec = ftvectors('30k.vi.vec')
# zhvec = ftvectors('30k.zh.vec')
[mostSimilar(envec[e], frvec) for e in ['computer', 'germany', 'matrix', 'physics', 'yeast']]
# !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/links.tab
links = [s.split() for s in open('links.tab')]
french_list=populate_language_links('fr', links)
french_accuracy, french_baseline_acc = evaluate_generic(french_list, frvec)
print('The Baseline Accuracy for french is : ', french_baseline_acc)
print('The Accuracy for french is : ', french_accuracy)

## TODO: Compute English-X Wikipedia retrieval accuracy.
spanish_list=populate_language_links('es', links)
spanish_accuracy, spanish_baseline_acc = evaluate_generic(spanish_list, esvec)
print('The Baseline Accuracy for Spanish is : ', spanish_baseline_acc)
print('The Accuracy for Spanish is : ', spanish_accuracy)
