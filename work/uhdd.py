
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
from optparse import OptionParser
import logging as log
from os.path import isfile
from tqdm import tqdm
from scipy import spatial
import warnings
warnings.filterwarnings("ignore")

# python class.py -w dataset_task2.txt -e word_embed1 -r word_embed2 -a 0.5 -f cosine

# In[2]:


# command line argument partsing
parser = OptionParser()
parser.add_option('-e',
                  '--embeddings-file',
                  dest="embeddings_file",
                  help="read embeddings from FILE",
                  metavar="FILE")
parser.add_option('-r',
                  '--embeddings-file2',
                  dest="embeddings_file2",
                  help="read embeddings from FILE",
                  metavar="FILE")
parser.add_option('-w',
                  '--wordpairs-file',
                  dest="wordpairs_file",
                  help="read word pairs from FILE",
                  metavar="FILE")
parser.add_option('-a',
                  '--alpha',
                  dest="alpha",
                  default=0.5,
                  help="Alpha to be multiplied with average threshold [default: %default]")

parser.add_option('-f',
                  '--scorefunction',
                  dest="scorefunction",
                  default='scorefunc6',
                  help="takes [dot|cosine|scorefunc1|scorefunc2|scorefunc3|scorefunc4|scorefunc5|scorefunc6, default: %default]")


(options, args) = parser.parse_args()

if not isfile(options.wordpairs_file):
    log.error("Cannot open word pairs file {0}".format(options.wordpairs_file))
    sys.exit(1)

if not isfile(options.embeddings_file):
    log.error("Cannot open word embeddings file {0}".format(options.embeddings_file))
    sys.exit(1)

if not isfile(options.embeddings_file2):
    log.error("Cannot open word embeddings file {0}".format(options.embeddings_file2))
    sys.exit(1)


# In[ ]:





# In[ ]:





# In[15]:


embedding_file_1 = options.embeddings_file
embedding_file_2 = options.embeddings_file2
dataset_file = options.wordpairs_file
alpha = float(options.alpha)

scorefunction = options.scorefunction
#scorefunction = "cosine"


# In[3]:


data = pd.read_csv(dataset_file, header=None, sep='\t', names =['word1', 'word2', 'class_binary', 'class_text'])


# In[4]:


data = data.iloc[:,:-1]
actual_label = np.array(data['class_binary'])


# In[5]:


data['word1'] = data['word1'].str.lower()
data['word2'] = data['word2'].str.lower()


# In[6]:


set_word_1 = set(data['word1'])


# In[7]:


set_word_2 = set(data['word2'])


# In[8]:


word1_index =dict([(w, i) for i, w in enumerate(set_word_1)])


# In[9]:


word2_index =dict([(w, i) for i, w in enumerate(set_word_2)])


# In[27]:


embedding_matrix_1 = {}
with open(embedding_file_1) as f:
    print('Reading Embedding file:', options.embeddings_file)
    for i, line in tqdm(enumerate(f)):

        s = line.split()
        word = s[0]
        word = word.lower()
        if word in set_word_1:
            
            try:
                embedding_matrix_1[word] = np.asarray(s[1:], dtype=np.float64)
            except ValueError:
                #print('Word not found in embedding 1:', s[0])
                continue




# In[28]:


embedding_matrix_2 = {}
with open(embedding_file_2) as f:
    print('Reading Embedding file:', options.embeddings_file2)
    for i, line in tqdm(enumerate(f)):

        s = line.split()
        word = s[0]
        word = word.lower()
        if word in set_word_2:
            
            try:
                embedding_matrix_2[word] = np.asarray(s[1:], dtype=np.float64)
            except ValueError:
                #print('Word not found in embedding 2:', s[0])
                continue


# In[ ]:





# In[29]:


def fn_cosine_sim(a, b, scorefunction):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def get_score(w1_vec1, w2_vec2, scorefunction):


    #if normalisation:
    #    if np.linalg.norm(w1_vec1)>0.0:
    #        w1_vec1 /= np.linalg.norm(w1_vec1)
    #    if np.linalg.norm(w2_vec2)>0.0:
    #        w2_vec2 /= np.linalg.norm(w2_vec2)

    if np.linalg.norm(w1_vec1)==0.0 or np.linalg.norm(w2_vec2)==0.0:
        cossim = 0.0
    else:
        cossim = 1.0-spatial.distance.cosine(w1_vec1, w2_vec2)

    
        
        
    if scorefunction == "dot":
        scor = w1_vec1.dot(w2_vec2)
    if scorefunction == "cosine":
        scor = cossim
    elif scorefunction == "scorefunc1":
        scor =  cossim-(np.linalg.norm(w1_vec1)-np.linalg.norm(w2_vec2))
    elif scorefunction == "scorefunc2":
        s1 = np.linalg.norm(w1_vec1)-np.linalg.norm(w2_vec2)
        s2 = np.linalg.norm(w1_vec1)+np.linalg.norm(w2_vec2)
        scor = cossim-s1/s2
    elif scorefunction == "scorefunc3":
        s1 = np.linalg.norm(w1_vec1)-np.linalg.norm(w2_vec2)
        s2 = max(np.linalg.norm(w1_vec1), np.linalg.norm(w2_vec2))
        scor = cossim-s1/s2
    elif scorefunction == "scorefunc4":
        scor = cossim * (np.linalg.norm(w2_vec2)/np.linalg.norm(w1_vec1))
    elif scorefunction == "scorefunc5":
        #scor = -(1+alpha * (np.linalg.norm(w1_vec1) - np.linalg.norm(w2_vec2)) * (np.arccosh(1+(2 * (np.linalg.norm(w1_vec1 - w2_vec2)**2 ) / ((1-(np.linalg.norm(w1_vec1)**2)) * (1-(np.linalg.norm(w2_vec2)**2)) ) ))))
        term1 = 1 + (alpha * (np.linalg.norm(w1_vec1) - np.linalg.norm(w2_vec2)))
        term2 = np.linalg.norm(w1_vec1 - w2_vec2)**2
        term3 = (1- (np.linalg.norm(w1_vec1)**2)) * (1-(np.linalg.norm(w2_vec2)**2))
        divterm = np.arccosh (1 + (2 * (term2 / term3)))
        scor = -(term1 * divterm)
    elif scorefunction == "scorefunc6":
        #scor = -(1+alpha * (np.linalg.norm(w1_vec1) - np.linalg.norm(w2_vec2)) * (np.arccosh(1+(2 * (np.linalg.norm(w1_vec1 - w2_vec2)**2 ) / ((1-(np.linalg.norm(w1_vec1)**2)) * (1-(np.linalg.norm(w2_vec2)**2)) ) ))))
        term1 = 1 + (alpha * (np.linalg.norm(w1_vec1) - np.linalg.norm(w2_vec2)))
        term2 = np.linalg.norm(w1_vec1 - w2_vec2)**2
        term3 = (1- (np.linalg.norm(w1_vec1)**2)) * (1-(np.linalg.norm(w2_vec2)**2))
        divterm = np.arccosh (1 + (2 * (term2 / term3)))
        scor = -(divterm)
    return np.nan_to_num(scor)


# In[63]:


def calculate_threshold(data):

    avg_threshold = []
    print('Calculating threshold.')
    for i in tqdm(range(1000)):
        sample = data.sample(n=None, frac=0.02, replace=False, weights=None, random_state=None, axis=None)
        sample_score = []
        for index, row in sample.iterrows():

            try:
                vec1 = embedding_matrix_1[row['word1']]
            except:
                vec1 = np.zeros(next(iter(embedding_matrix_1.values())).shape[0])

            try:
                vec2 = embedding_matrix_1[row['word2']]
            except:
                vec2 = np.zeros(next(iter(embedding_matrix_2.values())).shape[0])

            sample_score.append(get_score(np.array(vec1, dtype=np.float64), np.array(vec2, dtype=np.float64), scorefunction))
        avg_threshold.append(np.mean(sample_score))
    avg_threshold = (np.mean(avg_threshold))*alpha
    return avg_threshold


# In[64]:


avg_threshold = calculate_threshold(data)


# In[66]:


prediction = []
print('Calculating accuracy')
for index, row in tqdm(data.iterrows()):
    
    try:
        vec1 = embedding_matrix_1[row['word1']]
    except:
        vec1 = np.zeros(next(iter(embedding_matrix_1.values())).shape[0])
        
    try:
        vec2 = embedding_matrix_1[row['word2']]
    except:
        vec2 = np.zeros(next(iter(embedding_matrix_2.values())).shape[0])
        
    score = get_score(np.array(vec1, dtype=np.float64), np.array(vec2, dtype=np.float64), scorefunction)
    if score >= avg_threshold:
        prediction.append(True)
    else:
        prediction.append(False)
    
        
          
    


# In[67]:


print("%s: %.2f%%" % ('Accuracy', (float(sum(prediction == actual_label)) / float(len(prediction))) * 100))

print('settings used:')
print ("\t".join(map(str,('alpha', 'scorefunction'))))

print ("\t".join(map(str,(alpha,
                  scorefunction
                  ))))
