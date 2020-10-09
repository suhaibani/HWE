# coding: utf-8

# In[1]:

from optparse import OptionParser
import logging as log
import sys
from os.path import isfile
import os


import numpy as np
import pandas as pd
from scipy import spatial
import warnings
warnings.filterwarnings("ignore")
# In[2]:

results_directory = 'results'

if not os.path.isdir(results_directory):
 # if not os.path.exists(directory):
  os.makedirs(results_directory)

# python eval_num_pred.py -q word_embed1 -e word_embed2 -d dataset_num.tsv -t word_choice -a sub_add -w 1 -f cosine -n False -c True


parser = OptionParser()


parser.add_option('-q',
                  '--embeddings-file1',
                  dest="embeddings_file1",
                  help="read embeddings 1 from FILE",
                  metavar="FILE")

parser.add_option('-e',
                  '--embeddings-file2',
                  dest="embeddings_file2",
                  help="read embeddings 2 from FILE",
                  metavar="FILE")


parser.add_option('-t',
                  '--type-to-use',
                  dest="type_to_use",
                  default='word_choice',
                  help="Type to use  [word_choice|arithmetic_choice|addition|subtraction, default: %default]")


parser.add_option('-a',
                  '--arithmetic-choice-to-use',
                  dest="arithmetic_choice_to_use",
                  default='sub_add',
                  help="Takes  [sub_add|add_sub to bue used for (q_vector1 - q_vector2 + q_vector3) and (q_vector1 + q_vector2 - q_vector3) respectively. It only works if 'arithmetic_choice' is selected above, default: %default]")


parser.add_option('-w',
                  '--word-number-to-use',
                  dest="word_number_to_use",
                  default='0',
                  help="If above set to 'word_choice', this will be used. Takes [0|1|2|... as word number to be used instead of all, default: %default]")


parser.add_option('-d',
                  '--dataset-path',
                  dest="dataset_path",
                  help="read word pairs from FILE",
                  metavar="FILE")

parser.add_option('-f',
                  '--scorefunction',
                  dest="scorefunction",
                  default='cosine',
                  help="takes [dot|cosine|scorefunc1|scorefunc2|scorefunc3|scorefunc4|scorefunc5|scorefunc6, default: %default]")

parser.add_option('-c',
                  '--use-cand',
                  dest="use_cand",
                  default='False',
                  help="If cand used, nothing else will be functional. Only -f is used to calculate cand. Takes [False|True: %default]")

parser.add_option('-n',
                  '--use-normalisation',
                  dest="normalisation",
                  default='False',
                  help="takes [False|True: %default]")

(options, args) = parser.parse_args()


# command line argument validation
if options.dataset_path == None or options.embeddings_file1 == None:
    print (parser.print_help())
    sys.exit(1)

if not isfile(options.embeddings_file1):
    log.error("Cannot open word embeddings file {0}".format(options.embeddings_file1))
    sys.exit(1)

if not isfile(options.embeddings_file2):
    log.error("Cannot open word embeddings file {0}".format(options.embeddings_file2))
    sys.exit(1)

############### Custom Options #######################

#concept
#question_embeddings = 'embeds/glove_300'
question_embeddings = options.embeddings_file1
#main
#answer_embeddigs = 'embeds/glove_300'
answer_embeddigs = options.embeddings_file2

embedding_dim = 300
####################### OPTIONS TO CUSTOMIZE OUTPUTS #############################

#type_to_use = 'word_choice' # takes 'addition', 'subtraction', 'word_choice', 'arithmetic_choice'
type_to_use = options.type_to_use
#arithmetic_choice_to_use = 'add_sub' # Takes 'sub_add', 'add_sub' to bue used for (q_vector1 - q_vector2 + q_vector3) and (q_vector1 + q_vector2 - q_vector3) respectively. It only works if 'arithmetic_choice' is selected above.
arithmetic_choice_to_use = options.arithmetic_choice_to_use

#word_number_to_use = 0 # If above set to 'word_choice', this will be used. Takes 0, 1, 2 as word number to be used instead of all three.
word_number_to_use = int(options.word_number_to_use)


#eval_dataset_path = 'dataset_num.tsv'

eval_dataset_path = options.dataset_path

eval_dataset = pd.read_csv(eval_dataset_path, sep='\t', header=None)

#scorefunction = "scorefunc6" # takes "dot", "cosine"  "scorefunc1", "scorefunc2", "scorefunc3", "scorefunc4", "scorefunc5", "scorefunc6"

scorefunction = options.scorefunction
if options.use_cand == 'False':
    use_cand = False
else:
    use_cand = True

alpha = 1000

if options.normalisation == 'False':
    normalisation = False
else:
    normalisation = True

#normalisation = options.normalisation # Takes True or False

######################################################################
# In[3]:


question_words = eval_dataset.iloc[:, 0:-1].apply(lambda x: ' '.join(x), axis=1)
question_words = [item.lower() for item in question_words]

# In[4]:


answer_words = eval_dataset.iloc[:, -1]

# answer_words = [item.lower() for item in answer_words]
answer_words = [it.lower() for it in answer_words]


# In[5]:


############### DEFINING FUNCTIONS ###############

def fn_make_word_index_dictionary(question_words):
    q_word_list = set()
    for line in question_words:
        for word in (line.split()):
            if word not in q_word_list:
                q_word_list.add(word)
            # print(q_word_list)
        # print(line)
    q_word_index = dict([(w, i) for i, w in enumerate(q_word_list)])
    return q_word_index


# In[6]:


def my_embedding_initialize(word_indices, EMBEDDING_DIM, divident=1.0):
    n = len(word_indices)
    m = EMBEDDING_DIM
    emb = np.zeros((n, m), dtype=np.float32)

    # emb[:,:] = np.random.normal(size=(n,m)) / divident

    # emb[0, :] = np.zeros((1,m), dtype="float32")
    return (emb)


import copy
def my_embeddings(word_indices, EMBED_FILE, emb):
    word_indices1 = copy.deepcopy(word_indices)
    with open(EMBED_FILE) as f:
        for i, line in enumerate(f):

            s = line.split()
            word = s[0]
            word = word.lower()
            # print(word)
            if word in word_indices:
                # print("True")
                if word in word_indices1:
                    word_indices1.pop(word)
                try:
                    # print(len(s[1:]))
                    emb[word_indices[word], :] = np.asarray(s[1:])
                except ValueError:
                    #print(s[0])
                    continue
        #for i in word_indices1.keys():
        #    print(i)
    return (emb)


# In[7]:


def fn_cosine_sim(a, b, scorefunction):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def get_score(w1_vec1, w2_vec2, scorefunction):


    if normalisation:
        if np.linalg.norm(w1_vec1)>0.0:
            w1_vec1 /= np.linalg.norm(w1_vec1)
        if np.linalg.norm(w2_vec2)>0.0:
            w2_vec2 /= np.linalg.norm(w2_vec2)

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
    return scor



#########################################################


# In[8]:


q_word_index = fn_make_word_index_dictionary(question_words)

# In[9]:


a_word_index = fn_make_word_index_dictionary(answer_words)
a_index_word = {v: k for k, v in a_word_index.items()}

# In[10]:

print('dimension: %s'%embedding_dim)
print('Found %s unique question words' % len(q_word_index))

print('Found %s unique answer words' % len(a_word_index))

# In[11]:


print('Preparing Question Words embedding matrix')

q_embedding_matrix = my_embedding_initialize(q_word_index, embedding_dim)

q_embedding_matrix = my_embeddings(q_word_index, question_embeddings, q_embedding_matrix)

# np.save(open(embedding_file, 'wb'), embedding_matrix)


print('Null word embeddings in questions: %d' % np.sum(np.sum(q_embedding_matrix, axis=1) == 0))
print('Question words embedding matrix shape: ', q_embedding_matrix.shape)

print('Preparing Answers Words embedding matrix')

a_embedding_matrix = my_embedding_initialize(a_word_index, embedding_dim)

a_embedding_matrix = my_embeddings(a_word_index, answer_embeddigs, a_embedding_matrix)

# np.save(open(embedding_file, 'wb'), embedding_matrix)


print('Null word embeddings in answers: %d' % np.sum(np.sum(a_embedding_matrix, axis=1) == 0))
print('Answer words embedding matrix shape: ', a_embedding_matrix.shape)

# In[12]:


print('Calculating Answer Words....')

answer = []
wrong_answer = []
if not use_cand:
    for line in question_words:
        test_words = line.split()
        # print(test_words)

        # test_words = 'accessory    criminal    principal'
        # test_words = test_words.split()
        q_vector = np.zeros((1, embedding_dim), dtype=np.float32)

        if type_to_use == 'addition':
            for word in (test_words):
                word = word.lower()
                # print(word)
                q_idx = q_word_index[word]
                # print (q_idx)
                q_vector = q_vector + q_embedding_matrix[q_idx]

        elif type_to_use == 'subtraction':
            for word in (test_words):
                word = word.lower()
                # print(word)
                q_idx = q_word_index[word]
                # print (q_idx)
                q_vector = q_vector - q_embedding_matrix[q_idx]

        elif type_to_use == 'word_choice':
            if len(test_words) < int(word_number_to_use) + 1:
                print('Length of words is less than word number mentioned...')
                exit()
            else:
                word = test_words[word_number_to_use]
                word = word.lower()
                # print(word)
                q_idx = q_word_index[word]
                # print (q_idx)
                q_vector = q_embedding_matrix[q_idx]

        elif type_to_use == 'arithmetic_choice':
            if arithmetic_choice_to_use == 'sub_add':
                for j, word in enumerate(test_words):
                    word = word.lower()
                    # print(word)
                    q_idx = q_word_index[word]
                    # print (q_idx)
                    if j == 0 or j==1:
                        q_vector = q_vector - q_embedding_matrix[q_idx]
                    else:
                        q_vector = q_vector + q_embedding_matrix[q_idx]

            elif arithmetic_choice_to_use == 'add_sub':
                for j, word in enumerate(test_words):
                    word = word.lower()
                    # print(word)
                    q_idx = q_word_index[word]
                    # print (q_idx)
                    if j == 0 or j==1:
                        q_vector = q_vector + q_embedding_matrix[q_idx]
                    else:
                        q_vector = q_vector - q_embedding_matrix[q_idx]
            else:
                print('arithmetic_choice_to_use has only two options. 1) add_sub 2) sub_add... Please check if you have mentioned that right.')

        else:
            print('type_to_use has only three otpions. 1) subtraction 2) word_choice 3) arithmetic_choice... Please check if you have mentioned that right.')


    ##########################################################

        scor = []
        for idx, a_vector in enumerate(a_embedding_matrix):
            # cos_sim.append(1 - spatial.distance.cosine(q_vector, a_vector))
            scor.append(get_score(q_vector, a_vector, scorefunction))

        scor = np.array(scor)
        scor = np.nan_to_num(scor)
        # print(cos_sim)
        #sorted(cosine_sim)
        tmp = u""
        #if scorefunction != "scorefunc5":
        #    x = np.argsort(1-scor)
        #else:
        #    x = np.argsort(scor)
        x = np.argsort(1-scor)
        for idx, idx_val in enumerate(x):
            if idx==0:
                answer.append(a_index_word[idx_val])
            else:
                tmp += str(a_index_word[idx_val]) + ','
        wrong_answer.append(tmp)


elif use_cand:
    print('Define cand function here...')
    
    

    for line in question_words:
        cand =[]
        test_words = line.split()
      
        
        for idx, a_vector in enumerate(a_embedding_matrix):
            scor = []
            for word in (test_words):
                word = word.lower()
                # print(word)
                q_idx = q_word_index[word]
                # print (q_idx)
                q_vector = q_embedding_matrix[q_idx]


                    
                scor.append(get_score(q_vector, a_vector, scorefunction))
            cand.append(np.sum(scor))
        cand = np.array(cand)
        #print(np.argmax(cand))
        #print(np.argsort(-cand)[0])
        tmp = u""
        #if scorefunction != "scorefunc5":
        #    x = np.argsort(1-scor)
        #else:
        #    x = np.argsort(scor)
        x = np.argsort(-cand)
        for idx, idx_val in enumerate(x):
            if idx==0:
                answer.append(a_index_word[idx_val])
            else:
                tmp += str(a_index_word[idx_val]) + ','
        wrong_answer.append(tmp)



    #print(len(cand))
#the new method is like this:
#we have the words (b), (c),(d) and we want to use them predict (a).
#so we will check the possible (a's) as follows:

#cand(a) = f(a,b) + f(a,c) + f(a,d).
#and the highest a which return the maximum (cand (a)), will be the prediction.

#what is (f) above? we can set it to be one of the 6 score functions we have.

    #max_sim_idex = np.argmax(cos_sim)
    # print(max_sim_idex)

    #answer.append(a_index_word[max_sim_idex])

# # Compare Predicted Answer Words with Actual Answer Words

# In[13]:

print('Writing submission file...')

submission = pd.DataFrame()
submission['Actual_Word'] = answer_words
submission['Predicted_Word'] = answer
submission.to_csv('results/poincare_noCONTEXT_comp_d1.csv', sep='\t', index=False, encoding='utf-8')

wrong_submission = pd.DataFrame()
wrong_submission['Actual_Word'] = answer_words
wrong_submission['Predicted_Word'] = answer
wrong_submission['Next_Predictions'] = wrong_answer
wrong_submission.to_csv('results/poincare_noCONTEXT_comp_d1_top10.csv', sep='\t', index=False, encoding='utf-8')
# In[14]:


answer_words = np.array(answer_words)
answer = np.array(answer)
# print("%s: %.2f%%" % ('Accuracy', np.sum(answer_words == answer)/len(answer_words)*100))
correct = 0
for i, w in enumerate(answer_words):
    if w == answer[i]:
        correct = correct + 1
# print("%s: %.2f%%" % ('Accuracy', correct/len(answer_words)*100))
# print ('Accuracy: ', correct/len(answer_words)*100 )
# In[ ]:
print("Model: %s - %s" % (options.embeddings_file1 , options.embeddings_file2))
print("Score function: %s " % options.scorefunction)
print("Normalisation: %s " % options.normalisation)
print("%s: %.2f%%" % ('Accuracy', (float(correct) / float(len(answer_words))) * 100))
