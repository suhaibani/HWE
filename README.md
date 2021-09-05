# HWE: Hierarchical Word Embeddings
* HWE is a joint model for learning hierarchical word embeddings from both text corpora and a taxonomy (knowledge base). 
* * "Fine-Tuning Word Embeddings for Hierarchical Representation of Data Using a Corpus and a Knowledge Base for Various Machine Learning Applications"


# Contents
* ./src/reps.cc is the source code for training the model
* ./src/shi.py is the source code for the Supervised Hypernym Identiﬁcation task.
* ./src/gle.py is the source code for the Graded Lexical Entailment task.
* ./src/uhdd.py is the source code for the Unsupervised Hypernym Directionality and Detection task.
* ./src/hpp.py is the source code for the Hierarchical Path Prediction task.
* ./src/wd.py is the source code for the Word Decomposition task.
* ./work includes all the lexicon files and datasets used for training and evaluating the model.
# Requirements
* The model ./src/reps.cc is written in C++, therefore a C++0x compiler is required
* [C++ Eigen Library](http://eigen.tuxfamily.org/index.php?title=Main_Page) need to be installed
  * Alternatively, instead of installing Eigen, you can simply create a directory named 'eigen' in the same level as ./src and copy the source code of Eigen into it
* [Numpy](http://www.numpy.org/), [sklearn](https://scikit-learn.org/), [scipy](https://www.scipy.org/) and [pandas](https://pandas.pydata.org/) libraries for the evaluation tasks.
