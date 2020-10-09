import pandas as pd
import numpy as np
import argparse
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


def load_data(dataset, embeddings1, embeddings2):
    """ Loads the dataset and the two embedding matrices.

    Args:
        dataset (str): Path to dataset file.
        embeddings1 (str): Path to embedding file.
        embeddings2 (str): Path to embedding file.

    Returns:
        pandas dataframes
    """
    data = pd.read_csv(dataset, sep='\t', header=None)
    data.columns = ['word1', 'word2', "relationship"]
    emb1 = pd.read_csv(embeddings1, delim_whitespace=True, header=None, index_col=0)
    emb2 = pd.read_csv(embeddings2, delim_whitespace=True, header=None, index_col=0)

    return data, emb1, emb2


def create_train_test_sets(data, split, embeddings1, embeddings2, method='concatenation'):
    tmp1 = embeddings1.loc[data['word1']].reset_index(drop=True)
    tmp2 = embeddings2.loc[data['word2']].reset_index(drop=True)

    if method == 'concatenation':
        emb_data = pd.concat([tmp1, tmp2], axis=1, ignore_index=True, sort=False)
    elif method == 'subtraction':
        emb_data = tmp1 - tmp2
    else:
        print('Unknow method: {}'.format(method))
        return False

    emb_data = pd.concat([data, emb_data], sort=False, axis=1)
    emb_data.to_csv("delete.csv")
    na_words1 = emb_data.loc[pd.isna(emb_data.loc[:, int(emb_data.shape[1]/2) -20]), 'word1']
    na_words2 = emb_data.loc[pd.isna(emb_data.loc[:, int(emb_data.shape[1]/2) +20]), 'word2']
    na_words1 = [str(w) for w in na_words1]
    na_words2 = [str(w) for w in na_words2]


    with open('missing_words.txt', 'w') as file:
        if len(na_words1) > 0:
            file.write("Words missing in first embedding matrix: \n {} \n".format(",".join(na_words1)))
        if len(na_words2) > 0:
            file.write("Words missing in second embedding matrix: \n {}".format(",".join(na_words2)))

    emb_data.dropna(inplace=True)

    X_train, X_test = train_test_split(emb_data, test_size=1 - split)

    y_train = X_train['relationship'].astype(int)
    y_test = X_test['relationship'].astype(int)

    X_train.drop(['word1', 'word2', 'relationship'], inplace=True, axis=1)
    X_test.drop(['word1', 'word2', 'relationship'], inplace=True, axis=1)

    return X_train, X_test, y_train, y_test


def create_knn_classifier(X_train, y_train, k=3):
    """ Create a KNN classifier based on X_train and y_train with k as the number of nearest neighbors."""

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    return knn


def predict_knn(classifier, X_test):
    """Predict labels and class probabilities for new data."""

    pred_classes = classifier.predict(X_test)
    pred_probs = classifier.predict_proba(X_test)

    return pred_classes, pred_probs


def evaluate_predictions(y_test, pred_classes, pred_probs):
    """ Create different performance measures for the KNN classifier. The metrics are accuracy, auc, precision, recall
    and f1 score.

    Args:
        y_test (series): True labels.
        pred_classes (array): Predicted labels.
        pred_probs (array): Predicted probabilities for label == True.

    Returns:
        Prints output to console.
    """

    acc = accuracy_score(y_test, pred_classes)
    auc = roc_auc_score(y_test, pred_probs[:, 1])
    precision = precision_score(y_test, pred_classes)
    recall = recall_score(y_test, pred_classes)
    f1 = f1_score(y_test, pred_classes)

    print('Accuracy: {}'.format(round(acc, 3)))
    print('AUC: {}'.format(round(auc, 3)))
    print('Precision: {}'.format(round(precision, 3)))
    print('Recall: {}'.format(round(recall, 3)))
    print('F1 Score: {}'.format(round(f1, 3)))


parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('-w', type=str, help='Embedding matrix for the first word.')
parser.add_argument('-e', type=str, help='Embedding matrix for the second word.')
parser.add_argument('-d', type=str, help='Dataset file.')
parser.add_argument('-m', type=str, choices=['concatenation', 'subtraction'],
                    help='Method: concatenation or subtraction')
parser.add_argument('-k', type=int, help='Number of nearest neighbors.')
parser.add_argument('-t', type=float, help='Size of the training set.')

args = parser.parse_args(sys.argv[1:])

data, emb1, emb2 = load_data(args.d, args.w, args.e)
X_train, X_test, y_train, y_test = create_train_test_sets(data=data, split=args.t, embeddings1=emb1, embeddings2=emb2,
                                                          method=args.m)
knn = create_knn_classifier(X_train, y_train, k=args.k)
pred_classes, pred_probs = predict_knn(knn, X_test)
evaluate_predictions(y_test, pred_classes, pred_probs)
