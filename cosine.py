#Importa parte do dataset enwiki (arquivo enwiki-20.txt),
# faz a normalização dos textos, gera um bag of words e cria um arquivo com as contagens para cada feature (enwiki-vector.txt).

import numpy
import csv

## feature extraction
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import gensim
import scipy.sparse as sp
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer

from pre_processing import *

class text(object):
    def __init__(self):
        #train = file_reader = csv.reader(open(data_path, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
        self.data_path = ""

    # Reads a given CSV and stores the data in a list
    def read_text(self, data_path, test_list=False):
        file_reader = csv.reader(open(data_path, "rt", errors="ignore", encoding="utf-8"))
        sent_list = []
        # print(file_reader.shape)
        # print(file_reader.columns.values)
        for row in file_reader:
            sent_list.append(row[0])
        return sent_list

    # This will create and write into a new TXT
    def write_text(self, text, out_path):
        #filewriter = csv.writer(open(out_path, "w+"))
        filewriter = open(out_path, 'w')
        for linha in text:
            texto = ''
            for coluna in linha:
                texto = texto + str(coluna) + ' '

            #print(texto)
            texto = texto[0:texto.__len__() - 1]
            #print(texto)
            filewriter.write(texto + '\n')

class feature_extraction(object):
    def __init__(self):

        self.model = []

    # Extract bow features for the CORPUS.
    def bow_extractor(self, CORPUS, ngram_range=(1, 1)):
        vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
        features = vectorizer.fit_transform(CORPUS)
        return vectorizer, features

    # Display features and feature names for the corpus.
    def display_features(self, features, feature_names):
        df = pd.DataFrame(data=features, columns=feature_names)
        print(df)
        return df

    # TF-IDF  features to the bow matrix passed as parameter.
    def tfidf_transformer(self, bow_matrix):
        transformer = TfidfTransformer(norm='l2',
                                       smooth_idf=True,
                                       use_idf=True)
        tfidf_matrix = transformer.fit_transform(bow_matrix)
        return transformer, tfidf_matrix

    # Extract TF-IDF for the new document using built tfidf transformer.
    def tfidf_new_doc_features(self, new_doc, bow_vectorizer, tfidf_trans):
        new_doc_features = bow_vectorizer.transform(new_doc)
        nd_tfidf = tfidf_trans.transform(new_doc_features)
        nd_features = np.round(nd_tfidf.todense(), 2)
        feature_names = bow_vectorizer.get_feature_names()
        return nd_features, feature_names

    # Extract bow features for new document in the test.
    def bow_new_doc_features(self, bow_vectorizer, new_doc):
        new_doc_features = bow_vectorizer.transform(new_doc)
        new_doc_features = new_doc_features.todense()
        return new_doc_features

    # compute the tfidf-based feature vectors for documents
    def tfidf_extractor(self, corpus, ngram_range=(1, 1)):
        vectorizer = TfidfVectorizer(min_df=1,
                                     norm='l2',
                                     smooth_idf=True,
                                     use_idf=True,
                                     ngram_range=ngram_range)
        features = vectorizer.fit_transform(corpus)
        return vectorizer, features

    # define function to average word vectors for a text document
    def average_word_vectors(self, words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        for word in words:
            if word in vocabulary:
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model[word])

        if nwords:
            feature_vector = np.divide(feature_vector, nwords)
        return feature_vector

    # generalize above function for a corpus of documents.
    # to perform averaging of word vectors for a corpus of documents
    def averaged_word_vectorizer(self, corpus, model, num_features):
        vocabulary = set(model.wv.index2word)
        features = [self.average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
        return np.array(features)

    # Create the model word2vec.
    def create_model_word2vec(self, TOKENIZED_CORPUS, size, window, min_count, sample):
        model = gensim.models.Word2Vec(TOKENIZED_CORPUS, size=size, window=window, min_count=min_count,
                                       sample=sample)

        return model

    # define function to compute tfidf weighted averaged word vector for a document
    def tfidf_wtd_avg_word_vectors(self, words, tfidf_vector, tfidf_vocabulary, model, num_features):
        word_tfidfs = [tfidf_vector[0, tfidf_vocabulary.get(word)]
                       if tfidf_vocabulary.get(word)
                       else 0 for word in words]

        word_tfidf_map = {word: tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}
        feature_vector = np.zeros((num_features,), dtype="float64")
        vocabulary = set(model.wv.index2word)
        wts = 0.
        for word in words:
            if word in vocabulary:
                word_vector = model[word]
                weighted_word_vector = word_tfidf_map[word] * word_vector
                wts = wts + word_tfidf_map[word]
                feature_vector = np.add(feature_vector, weighted_word_vector)
        if wts:
            feature_vector = np.divide(feature_vector, wts)
        return feature_vector

    # generalize above function for a corpus of documents.
    # Created to perform TF-IDF weighted averaging of word vectors for a corpus of documents.
    def tfidf_weighted_averaged_word_vectorizer(self, corpus, tfidf_vectors, tfidf_vocabulary, model, num_features):
        docs_tfidfs = [(doc, doc_tfidf)
                       for doc, doc_tfidf
                       in zip(corpus, tfidf_vectors)]
        features = [
            self.tfidf_wtd_avg_word_vectors(tokenized_sentence, tfidf, tfidf_vocabulary, model, num_features)
            for tokenized_sentence, tfidf in docs_tfidfs]
        return np.array(features)


class cosine(object):
    def __init__(self):
        self.corpus = []
        self.norm_corpus = []

    def norm(self, x):
        return numpy.linalg.norm(x)

    def dot(self, x, y):
        sum = 0
        for i in range(len(x)):
            sum = sum + x[i] * y[i]
        return sum

    def cos(self, x, y):
        return cosine.dot(x,y) / (cosine.norm(x) * cosine.norm(y))

    def toUnit(self, x):
        n = cosine.norm(x)
        return [i/n for i in x]

    def prefix(self, x, p):
        return x[0:p]

    def suffix(self, x, p):
        return x[-p:]

read = text()
cosine = cosine()
pre = pre_processing()

cosine.corpus = read.read_text("enwiki-20.txt")

cosine.norm_corpus = pre.normalize_corpus(cosine.corpus)

new_doc = ['loving this blue sky today abandoned']

extract = feature_extraction()
# bag of words features
bow_vectorizer, bow_train_features = extract.bow_extractor(cosine.norm_corpus)
features = bow_train_features.todense()
bow_test_features = bow_vectorizer.transform(new_doc)
new_doc_features = bow_test_features.todense()
feature_names = bow_vectorizer.get_feature_names()

print('features')
print (features)
print('bow_test_features')
print(bow_test_features)
print('new_doc_features')
print (new_doc_features)
print('feature names')
print (feature_names)
feat = []
for row in range(features.shape[0]):
    linha = []
    for column in range(features.shape[1]):
        linha.append(features[row,column])
        #print(feature_names[column], ' - ' , feat[row, column])
    feat.append(linha)
    #print(feat[row])

print (feat)
#read.write_text(feat, "enwiki-vector-tfidf-4.txt")

#TFIDF
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(cosine.norm_corpus)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
dfsk = pd.DataFrame(denselist, columns=feature_names)

featTFIDF = []
for row in range(dense.shape[0]):
    linha = []
    for column in range(dense.shape[1]):
        linha.append(dense[row,column])
        #print(feature_names[column], ' - ' , feat[row, column])
    featTFIDF.append(linha)
    #print(feat[row])

print (featTFIDF)

read.write_text(featTFIDF, "enwiki-vector-tfidf-20.txt")


new_doc_features = bow_vectorizer.transform(new_doc)
new_doc_features = new_doc_features.todense()
print('new doc features')
print (new_doc_features)

tfidf_vectorizer, tfidf_train_features = extract.tfidf_extractor(cosine.corpus)
tfidf_test_features = tfidf_vectorizer.transform(new_doc)
print(tfidf_test_features)


x = [24,2,20,2]
y = [0.01,0.01,0.01,0.01]
z = [24,1,10,1]

print(cosine.prefix(x,2))
print(cosine.suffix(x,2))

#for i in range(feat.__len__()):
#    a = a + feat[i] ** 2
print('norma', numpy.sqrt(numpy.dot(feat, feat)))

featU = cosine.toUnit(feat)
print(feat, cosine.norm(feat))

xu = cosine.toUnit(x)
yu = cosine.toUnit(y)
zu = cosine.toUnit(z)
print(xu, cosine.norm(xu))
print(yu, cosine.norm(yu))
print(zu, cosine.norm(zu))

print(cosine.cos(x,y))
print(cosine.dot(xu,yu))

print(cosine.cos(x,z))
print(cosine.dot(xu,zu))