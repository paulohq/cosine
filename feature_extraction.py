from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import gensim
import scipy.sparse as sp
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer

class feature_extraction(object):
    def __init__(self):

        self.model = []

    #Extract bow features for the CORPUS.
    def bow_extractor(self, CORPUS, ngram_range=(1, 1)):
        vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
        features = vectorizer.fit_transform(CORPUS)
        return vectorizer, features

    #Display features and feature names for the corpus.
    def display_features(self, features, feature_names):
        df = pd.DataFrame(data=features, columns=feature_names)
        print(df)
        return df

    #TF-IDF  features to the bow matrix passed as parameter.
    def tfidf_transformer(self, bow_matrix):
        transformer = TfidfTransformer(norm='l2',
                                       smooth_idf=True,
                                       use_idf=True)
        tfidf_matrix = transformer.fit_transform(bow_matrix)
        return transformer, tfidf_matrix

    #Extract TF-IDF for the new document using built tfidf transformer.
    def tfidf_new_doc_features(self, new_doc, bow_vectorizer, tfidf_trans):
        new_doc_features = bow_vectorizer.transform(new_doc)
        nd_tfidf = tfidf_trans.transform(new_doc_features)
        nd_features = np.round(nd_tfidf.todense(), 2)
        feature_names = bow_vectorizer.get_feature_names()
        return nd_features, feature_names

    #Extract bow features for new document in the test.
    def bow_new_doc_features(self, bow_vectorizer, new_doc):
        new_doc_features = bow_vectorizer.transform(new_doc)
        new_doc_features = new_doc_features.todense()
        return new_doc_features

    #compute the tfidf-based feature vectors for documents
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
    #to perform averaging of word vectors for a corpus of documents
    def averaged_word_vectorizer(self, corpus, model, num_features):
        vocabulary = set(model.wv.index2word)
        features = [self.average_word_vectors(tokenized_sentence, model, vocabulary,num_features)
                                    for tokenized_sentence in corpus]
        return np.array(features)

    #Create the model word2vec.
    def create_model_word2vec(self, TOKENIZED_CORPUS, size, window, min_count, sample):
        model = gensim.models.Word2Vec(TOKENIZED_CORPUS, size=size, window=window, min_count=min_count, sample=sample)

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
    #Created to perform TF-IDF weighted averaging of word vectors for a corpus of documents.
    def tfidf_weighted_averaged_word_vectorizer(self, corpus, tfidf_vectors, tfidf_vocabulary, model, num_features):
        docs_tfidfs = [(doc, doc_tfidf)
                       for doc, doc_tfidf
                       in zip(corpus, tfidf_vectors)]
        features = [self.tfidf_wtd_avg_word_vectors(tokenized_sentence, tfidf, tfidf_vocabulary, model, num_features)
                            for tokenized_sentence, tfidf in docs_tfidfs]
        return np.array(features)

extract = feature_extraction()



