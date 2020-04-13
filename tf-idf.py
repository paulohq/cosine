# teste de geração de tfidf do corpus

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import math


def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


def computeIDF(documents):
    import math
    N = len(documents)

    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

print(stopwords.words('english'))

documentA = "the man keeps walking" #'the man went out for a walk'
documentB = "the children study" #'the children sat around the fire'
documentC = "the woman teach the lesson"
documentD = "the woman teach the children" #"the idiot speak shit"

bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ')
bagOfWordsC = documentC.split(' ')
bagOfWordsD = documentD.split(' ')

uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB)).union(set(bagOfWordsC)).union(set(bagOfWordsD))

numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
    numOfWordsA[word] += 1

numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
    numOfWordsB[word] += 1

numOfWordsC = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsC:
    numOfWordsC[word] += 1

numOfWordsD = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsD:
    numOfWordsD[word] += 1

print("Num Palavras A:",numOfWordsA)

print("Num Palavras B:",numOfWordsB)

print("Num Palavras C:",numOfWordsC)
print("Num Palavras D:",numOfWordsD)

tfA = computeTF(numOfWordsA, bagOfWordsA)
print("tfA", tfA)
tfB = computeTF(numOfWordsB, bagOfWordsB)
print("tfB", tfB)
tfC = computeTF(numOfWordsC, bagOfWordsC)
print("tfC", tfC)
tfD = computeTF(numOfWordsD, bagOfWordsD)
print("tfD", tfD)

idfs = computeIDF([numOfWordsA, numOfWordsB, numOfWordsC, numOfWordsD])

print("IDF:", idfs)

tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
tfidfC = computeTFIDF(tfC, idfs)
tfidfD = computeTFIDF(tfD, idfs)

print("TFIDF A: ", tfidfA)
print("TFIDF B: ", tfidfB)
print("TFIDF C: ", tfidfC)
print("TFIDF D: ", tfidfD)


df = pd.DataFrame([tfidfA, tfidfB, tfidfC, tfidfD])

print("df")
print(df)


vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([documentA, documentB, documentC, documentD])
feature_names = vectorizer.get_feature_names()
print("feature names:", feature_names)
dense = vectors.todense()
denselist = dense.tolist()
dfsk = pd.DataFrame(denselist, columns=feature_names)


for doc in denselist:
    print(doc)
    n = 0.0
    y = 0
    for i in range(len(doc)):
        n = n + doc[i] * doc[i]

    n = math.sqrt(n)
    print("l2 norm = ", n)

    # for c in doc:
    #     print(c, ",")
    # print("\n")
print("df sklearn:")
print(dfsk)