import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


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

documentA = 'the man went out for a walk'
documentB = 'the children sat around the fire'
documentC = "the woman teach the lesson"
documentD = "the idiot speak shit"

bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ')

uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))

numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
    numOfWordsA[word] += 1

numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
    numOfWordsB[word] += 1

print("Num Palavras A:",numOfWordsA)

print("Num Palavras B:",numOfWordsB)

tfA = computeTF(numOfWordsA, bagOfWordsA)
print("tfA", tfA)
tfB = computeTF(numOfWordsB, bagOfWordsB)
print("tfB", tfB)

idfs = computeIDF([numOfWordsA, numOfWordsB])

print("IDF:", idfs)

tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)

df = pd.DataFrame([tfidfA, tfidfB])

print("df")
print(df)


vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([documentA, documentB, documentC, documentD])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
dfsk = pd.DataFrame(denselist, columns=feature_names)

for doc in denselist:
    for c in doc:
        print(c, ",")
    print("\n")
print("df sklearn:")
print(dfsk)