#Faz o cálculo da similaridade do cosseno com os vetores lidos do arquivo enwiki-vector-20.txt.

## feature extraction
import pandas as pd
import numpy
import nltk
import gensim
import scipy.sparse as sp
from numpy.linalg import norm


class text(object):
    def __init__(self):
        self.data_path = ""

    # Reads a given TXT and stores the data in a list
    def read_text(self, data_path):
        file_reader = open(data_path, 'r')
        lista = []
        for linha in file_reader:
            vector = []
            print(linha)
            vector = list(map(int, linha.rsplit(" ")))
            print(vector)

            lista.append(vector)
        file_reader.close()

        return lista

        for row in file_reader:
            sent_list.append(row[0])
        return sent_list

    # This will create and write into a new TXT
    def write_text(self, text, out_path):
        filewriter = open(out_path, 'w')
        for linha in text:
            texto = ''
            for coluna in linha:
                texto = texto + str(coluna) + ' '

            texto = texto[0:texto.__len__() - 1]
            filewriter.write(texto + '\n')

        filewriter.close()

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

feat = read.read_text("enwiki-vector-20.txt")

x = [24,2,20,2]
y = [0.01,0.01,0.01,0.01]
z = [24,1,10,1]

print(cosine.prefix(x,2))
print(cosine.suffix(x,2))


featU = cosine.toUnit(feat)
print(featU)
print(cosine.norm(featU))

#print('norma featU:', numpy.sqrt(numpy.dot(featU, featU)))

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

print('Coseno com os vetores sem normalização:')
#print('Coseno entre mesmo vetor:', cosine.cos(feat[0], feat[0]))
for i in range(feat.__len__()):
    print('Coseno de 0 com ', i , ":", cosine.cos(feat[0], feat[i]))

print('Coseno com os vetores já normalizados:')
for i in range(featU.__len__()):
    print('Coseno de 0 com ', i , ":", cosine.dot(featU[0], featU[i]))