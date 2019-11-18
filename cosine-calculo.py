#Faz o cálculo da similaridade do cosseno com os vetores lidos do arquivo enwiki-vector-20.txt.

## feature extraction
import pandas as pd
import numpy as np
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
            #print(linha)
            vector = list(map(int, linha.rsplit(" ")))
            #print(vector)

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
        return np.linalg.norm(x)

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

    def toUnitMatrix(self, x):
        v = []
        for l in range(len(x)):
            n = cosine.norm(x[l])
            v.append( [i/n for i in x[l]])
        return v

    def prefix(self, x, p):
        return x[0:p]

    def suffix(self, x, p):
        return x[-p:]

    # calcula o cosseno entre um vetor especificado e os outros vetores da matriz de features (inclusive com ele mesmo).
    # x: índice do vetor que será comparado com os outros vetores.
    # matriz: matriz de features.
    def cosine_between_vectors(self, i, matriz):
        for linha in range(matriz.__len__()):
            print('Cosseno de', i, 'com', linha , ":", cosine.cos(matriz[i], matriz[linha]))

    # calcula o produto escalar entre um vetor especificado e os outros vetores da matriz de features (inclusive com ele mesmo).
    # Como os vetores já estão normalizados então o cálculo será o cosseno.
    # x: índice do vetor que será comparado com os outros vetores.
    # matriz: matriz de features.
    def dot_between_vectors(self, i, matriz):
        for linha in range(matriz.__len__()):
            print('Cosseno de', i, 'com', linha , ":", cosine.dot(matriz[i], matriz[linha]))

read = text()
cosine = cosine()

feat = np.array(read.read_text("enwiki-vector-20.txt"))

x = np.array([24,2,20,2])
y = np.array([0.01,0.01,0.01,0.01])
z = np.array([24,1,10,1])

a = np.array([1,2,3,4,5,6,7,8,9,10,11])

print(cosine.prefix(a, round(a.__len__()/2)))
print(cosine.suffix(a, round(a.__len__()/2) - 1))

print(cosine.prefix(x,2))
print(cosine.suffix(x,2))
print('---------------')

print('Tamanho do vetor feat: %+d' % feat.__len__())
for i in range(feat.__len__()):
    print('Tamanho do vetor feat[%d]: %d' % (i, feat[i].__len__()))
    print('Prefixo do vetor feat[%d]: %s' % (i, str(cosine.prefix(feat[i], round(feat[i].__len__()/2)))))
    print('Tamanho do prefixo do vetor feat[%d]: %d' % (i, cosine.prefix(feat[i], round(feat[i].__len__()/2)).__len__()))
    print('Sufixo  do vetor feat[%d]: %s' % (i, str(cosine.suffix(feat[i], round(feat[i].__len__()/2) - 1))))
    print('Tamanho do sufixo do vetor feat[%d]: %d' % (i, cosine.suffix(feat[i], round(feat[i].__len__()/2)).__len__() -1 ))
print('---------------')

featU = cosine.toUnitMatrix(feat)
print(feat)
print(cosine.norm(feat))
print(featU)
print(cosine.norm(featU))
print('---------------')

#print('norma featU:', np.sqrt(np.dot(featU, featU)))

#Normaliza os vetores.
xu = cosine.toUnit(x)
yu = cosine.toUnit(y)
zu = cosine.toUnit(z)
#
print(xu, cosine.norm(xu))
print(yu, cosine.norm(yu))
print(zu, cosine.norm(zu))
print('---------------')
#Calcula o cosseno entre os vetores x e y (não normalizados).
print(cosine.cos(x,y))
#Calcula o produto escalar entre os vetores xu e yu (já normalizados)
print(cosine.dot(xu,yu))
print('---------------')
#Calcula o cosseno entre os vetores x e z (não normalizados).
print(cosine.cos(x,z))
#Calcula o produto escalar entre os vetores xu e zu (já normalizados)
print(cosine.dot(xu,zu))
print('---------------')
print('Cosseno com os vetores sem normalização:')
cosine.cosine_between_vectors(0, feat)

print('Cosseno com os vetores com normalização:')
cosine.dot_between_vectors(0, featU)
print('---------------')


for i in range(feat.__len__()):
    #if (i == 0 or i == 1 or i == 90):
        cont = 0
        for j in range(feat[i].__len__()):
            if (feat[i][j] != 0):
                #print('feature (', i, ",", j,  "):", feat[i][j])
                cont += 1
        print('qtde features > 0 do vetor', i, ': ', cont)

