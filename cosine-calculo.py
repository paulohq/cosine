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

    # Normaliza o vetor passado como parâmetro.
    def toUnit(self, x):
        n = cosine.norm(x)
        return [i/n for i in x]

    # Normaliza os vetores, da Matriz passada como parâmetro, fazendo a soma dos elementos de cada vetor para fazer
    # a normalização.
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
    # i: índice do vetor que será comparado com os outros vetores.
    # matriz: matriz de features.
    def dot_between_vectors(self, i, matriz):
        for linha in range(matriz.__len__()):
            print('Cosseno de', i, 'com', linha , ":", cosine.dot(matriz[i], matriz[linha]))

    # Verifica a intersecção entre o vetor v (consulta) e a matriz (dataset).
    # v: vetor consulta
    # matriz: dataset
    def intersect_between_vectors(self, v, matriz):
        vet_prefixo = []
        for i in range(matriz.__len__()):
            # if (i == 0 or i == 1 or i == 90):
            intersect = 0
            for j in range(matriz[i].__len__()):
                if ((matriz[i][j] != 0) and (v[j] != 0)):
                    # print('feature (', i, ",", j,  "):", feat[i][j])
                    intersect += 1
            print('intersecção entre o vetor v e linha ', i, ' da matriz: ', intersect)
            # Calula o tamanho do prefixo para o vetor i da matriz com a fórmula: (h - k + 1)
            # h: o tamanho do vetor
            # k: a intersecção entre o veotr v e o vetor i da matriz
            prefixo = matriz[i].__len__() - intersect + 1
            vet_prefixo.append(prefixo)
            print('tamanho do prefixo linha ', i, ' da matriz: ', prefixo)

        return vet_prefixo

    # calcula o produto escalar entre um vetor e os outros vetores da matriz de features mas apenas nos prefixos.
    # Como os vetores já estão normalizados então o cálculo será o cosseno.
    # x: vetor consulta que será comparado com os vetores da matriz.
    # matriz: matriz de features.
    # prefixo: vetor com os prefixos de cada vetor da matriz que serão usados para o cálculo do cosseno.
    def dot_between_vectors_prefix(self, x, matriz, prefixo):
        for linha in range(matriz.__len__()):
            # Recupera o prefixo do vetor x de acordo com o tamanho do prefixo que está no vetor prefixo[linha].
            q = x[0:prefixo[linha]]
            # Recupera o sufixo do vetor x de acordo com o tamanho do prefixo que está no vetor prefixo[linha].
            q_sufixo = x[prefixo[linha]:]
            # Recupera o prefixo do vetor atual da matriz[linha] de acordo com o tamanho do prefixo que está no vetor prefixo[linha].
            v = matriz[i][0:prefixo[linha]]
            # Recupera o sufixo do vetor atual da matriz[linha] de acordo com o tamanho do prefixo que está no vetor prefixo[linha].
            v_sufixo = matriz[i][prefixo[linha]:]
            print('Cosseno do prefixo vetor ''x'' com vetor ', linha , " da matriz:", cosine.dot(q, v))
            intersect = 0
            for j in range(v.__len__()):
                if ((v[j] != 0) and (q[j] != 0)):
                    intersect += 1
            print('intersecção entre prefixo do vetor v e linha ', linha, ' da matriz: ', intersect)
            print('Cosseno do sufixo vetor ''x'' com vetor ', linha , " da matriz:", cosine.dot(q_sufixo, v_sufixo))
            intersect = 0
            for j in range(v_sufixo.__len__()):
                if ((v_sufixo[j] != 0) and (q_sufixo[j] != 0)):
                    intersect += 1
            print('intersecção entre sufixo do vetor v e linha ', linha, ' da matriz: ', intersect)
            print('---')

        print('-------------')


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

#seta a variavel consulta com o primeiro vetor da matriz featU.
consulta = featU[0]
vet_prefixo = cosine.intersect_between_vectors(consulta, featU)
print('---------------')
print('Tamanho do vetor feat:')
print(feat.shape)
print('---------------')
print('Cosseno dos prefixos dos vetores com normalização:')
cosine.dot_between_vectors_prefix(consulta, featU, vet_prefixo)
print('---------------')

for i in range(feat.__len__()):
    #if (i == 0 or i == 1 or i == 90):
        cont = 0
        for j in range(feat[i].__len__()):
            if (feat[i][j] != 0):
                #print('feature (', i, ",", j,  "):", feat[i][j])
                cont += 1
        print('qtde features > 0 do vetor', i, ': ', cont)

