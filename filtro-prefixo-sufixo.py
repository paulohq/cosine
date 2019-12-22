#Faz o cálculo da similaridade do cosseno com os vetores lidos do arquivo enwiki-vector-20.txt.

## feature extraction
import pandas as pd
import numpy as np
import nltk
import gensim
import scipy.sparse as sp
from numpy.linalg import norm
import math


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
        self.Index = []
        self.se = []
        self.Ndi = []

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
    #def cosine_between_vectors(self, i, matriz):
    #    for linha in range(matriz.__len__()):
    #        print('Cosseno de', i, 'com', linha , ":", cosine.cos(matriz[i], matriz[linha]))

    def cosine_between_vectors(self, v1, v2):
        print('Cosseno entre v1 e v2:', cosine.cos(v1, v2))

    # calcula o produto escalar entre um vetor especificado e os outros vetores da matriz de features (inclusive com ele mesmo).
    # Como os vetores já estão normalizados então o resultado do cálculo ( x * y ) será igual ao resultado do cálcuo do cosseno ( x * y/(norma(x) * norma(y)) ).
    # i: índice do vetor que será comparado com os outros vetores.
    # matriz: matriz de features.
    #def dot_between_vectors(self, i, matriz):
    #    for linha in range(matriz.__len__()):
    #        print('Cosseno de', i, 'com', linha , ":", cosine.dot(matriz[i], matriz[linha]))

    def dot_between_vectors(self, v1, v2):
        x = cosine.dot(v1, v2)
        print('Cosseno entre v1 e v2:', x)
        return x

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
            # k: a intersecção entre o vetor v e o vetor i da matriz
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
            v = matriz[linha][0:prefixo[linha]]
            # Recupera o sufixo do vetor atual da matriz[linha] de acordo com o tamanho do prefixo que está no vetor prefixo[linha].
            v_sufixo = matriz[linha][prefixo[linha]:]
            cos_prefixo = cosine.dot(q, v)
            print('Cosseno do prefixo vetor ''x'' com vetor ', linha , " da matriz:", cos_prefixo)
            intersect = 0
            for j in range(v.__len__()):
                if ((v[j] != 0) and (q[j] != 0)):
                    intersect += 1
            print('intersecção entre prefixo do vetor v e linha ', linha, ' da matriz: ', intersect)
            cos_sufixo = cosine.dot(q_sufixo, v_sufixo)
            print('Cosseno do sufixo vetor ''x'' com vetor ', linha , " da matriz:", cos_sufixo)
            intersect = 0
            for j in range(v_sufixo.__len__()):
                if ((v_sufixo[j] != 0) and (q_sufixo[j] != 0)):
                    intersect += 1
            print('intersecção entre sufixo do vetor v e linha ', linha, ' da matriz: ', intersect)
            print('Cosseno entre o vetor ''x'' e o vetor ', linha, " da matriz: ", cos_prefixo + cos_sufixo)
            cosseno_total = cosine.dot_between_vectors(x, matriz[linha])
            print("Cosseno total: {0:18.16f},  Cosseno do prefixo + sufixo: {1:18.16f},  Diferença: {2:18.16f}".format(cosseno_total, cos_prefixo + cos_sufixo, cosseno_total - (cos_prefixo + cos_sufixo) ))
            print('---------------------------------------------------------------------------------------')


        print('-------------')

    def index(self, id_documento, d, Index, se, t):
        #norma do sufixo do vetor recebe o valor máximo
        b = 1
        #variável auxiliar para armazenar a posiçaõ da última característica do prefixo.
        j_aux = 0
        #Laço para percorres todas as características do vetor d
        for j in range(len(d)):
            #Se o valor da característica do vetor d for maior que zero e a raiz quadrada do sufixo do vetor d for maior ou igual ao threshold
            #então indexa o documento d, o valor da característica j e o sufixo de d (a partir da característica j).
            if ((d[j] > 0) & ( math.sqrt(b) >= t)):
                #calcula o valor da característica j ao quadrado e subtrai da norma do sufixo
                b = b - (d[j] * d[j])
                if (b < 0):
                    b = 0
                #Calcula a norma do sufixo do vetor d a partir da característica j (produto escalar entre o vetor e ele mesmo).
                sufixo_d = cosine.dot(d[j+1:], d[j+1:])
                j_aux = j + 1
                #norma = cosine.norm(d)
                #id do documento, id da característica, o valor da característica e a norma do sufixo de (a partir da característica j).
                r = [id_documento, j, d[j], sufixo_d]
                # indexa o id do documento, o id da característica (mas precisa ser o id do documento), o valor da característica j e o sufixo de d (a partir da característica j).
                self.Index.append(r)
            #else:
            #    break

        prefixo_d = cosine.dot(d[0:j_aux], d[0:j_aux])
        #Armazena o id do documento, a norma do sufixo e o posição da última posição do prefixo indexado + 1 que é a primeira posição do sufixo.
        se_aux = [id_documento, sufixo_d, j_aux]
        # Armazena o sufixo do vetor d no vetor suffix estimate.
        self.se.append(se_aux)
        print("suffix extimate:", self.se)
        print("prefix + suffix:", prefixo_d + sufixo_d)
        #return Index, se

    def findNeighbors(self, di, t, featU):
        #Armazena o valor máximo da norma do vetor di.
        r = []

        #Variável auxiliar para armazenar o valor acumualado de similaridade.
        #acum = []
        #Vetor que armazena o id do documento candidato e o valor acumulado da similaridade entre o documento de consulta e o documento candidado.
        #[id_documento, valor_similaridade]
        A = []
        for x in self.Index:
            A = np.insert(A, x[0], x[0])

        A = sorted(set(A))
        r = sorted(set(A))
        for i in range(len(A)):
            A[i] = 0
            r[i] = 1


        #Fase de geração de candidatos (indexa o prefixo).
        #Percorre as características do vetor di (consulta) passado como parâmetro.
        for j in range(len(di)):
            if (di[j] > 0):
                #Percorre o array Index com o registro (dc = id_documento, c = id_caracteristica, dcj = valor_caracteristica, norma_sufixo_dc = norma do sufixo do vetor candidato).
                for (dc, c, dcj, norma_sufixo_dc) in self.Index:
                    #Se a característica do vetor dc é igual a característica do vetor di.
                    if (c == j):
                        #Se o acumulado do documento maior que zero ou acumulado do documento diferente de zero e a norma do sufixo maior que threshold.
                        if (math.sqrt(r[dc]) >= t):
                            acum = A[dc]
                            #Acumulado do documento recebe acumulado do documento + similaridade entre a característica de di e dc.
                            acum = acum + (di[j] * dcj)

                            # Adciona à variável acumulador o id do documento e o valor acumulado da similaridade entre di e dc.
                            A[dc] = acum

                            #calcula a norma do sufixo do vetor d a partir da característica j + 1.
                            norma_sufixo_di = cosine.dot(di[j+1:],di[j+1:])
                            #Se acumulado do documento somado com norma do sufixo de di multiplicado com a norma do sufixo de dc é menor que o threshold
                            #então zera o acumulado (funciona como poda).
                            if (acum + (norma_sufixo_di * norma_sufixo_dc) < t):
                                A[dc] = acum
                                #A = np.insert(A, dc , 0)
                        #subtrai da norma do sufixo o valor da característica j ao quadrado.
                        r[dc] = r[dc] - (di[j] * di[j])

        #Fase de verificação de candidatos (indexa o sufixo do vetor)
        #Percorre o vetor com os valores acumulados na fase de geração de candidatos.
        for (dc, acumulado_dc) in enumerate(A):
            #Se o valor acumulado (na fase candidate generation) do candidato k for maior que zero então faça.
            if (acumulado_dc > 0):
                #Se o valor acumulado do candidado k somado com o suffix estimate do candidato k for maior que o threshold então
                #se for menor o candidato será podado, pois passa para o próximo candidado no loop.
                if (acumulado_dc + cosine.se[dc][1] >= t):
                    #Percorre as características do sufixo do vetor.
                    for j in range( cosine.se[dc][2], len(di) ):
                        #se o valor da característica de dc é maior que zero e o valor da característica de di é maior que zero.
                        if (featU[dc][j] > 0 and di[j] > 0):
                            #Acumula o valor da similaridade entre o vetor di e dc.
                            A[dc] = A[dc] + (di[j] * featU[dc][j])

                            # Calcula a norma do sufixo do vetor d a partir da característica j.
                            norma_sufixo_d = cosine.dot(di[j:], di[j:])
                            # Calcula a norma do sufixo do vetor dc a partir da característica j.
                            norma_sufixo_dc = cosine.dot(featU[dc][j:], featU[dc][j:])
                            #Se o valor acumulado somado como a norma do sufixo de d multiplicado com a norma do sufixo de dc
                            #for menor que threshold então passa para o próximo candidato do vetor acumulado.
                            if (A[dc] + (norma_sufixo_d * norma_sufixo_dc) < t):
                                break
                #Se o acumulado do candidato for maior que threshold então adiciona o candidato e o seu valor acumulado
                #no vetor de vetores similares de di.
                if (A[dc] > t):
                    self.Ndi.append([dc, A[dc]])
        print("Vetores após aplicação do filtro:")
        print(self.Ndi)



read = text()
cosine = cosine()

feat = np.array(read.read_text("enwiki-vector-4.txt"))

#feat = np.array([[5,0,9,6]
#                ,[0,8,4,7]
#                ,[3,4,8,0]
#                ,[9,3,7,5]])

#feat = np.array([[1,2,3,4,5,6,7,8,9,10],
#                [0,0,0,0,0,1,2,3,4,5],
#                [0,0,0,0,0,6,7,8,9,10],
#                [1,2,3,4,5,1,2,3,4,5]])


#a = np.array([1,2,3,4,5,6,7,8,9,10,11])

#print(cosine.prefix(a, round(a.__len__()/2)))
#print(cosine.suffix(a, round(a.__len__()/2) - 1))

#print(cosine.prefix(x,2))
#print(cosine.suffix(x,2))
#print('---------------')

featU = cosine.toUnitMatrix(feat)
for i in range(featU.__len__()):
    featNorm = cosine.norm(featU[i])

Index = [[] ]
se = [[]]
threshold = 0.8
for i in range (featU.__len__()):
    cosine.index(i, featU[i], Index, se, threshold)

#for i in range (featU.__len__()):
cosine.findNeighbors(featU[0], threshold, featU)


#print('Cosseno com os vetores sem normalização:')
#for linha in range(feat.__len__()):
#    cosine.cosine_between_vectors(feat[0], feat[linha])

#print('Cosseno com os vetores com normalização:')
#for linha in range(feat.__len__()):
#    cosine.dot_between_vectors(featU[0], featU[linha])
#print('---------------')

#seta a variavel consulta com o primeiro vetor da matriz featU.
#consulta = featU[0]
#vet_prefixo = cosine.intersect_between_vectors(consulta, featU)
#print('---------------')
#print('Tamanho do vetor feat:')
#print(feat.shape)
#print('---------------')
#print('Cosseno dos prefixos dos vetores com normalização:')
#cosine.dot_between_vectors_prefix(consulta, featU, vet_prefixo)
#print('---------------')

