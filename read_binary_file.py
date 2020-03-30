# Lê o arquivo binário gerado no formato:
#<LI999999I999999f>
# [4-byte ID_DOC][4-byte RECORD SIZE (qtde de features no documento)][4-byte FEATURE_ID][4-byte FEATURE_VALUE]
import struct
import numpy as np

class read_file(object):
    def __init__(self):
        self.data_path = ""

    # Lê arquivo binário com os documento e features, formato:
    # record size (quantidade de feature_id e feature value do documento) doc_id feature1_id feature1_value feature2_id feature2_value feature3_id feature3_value ...
    def read_tfidf_binary(self, fname='bin_file_tfidf'):
        # file = open(fname, 'rb')

        # [4-byte ID_DOC][4-byte RECORD SIZE][4-byte FEATURE_ID][4-byte FEATURE_VALUE]
        # format: <LI999999999999If
        docs = []
        feat_id = []
        feat_value = []
        feat = []
        #Abre arquivo binário para leitura
        with open(fname, "rb") as fid:
            while True:
                for i in range(2):
                    feat_id = []
                    feat_value = []

                    #Recupera os primeiros 4 bytes do arquivo com o id do documento.
                    b_id_doc = fid.read(4)
                    if not b_id_doc: break
                    #armazena o id do documento codificado em numérico (long) na variável id_doc
                    id_doc = struct.unpack('<L', b_id_doc)
                    #adiciona o id do documento em uma lista.
                    docs.append(id_doc[0])

                    #recupera os primeiros 4 bytes do arquico com a quantidade de features do documento.
                    size_record = fid.read(4)
                    if not size_record: break

                    #armazena a quantidade codificada em numérico (integer) na variável size.
                    size = struct.unpack('<I', size_record)

                    sz = size[0]
                    #Laço para percorrer a quantidade de features (feature_id) que o documento tem.
                    for i in range((sz)):
                        #recupera o id da feature armazenado em 4 bytes.
                        b_feature_id = fid.read(4)
                        if not b_feature_id: break
                        #armazena o id da feature codificada em numérico (integer).
                        feature_id = struct.unpack('<I', b_feature_id)
                        feat_id.append(feature_id[0])

                    #Laço para percorrer a quantidade de features (feature_value) que o documento tem.
                    for j in range(sz):
                        # recupera o valor da feature armazenado em 4 bytes.
                        b_feature_value = fid.read(4)
                        if not b_feature_value: break
                        # armazena o valor da feature codificada em numérico (float).
                        feature_value = struct.unpack('<f', b_feature_value)
                        feat_value.append(feature_value[0])


                    #adiciona os vetores com os id's e valores das feautures.
                    feat.append([feat_id, feat_value])

                else:
                    continue
                break
        fid.close()

read = read_file()

read.read_tfidf_binary('bin-dataset-20')