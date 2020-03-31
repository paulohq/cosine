# Lê o arquivo binário gerado no formato:
#<LI999999I999999f>
# [4-byte ID_DOC][4-byte RECORD SIZE (qtde de features no documento)][4-byte FEATURE_ID][4-byte FEATURE_VALUE]
import struct
import numpy as np

class read_file(object):
    def __init__(self):
        self.data_path = ""

    #Grava arquivo binário com os documento e features, formato:
    #doc_id feature1_id feature1_value feature2_id feature2_value feature3_id feature3_value ...
    #Grava o id do documento, a quantidade de features do documento e grava todos os id's de features juntos e depois todos os valores das features.
    def write_tfidf_binary(self, featU, out_path = 'bin_file_tfidf'):

        fd_out = open(out_path, 'wb')

        # [4-byte ID_DOC][4-byte RECORD SIZE][4-byte FEATURE_ID][4-byte FEATURE_VALUE]...
        # format: <IL999999999999If

        doc_id = 0
        feature_id = []
        feature_value = []
        fmt = ''
        for d in featU:
            count = 0;
            feature_id = []
            feature_value = []
            for feat_id in range(len(d)):
                if (d[feat_id] != 0):
                    count += 1
                    feature_id.append(feat_id)
                    feature_value.append(d[feat_id])

            fmt = '<LI'+str(count)+'I'+str(count)+'f'
            entry = struct.pack(fmt, doc_id, count, *feature_id, *feature_value)
            fd_out.write(entry)
            fd_out.flush()
            doc_id += 1

        fd_out.close()

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

    #Grava arquivo binário com os documento e features, formato:
    #doc_id feature_amount feature1_id feature1_value feature2_id feature2_value feature3_id feature3_value ...
    # Grava o id do documento, a quantidade de features do documento e grava os pares id e valor das features juntos.
    def write_tfidf_pairs_binary(self, featU, out_path = 'bin_file_tfidf'):

        fd_out = open(out_path, 'wb')

        # [4-byte ID_DOC][4-byte FEATURE_AMOUNT][4-byte FEATURE_ID][4-byte FEATURE_VALUE]...
        # format: <LIIf

        doc_id = 0
        feature_id = []
        feature_value = []
        fmt = ''
        for d in featU:
            count = 0;
            feature_id = []
            feature_value = []
            for feat_id in range(len(d)):
                if (d[feat_id] != 0):
                    count += 1
                    feature_id.append(feat_id)
                    feature_value.append(d[feat_id])

            #Grava o id do documento e a quantidade de features do documento.
            fmt = '<LI'
            entry = struct.pack(fmt, doc_id, count)
            fd_out.write(entry)
            fd_out.flush()

            #Percorre os vetores com os id's e valores das features
            for i in range(feature_id.__len__()):
                #Grava cada par de id e valor das features.
                fmt = '<If'
                entry = struct.pack(fmt, feature_id[i], feature_value[i])
                fd_out.write(entry)
                fd_out.flush()

            doc_id += 1

        fd_out.close()

    # Lê arquivo binário com os documento e features, formato:
    # record size (quantidade de feature_id e feature value do documento) doc_id feature1_id feature1_value feature2_id feature2_value feature3_id feature3_value ...
    def read_tfidf_pairs_binary(self, fname='bin_file_tfidf'):
        # file = open(fname, 'rb')

        # [4-byte ID_DOC][4-byte RECORD SIZE][4-byte FEATURE_ID][4-byte FEATURE_VALUE]
        # format: <LIIf
        docs = []
        pairs = []
        feature = []
        #Abre arquivo binário para leitura
        with open(fname, "rb") as fid:
            while True:
                for i in range(2):
                    pairs = []

                    #Recupera os primeiros 4 bytes do arquivo com o id do documento.
                    b_id_doc = fid.read(4)
                    if not b_id_doc: break
                    #armazena o id do documento codificado em numérico (long) na variável id_doc
                    id_doc = struct.unpack('<L', b_id_doc)
                    #adiciona o id do documento em uma lista.
                    docs.append(id_doc[0])

                    #recupera os próximos 4 bytes do arquico com a quantidade de features do documento.
                    size_record = fid.read(4)
                    if not size_record: break

                    #armazena a quantidade codificada em numérico (integer) na variável size.
                    size = struct.unpack('<I', size_record)

                    sz = size[0]
                    #Laço para percorrer a quantidade de features (feature_id) que o documento tem.
                    for i in range((sz)):
                        #recupera o id da feature armazenado em 4 bytes.
                        b_feature = fid.read(8)
                        if not b_feature: break
                        #armazena o id da feature codificada em numérico (integer).
                        pair = struct.unpack('<If', b_feature)
                        pairs.append(pair)

                    # adiciona os vetores com os id's e valores das feautures.
                    feature.append(pairs)
                else:
                    continue
                break
        fid.close()

read = read_file()

read.read_tfidf_pairs_binary('bin-dataset-20')