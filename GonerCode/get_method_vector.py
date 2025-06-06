import numpy as np
import os
import json
import time
from multiprocessing import Pool
from functools import partial
import pandas as pd
import pickle

def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)


def mkdata(file, datas, ids):
    with open(file, 'r') as f:
        a = f.read()
        data = a.split(',')[:-1]
        datas.append(data)
        id = file.split('/')[-1].split('.txt')[0]
        ids.append(id)


# def xiancheng(document, typeindex, lenth, max, n):
#     temp = {}
#     for word in document:
#         # 存储每个文档中每个词的词频
#         temp[word] = temp.get(word, 0) + 1  # /len(document)
#
#     feature = [0 for j in range(n)]
#     for q in temp:
#         posotion = typeindex[q]
#         for p in posotion:
#             feature[p] = temp[q]
#
#     matrix = []
#     s = 0
#     for l in lenth:
#         fea = feature[s:s + l]
#         s = s + l
#         fea.extend(np.zeros(max - l, dtype=int))
#         matrix.append(fea)
#     #np.save(outpath, sim)


def main(pickle_path, orderpath, lenthpath, output_pickle_path, n, chunk_size=100):
    # Wczytanie danych z pliku pickle
    df = pd.read_pickle(pickle_path)

    # Grupowanie danych według plików
    grouped = df.groupby('file')['type'].apply(list).reset_index()
    documents = grouped['type'].tolist()
    fileids = grouped['file'].tolist()

    with open(orderpath, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)

    with open(lenthpath, 'r') as f:
        a = f.read()
        le = a.split(',')
        lenth = [int(i) for i in le]
        max_len = np.max(lenth)

    results = []
    chunk_counter = 0

    for i in range(len(documents)):
        document = documents[i]
        print(f"Processing file {i + 1}/{len(documents)}: {fileids[i]}")

        temp = {}
        for word in document:
            # Liczenie częstotliwości wystąpień słów w dokumencie
            temp[word] = temp.get(word, 0) + 1

        feature = [0 for _ in range(n)]
        for q in temp:
            if q in json_data:
                posotion = json_data[q]
                for p in posotion:
                    feature[p] = temp[q]

        matrix = []
        s = 0
        for l in lenth:
            fea = feature[s:s + l]
            s = s + l
            fea.extend(np.zeros(max_len - l, dtype=int))
            matrix.append(fea)

        # Dodanie wyników do listy
        results.append({'file': fileids[i], 'matrix': matrix})

        # Zapis wyników w częściach
        if len(results) >= chunk_size or i == len(documents) - 1:
            chunk_df = pd.DataFrame(results)
            chunk_file = f"{output_pickle_path}_chunk_{chunk_counter}.pkl"
            chunk_df.to_pickle(chunk_file)
            print(f"Zapisano chunk: {chunk_file}")
            results = []
            chunk_counter += 1


# def main_threshold(javapath, orderpath, n):
#     javalist = []
#     listdir(javapath, javalist)
#
#     documents = []
#     fileids = []
#     for javafile in javalist:
#         #print(javafile)
#         mkdata(javafile, documents, fileids)
#     features = {}
#
#     with open(orderpath, 'r', encoding='utf8') as fp:
#         json_data = json.load(fp)
#
#     for i in range(len(documents)):
#         document = documents[i]
#         temp = {}
#         for word in document:
#             # 存储每个文档中每个词的词频
#             temp[word] = temp.get(word, 0) + 1  # /len(document)
#
#         feature = [0 for j in range(n)]
#         for q in temp:
#             posotion = json_data[q]
#             for p in posotion:
#                 feature[p] = temp[q]
#
#         name = fileids[i]
#         print(name)
#         features[name] = feature
#
#     feature_json = 'features_threshold.json'
#     file = open(feature_json, "w")
#     json.dump(features, file, indent=4, separators=(',', ':'))
#     file.close()
#
#
# def main2():
#     # 从文件夹中读取所有Java文件对应的文档
#     javapath = './GCJ_2gram_txt'
#     javalist = []
#     listdir(javapath, javalist)
#
#     documents = []
#     fileids = []
#     for javafile in javalist:
#         print(javafile)
#         mkdata(javafile, documents, fileids)
#
#     with open('order_dict.json', 'r', encoding='utf8') as fp:
#         json_data = json.load(fp)
#
#     with open('2gram_lenth.csv', 'r') as f:
#         a = f.read()
#         le = a.split(',')
#         lenth = [int(i) for i in le]
#         max = np.max(lenth)
#
#     pool = Pool(16)
#     pool.map(partial(xiancheng, typeindex=json_data, lenth=lenth, max=max, n=624), documents)
#
#
# def main3():
#     # 从文件夹中读取所有Java文件对应的文档
#     javapath = './3typetxt_old'
#     javalist = []
#     listdir(javapath, javalist)
#
#     documents = []
#     fileids = []
#     for javafile in javalist:
#         print(javafile)
#         mkdata(javafile, documents, fileids)
#
#     with open('./268/268_14884_dict.json', 'r', encoding='utf8') as fp:
#         json_data = json.load(fp)
#
#     with open('./268/268lenth.csv', 'r') as f:
#         a = f.read()
#         le = a.split(',')
#         lenth = [int(i) for i in le]
#         max = np.max(lenth)
#
#     pool = Pool(8)
#     pool.map(partial(xiancheng, typeindex=json_data, lenth=lenth, max=max, n=14884), documents)


# n的值：BCB—3gram-14884，BCB—2gram-1106，GCJ—2gram-624  withoutposition: BCB—2gram-1106   GCJ—2gram-624


if __name__ == '__main__':
    start1 = time.time()
    # main('./GCJ_2gram_txt', '2-gram-order-dict.json', '2-gram-lenth.csv', './GCJ_2_matrix/', 624)  # gcj-2-gram
    main('.\\BCB_2-gram_results.pkl', '.\\GonerCode\\BCB_2gram\\2-gram-order-dict.json', '.\\GonerCode\\BCB_2gram\\2-gram-lenth.csv', '.\\MethodVectors\\BCB_2_matrix.pkl', 1106)  # bcb-2-gram
    end1 = time.time()
    t1 = end1 - start1
    print('generate feature time:')
    print(t1)

   # start2 = time.time()
   # main('./BCB_3gram_txt/', './BCB_3gram/3-gram-order-dict.json', './BCB_3gram/3-gram-lenth.csv', './BCB_3_matrix/', 14884)  # bcb-3-gram
   # end2 = time.time()
   # t2 = end2 - start2
   # print('generate feature time:')
   # print(t2)
   # print(t1)
