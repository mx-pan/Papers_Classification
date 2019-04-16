# -*- coding:utf-8 -*-
# Author：hankcs
# Date: 2018-04-28 10:07

from pyhanlp import *
from test_utility import ensure_data
import os


WordVectorModel = JClass('com.hankcs.hanlp.mining.word2vec.WordVectorModel')
DocVectorModel = JClass('com.hankcs.hanlp.mining.word2vec.DocVectorModel')
model_path = os.path.join(
    ensure_data('hanlp-wiki-vec-zh', 'http://hanlp.linrunsoft.com/release/model/hanlp-wiki-vec-zh.zip'),
    'hanlp-wiki-vec-zh.txt')
word2vec = WordVectorModel(model_path)
doc2vec = DocVectorModel(word2vec)

categories_file = './categories/categories for test_1.txt'
docs = []
categories = open(categories_file,'r')
lines = categories.readlines()
for line in lines:
    line = line[line.find('*')+2:line.rfind('*')-2]
    docs.append(line)
for idx, doc in enumerate(docs):
    doc2vec.addDocument(idx, doc)

#print(word2vec.nearest('语言'))

input_path = './inputs/'
input_list = os.listdir(input_path)
for input_file_name in input_list:
    input_file = open(input_path + input_file_name,'r')
    lines = input_file.readlines()
    title = lines[1]
    print (title)

    # key =str(HanLP.extractKeyword(input, 2))
    print('输入: ' + title)
    print('↓↓↓↓##### 相似度 怕是假的吧 #####↓↓↓↓')
    for res in doc2vec.nearest(title):
        print('%s = %.2f' % (docs[res.getKey().intValue()], res.getValue().floatValue()))