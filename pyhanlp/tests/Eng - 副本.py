# coding: gbk
from os import listdir
import os.path
import os
import numpy as np
import pandas as pd
import re
import sys
import importlib
importlib.reload(sys)
from gensim.models import Word2Vec
import logging,gensim,os
from gensim import corpora
from gensim import corpora, models, similarities

def GetFileList(dir, FileList):
    newDir = dir
    if os.path.isfile(dir):
        FileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            # if s == "xxx":
            # continue
            newDir = os.path.join(dir, s)
            GetFileList(newDir, FileList)
    return FileList

def GetAbstract(FileList):
    content = pd.concat([pd.read_table(name, encoding="gbk") for name in FileList], axis=1)#读取待分类文件及内容
    content=content.T
    content.columns=['title','no','Key_word','no','Abstract']
    content.drop(['no'],axis=1,inplace=True)
    Abstract= content['Abstract']
    return Abstract

def GetLabelList(LabelPath):#输入包含标签的文件绝对路径
    f = open(LabelPath, encoding="gbk")
    LabelList = []
    for line in f.readlines():
         LabelList.append(line.strip('\n').strip('*'))
    f.close()
    return LabelList

def CleanText(text,StopList):#格式转换
    texts=list(text)
    text = [[word for word in label.lower().split() if word not in StopList]
             for label in texts]
    return text

def GetDictionary(FileList,StopList,SaveDictPath):#将FileList所有文件合并，自动生成字典
    class TextLoader(object):
        def __init__(self):
            pass

        def __iter__(self):
            for file in FileList:
                input = open(file,'r',encoding='gbk')
                line = str(input.readline())
                counter = 0
                while line!=None and len(line) > 4:
                #print line
                    segments = line.lower().split(' ')
                    yield  segments
                    line = str(input.readline())
    sentences = TextLoader()
    texts = [[word for word in sentences if word not in StopList]#将训练文本去除停用词
             for sentences in sentences]
    dictionary = corpora.Dictionary(texts)#建立词典并保存
    dictionary.save(SaveDictPath)
    return dictionary


def Classify(StopList, FilePath, SaveDictPath, LabelPath):
    FileList = GetFileList(FilePath, [])
    LabelList = GetLabelList(LabelPath)
    labeltexts = CleanText(LabelList, StopList)
    dictionary = GetDictionary(FileList, StopList, SaveDictPath)
    corpus = [dictionary.doc2bow(text) for text in labeltexts]
    Abstract = GetAbstract(FileList)
    Abstracts = list(Abstract)

    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=40)  # 在 lsi 空间中计算文本相似度
    index = similarities.MatrixSimilarity(lsi[corpus])
    res = []
    for i in range(len(Abstract)):
        new_doc = Abstracts[i]
        new_doc = new_doc.lower().split()
        new_vec = [word for word in new_doc if word not in StopList]
        vec_bow = dictionary.doc2bow(new_vec)
        vec_lsi = lsi[vec_bow]
        sims = index[vec_lsi]
        res.append(FileList[i] + ' 属于 ' + LabelList[np.argmax(sims)])
    return res

StopList = set('\n,.\their been it we in such both which abstract(#br)in or has was new the from by that as these the have be were an by with for a of the and to in （#br）paper **题目**\n **关键词**\n **摘要**\n a on is are this this'.split())

FilePath = '/Users/yuchengqi/PycharmProjects/Text_segmentation/英文样本集'#此处为包含文件的
SaveDictPath='/Users/yuchengqi/PycharmProjects/Text_segmentation/control.dict'#'希望保存字典文件的路径，可以随便写一个
LabelPath='categories for test_2.txt'#标签文件的绝对路径
ResultPath='/Users/yuchengqi/PycharmProjects/Text_segmentation/EnglishResult.txt'#输出结果的路径
res=Classify(StopList,FilePath,SaveDictPath,LabelPath)#res为输出结果

fresult = open(ResultPath,'w')
fresult.write('\n'.join(res))
fresult.close()