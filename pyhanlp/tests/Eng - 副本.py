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
            # �����Ҫ����ĳЩ�ļ��У�ʹ�����´���
            # if s == "xxx":
            # continue
            newDir = os.path.join(dir, s)
            GetFileList(newDir, FileList)
    return FileList

def GetAbstract(FileList):
    content = pd.concat([pd.read_table(name, encoding="gbk") for name in FileList], axis=1)#��ȡ�������ļ�������
    content=content.T
    content.columns=['title','no','Key_word','no','Abstract']
    content.drop(['no'],axis=1,inplace=True)
    Abstract= content['Abstract']
    return Abstract

def GetLabelList(LabelPath):#���������ǩ���ļ�����·��
    f = open(LabelPath, encoding="gbk")
    LabelList = []
    for line in f.readlines():
         LabelList.append(line.strip('\n').strip('*'))
    f.close()
    return LabelList

def CleanText(text,StopList):#��ʽת��
    texts=list(text)
    text = [[word for word in label.lower().split() if word not in StopList]
             for label in texts]
    return text

def GetDictionary(FileList,StopList,SaveDictPath):#��FileList�����ļ��ϲ����Զ������ֵ�
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
    texts = [[word for word in sentences if word not in StopList]#��ѵ���ı�ȥ��ͣ�ô�
             for sentences in sentences]
    dictionary = corpora.Dictionary(texts)#�����ʵ䲢����
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

    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=40)  # �� lsi �ռ��м����ı����ƶ�
    index = similarities.MatrixSimilarity(lsi[corpus])
    res = []
    for i in range(len(Abstract)):
        new_doc = Abstracts[i]
        new_doc = new_doc.lower().split()
        new_vec = [word for word in new_doc if word not in StopList]
        vec_bow = dictionary.doc2bow(new_vec)
        vec_lsi = lsi[vec_bow]
        sims = index[vec_lsi]
        res.append(FileList[i] + ' ���� ' + LabelList[np.argmax(sims)])
    return res

StopList = set('\n,.\their been it we in such both which abstract(#br)in or has was new the from by that as these the have be were an by with for a of the and to in ��#br��paper **��Ŀ**\n **�ؼ���**\n **ժҪ**\n a on is are this this'.split())

FilePath = '/Users/yuchengqi/PycharmProjects/Text_segmentation/Ӣ��������'#�˴�Ϊ�����ļ���
SaveDictPath='/Users/yuchengqi/PycharmProjects/Text_segmentation/control.dict'#'ϣ�������ֵ��ļ���·�����������дһ��
LabelPath='categories for test_2.txt'#��ǩ�ļ��ľ���·��
ResultPath='/Users/yuchengqi/PycharmProjects/Text_segmentation/EnglishResult.txt'#��������·��
res=Classify(StopList,FilePath,SaveDictPath,LabelPath)#resΪ������

fresult = open(ResultPath,'w')
fresult.write('\n'.join(res))
fresult.close()