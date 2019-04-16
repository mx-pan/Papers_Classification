#usr/bin/python
# -*- coding:utf-8 -*-
import easygui as g
import os
from pyhanlp import *
from test_utility import ensure_data
import numpy as np
import sys
from pyhanlp import SafeJClass

# 输出
ouput_path = 'output'
if os.path.exists(ouput_path):
    os.system('rd /s /q '+ouput_path)
os.system('md '+ouput_path)
correct_classifications = ouput_path + '/correct_classifications.txt'
# if os.path.isfile(correct_classifications):
    # os.system("del "+ correct_classifications)
os.system("type nul> "+ correct_classifications)
incorrect_classifications = ouput_path + '/incorrect_classifications.txt'
# if os.path.isfile(incorrect_classifications):
    # os.system("del "+incorrect_classifications)
os.system("type nul> "+incorrect_classifications)
output_by_category_path = ouput_path + '\\output_by_category'
os.system("md "+output_by_category_path)
output_by_paper = ouput_path + '/output_by_paper.txt'
os.system("type nul> "+output_by_paper)

# 读取模型
WordVectorModel = JClass('com.hankcs.hanlp.mining.word2vec.WordVectorModel')
DocVectorModel = JClass('com.hankcs.hanlp.mining.word2vec.DocVectorModel')
#added, model 2
NaiveBayesClassifier = SafeJClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')
# ↑ ↑ ↑
# 使用根据hanlp训练的词向量1↓
model_path_chn = os.path.join(
    ensure_data('hanlp-wiki-vec-zh', 'http://hanlp.linrunsoft.com/release/model/hanlp-wiki-vec-zh.zip'),
    'hanlp-wiki-vec-zh.txt')
# 使用根据维基百科训练的词向量1↓
# model_path_eng ='D:\\papers_classification\\pyhanlp\\tests\\data\\wiki.zh.vec'
# model_path = ensure_data('wiki.zh.vec', '')
# 使用根据百度百科训练的词向量↓
model_path_eng ='D:\\papers_classification\\pyhanlp\\tests\\data\\sgns.baidubaike.bigram-char'
# 使用根据维基百科训练的词向量2↓
# model_path_eng ='D:\\papers_classification\\pyhanlp\\tests\\data\\sgns.wiki.bigram-char'



# 欢迎窗口
def Welcome(Title):
    if g.msgbox("您好！\n欢迎使用（不）智能论文分类器！", title = Title, ok_button = "开始分类"):
        pass
    else:
        sys.exit(0)

# 选择语言
def Language(Title):
    language = g.buttonbox("请选择语言在种类",choices=("中文","English"))
    if not language:
        sys.exit(0)
    return language

# 输入路径窗口
def InputPaths(Title):
    path_names = ["类别文件地址","待分类文本地址"]
    paths_default = ["categories\\categories for test_1.txt","categories\\中文样本集\\大数据采集、存储、分析、处理与应用"]
    input_paths = g.multenterbox("请提供以下文件的路径：",Title,path_names,paths_default)
    return input_paths
    if not input_paths:
        sys.exit(0)

# 确认类别文件窗口
def CategoryTxt(Title,lines):
    if g.textbox("请确认类别文件", Title, lines):
        pass
        return 0
    else:
        return 1

# 分类完毕确认窗口
def Complete(Title):
    ifcontinue = g.ynbox("分类完毕！结果请见output文件夹！\n是否继续分类？",title = Title)
    return ifcontinue

# 再见窗口
def Exit(Title):
    g.msgbox("感谢使用，再见！", title=Title, ok_button="ヾ(￣▽￣)Bye~Bye~")
    sys.exit(0)

def  Main(welcomed = 0): 
    Title = "（不）智能论文分类器"
    # 欢迎页
    if welcomed == 0:
        Welcome(Title)
    
    # 语言选择 
    language = Language(Title)
    if language == "中文":
        model_path = model_path_chn
    else:
        model_path = model_path_eng
    print(model_path)
    word2vec = WordVectorModel(model_path)
    doc2vec = DocVectorModel(word2vec)
    
    # 读取类别文件
    input_paths = InputPaths(Title)    
    categories_file = input_paths[0]
    if categories_file[-3:] != "txt":
        g.msgbox("请输入类别txt文件地址！", title = Title, ok_button="好的，我重输")
        input_paths = InputPaths(Title)

    # 读取类别
    categories_file = input_paths[0]
    docs = []
    categories = open(categories_file,'r')
    lines = categories.readlines()
    for line in lines:
        line = line[line.find('*')+2:line.rfind('*')-2]
        docs.append(line)
        # 创建输出文件
        
        os.system('mkdir ' + output_by_category_path + '\\' + line)
        os.system("type nul> "+ output_by_category_path + '\\' + line + '\\all.txt')
        with open(output_by_category_path + '\\' + line + '\\all.txt','w') as f:
            f.write(line+'\n')
    for idx, doc in enumerate(docs):
        doc2vec.addDocument(idx, doc)
    categories_check = CategoryTxt(Title,lines)
    while categories_check !=0 :
        InputPaths(Title)
        categories_check = CategoryTxt(Title,lines)

    # 检验输入路径正确性
    papers_path = input_paths[1]
    if papers_path[-1:] != '\\':
        papers_path += '\\'

    correct_label = papers_path[papers_path.find("中文样本集")+6:-1]
    correct_num = 0

    # 整理输入文件
    input_list = os.listdir(papers_path)
    count = 0
    for input_file_name in input_list:
        input_file = open(papers_path + input_file_name,'r')
        lines = input_file.readlines()
        paper_title = lines[1]
        paper_keywords = lines[3]
        paper_abstract = ''
        for i in range(5,len(lines)):
            paper_abstract += lines[i]

        biases = [paper_title, paper_keywords, paper_abstract]
        biases_name = ["标题","关键词","摘要"]
        biases_weights = [1,1,0.5]
        confidence = np.zeros(len(docs))
        for i,bias in enumerate(biases):
            # key =str(HanLP.extractKeyword(input, 2))
            print(biases_name[i] + ': ' + bias)
            # g.msgbox(biases_name[i], title = Title, ok_button="ok")
            # print('↓↓↓↓##### 相似度 怕是假的吧 #####↓↓↓↓')
            
            # for res in doc2vec.nearest(title):
            #     print('%s = %.2f' % (docs[res.getKey().intValue()], res.getValue().floatValue()))
            output_list = []
            for res in doc2vec.nearest(bias):
                output_temp = str(docs[res.getKey().intValue()]) + ' ' + str(res.getValue().floatValue())
                output_list.append(output_temp)
            #print('%s = %.2f' % (docs[res.getKey().intValue()], res.getValue().floatValue()))
            # print (docs)
            for output in output_list:
                label = output[:output.rfind(' ')]
                confidence[docs.index(label)] += float(output[output.rfind(' ')+1:]) * biases_weights[i]
            print ('最相近类别：'+ output_list[0]) 
        print ("Correct Label: " + correct_label)
        predicted_label = docs[int(np.where(confidence == np.max(confidence))[0])]
        print("Prediction: " + predicted_label)
        # g.msgbox("分类结果：\n"+predicted_label, title=Title, ok_button="ok")
        count = count + 1
        print(count)

        # 训练用↓
        if predicted_label.find(correct_label) >=0:
            correct_num += 1
            which_file = correct_classifications
        else:
            which_file = incorrect_classifications
        with open(which_file,'a') as f:
                f.write('#'+str(count)+'\n')
                f.write(bias)
                f.write("Correct Label: " + correct_label+'\n')
                f.write("Prediction: " + output_list[0]+'\n')
                f.write("\n")
        
        # 输出结果
        # 按文章
        with open(output_by_paper,'a') as f:
            f.write('#'+str(count) + ' ' + input_file_name + '\n')
            f.write(predicted_label + '\n')

        # 按类别
        output_by_category_txt_all = output_by_category_path + '\\' + predicted_label + '\\all.txt'
        with open(output_by_category_txt_all,'a') as f:
            f.write(input_file_name + '\n')
        os.system('copy ' + papers_path + input_file_name + ' ' + output_by_category_path + '\\' + predicted_label )
        print ('copy ' + papers_path + input_file_name + ' ' + output_by_category_path + '\\' + predicted_label)
    print ("Accuracy: "+ str(correct_num / count))
    ifcontinue = Complete(Title)
    
    if ifcontinue == True:
        Main(1)
    else:
        Exit(Title)
Main(0)

    
    