#usr/bin/python
path = "D:\\papers_classification\\pyhanlp\\tests\\categories_test\\英文论文测试数据集\\test_2_1.txt"
new_path = 'codingtest.txt'
file = open(path,'r',encoding = 'gbk')

lines = file.readlines()
for line in lines:
    print (line)
    with open(new_path,'a',encoding = 'gbk') as new_file:
        new_file.write(line)