# -*- coding:utf-8 -*-
from pyhanlp import *

# txt_path = 'tests\\categories\\categories for test_1.txt'
# txt = open(txt_path,'r')
# lines = txt.readlines()
# for line in lines:
#     print (line)
#     print(HanLP.segment(line))
# print(HanLP.segment('你好，欢迎在Python中调用HanLP的API'))
'''
for term in HanLP.segment('下雨天地面积水'):
    print('{}\t{}'.format(term.word, term.nature)) # 获取单词与词性
testCases = 
    "商品和服务",
    "结婚的和尚未结婚的确实在干扰分词啊",
    "买水果然后来世博园最后去世博会",
    "中国的首都是北京",
    "欢迎新老师生前来就餐",
    "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
    "随着页游兴起到现在的页游繁盛，依赖于存档进行逻辑判断的设计减少了，但这块也不能完全忽略掉。"]
for sentence in testCases: print(HanLP.segment(sentence))
# 关键词提取
document = "水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" \
           "根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，" \
           "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" \
           "严格地进行水资源论证和取水许可的批准。"
print(HanLP.extractKeyword(document, 2))
# 自动摘要
print(HanLP.extractSummary(document, 3))
# 依存句法分析
print(HanLP.parseDependency("徐先生还具体帮助他确定了把画雄鹰、松鼠和麻雀作为主攻目标。"))
'''
# 关键词提取
document = "光纤传感网络广泛地应用在我们的日常生活当中,在重大工程健康检测、航空航天、安防等领域起着重要的作用。可是由于传感类型繁多,网络庞杂,需要的维护检测成本高,尤其专门的检测设备动辄几十万以上,且存在设备操作复杂、界面不友好、体型庞大、不能移动测量、功能单一等问题,给传感网络的问题定位、排查故障线路等带来诸多问题。基于以上困境,本文对现存的传感网络进行深入研究,研究了光纤传感网络的相关理论和技术特点,区别传感网络的适用测试范围。重点研究了光纤传感网络的光信号接收检测,致力于研究制作出能够检测微弱信号的光电探测器,并使探测器便于携带,易于升级和扩展应用,与其他外设设备组合使用,降低接收成本且能保证探测的可靠性。为了提高便携性,可扩展性,结合市场广泛存在的智能手机作为探测器的后续转换显示设备,将探测器结合手机作为便携式的仪器仪表,用于光纤传感网络微弱信号的检测。本文主要内容及成果有:1、详细论述了便于与手机结合的光电探测模块的设计过程,具体内容包括:探测器的设计原理、光电探测器与手机的接口电路的设计、器件选型、噪声分析、仿真、原理图和印制板图设计、焊接电路板、实验验证等。创新点为:(1)将光探测器检转换的后的电压信号再经电压/频率(V/F)转换的方式输出到手机,结合手机上安装的APP和其自身音频转换模块进行后续处理,并在手机屏幕上显示测量结果。(2)为了扩大检测范围,实现微弱光信号的检查,给探测器增加了自动切换档位的功能。基于此思想,在第二章中设计了一款可以检测低频微弱信号的便携式光功率计,它可应用于光纤传感网络的功率测试。2、在光纤传感系统中,使用的更多的是动态信号的检测,针对动态信号的特点,同样是结合智能手机作为接收终端,设计了一款可以探测动态信号的光电探测器,设计过程包括:原理分析、仿真确定相关参数、原理图和印制板图设计、光电探测器与手机的接口电路的设计和焊接调试。在深入分析光缆识别系统相关原理后,使用该探测器,结合手机制作了一个便携式的光缆识别仪,可用于光缆中不同线路的排查工作。此外,还深入分析了相干检测的原理及相关系统结构,研究如何进一步提升探测的灵敏度。"
print(HanLP.extractKeyword(document, 40))
# 自动摘要
print(HanLP.extractSummary(document, 3))