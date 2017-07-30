# encoding: utf-8
import nltk
from nltk.tokenize.stanford_segmenter import StanfordSegmenter

segmenter=StanfordSegmenter(
    #分词依赖的jar包
    path_to_jar=r"/home/jiangix/document/stanford-segmenter/stanford-segmenter.jar",
    path_to_slf4j=r"/home/jiangix/document/slf4j/slf4j-api.jar",
    #分词数据文件夹
    path_to_sihan_corpora_dict=r"/home/jiangix/document/stanford-segmenter/data",
    #基于北大在2005backoof上提供的人名日报语料库
    path_to_model=r"/home/jiangix/document/stanford-segmenter/data/pku.gz",
    path_to_dict=r"/home/jiangix/document/stanford-segmenter/data/dict-chris6.ser.gz"
    )

segmenter.default_config('zh')
result=segmenter.segment(u'我喜欢学习编程')

print (result)