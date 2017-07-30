# coding:utf-8

import sys
default_encoding = "utf-8"
if(default_encoding != sys.getdefaultencoding()):
    reload(sys)
    sys.setdefaultencoding(default_encoding)


def text():
    f1 = open('total_pos.txt', 'r', encoding='utf-8')
    f2 = open('total_neg.txt', 'r', encoding='utf-8')
    line1 = f1.readline()
    line2 = f2.readline()
    str = ''
    while line1:
        str += line1
        line1 = f1.readline()
    while line2:
        str += line2
        line2 = f2.readline()
    f1.close()
    f2.close()
    #print (str)
    return str

# 把单个字作为特征

def bag_of_words(words):
    return dict([(word, True) for word in words])
text="我是一名北京理工大学的学生是一名是一名是一名是一名"
print(bag_of_words(text))



#把双个字作为特征--使用卡方统计的方法，选择排名前1000的双字
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
def bigram(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)  # 把文本变成双字搭配的形式
    bigrams = bigram_finder.nbest(score_fn, n)  # 使用卡方统计的方法，选择排名前1000的双字
    newBigrams = [u + v for (u, v) in bigrams]
    return bag_of_words(newBigrams)
print(bigram(text,score_fn=BigramAssocMeasures.chi_sq,n=3))



# 把单个词和双个词一起作为特征

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    newBigrams = [u + v for (u, v) in bigrams]
    a = bag_of_words(words)
    b = bag_of_words(newBigrams)
    a.update(b)  # 把字典b合并到字典a中
    return a  # 所有单个字和双个字一起作为特征
print(bigram_words(text,score_fn=BigramAssocMeasures.chi_sq,n=1000))


import jieba


def read_file(filename):
    stop = [line.strip() for line in open('stop.txt', 'r', encoding='utf-8').readlines()]  # 停用词
    f = open(filename, 'r')
    line = f.readline()
    #line.decode('utf-8').encode('gbk', 'ignore')
    str = []
    while line:
        s = line.split('\t')
        fenci = jieba.cut(s[0], cut_all=False)  # False默认值：精准模式
        str.append(list(set(fenci) - set(stop)))
        line = f.readline()
    return str


from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures

# 获取信息量最高(前number个)的特征(卡方统计)


def jieba_feature(number):
    posWords = []
    negWords = []
    for items in read_file('total_pos.txt'):  # 把集合的集合变成集合
        for item in items:
            posWords.append(item)
    # print posWords
    for items in read_file('total_neg.txt'):
        for item in items:
            negWords.append(item)
    # print negWords
    word_fd = FreqDist()  # 可统计所有词的词频
    cond_word_fd = ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词频
    for word in posWords:
        # print word
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in negWords:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1
    pos_word_count = cond_word_fd['pos'].N()  # 积极词的数量
    neg_word_count = cond_word_fd['neg'].N()  # 消极词的数量
    total_word_count = pos_word_count + neg_word_count
    word_scores = {}  # 包括了每个词和这个词的信息量
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count),
                                               total_word_count)  # 计算积极词的卡方统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count),
                                               total_word_count)  # 计算消极词的卡方统计量
        word_scores[word] = pos_score + neg_score  # 一个词的信息量等于积极卡方统计量加上消极卡方统计量
    best_vals = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)[
                :number]  # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_words = set([w for w, s in best_vals])
    return dict([(word, True) for word in best_words])


def build_features():
    # 四种特征选取方式，越来越好
    # feature = bag_of_words(text())#单个词
    # feature = bigram(text(),score_fn=BigramAssocMeasures.chi_sq,n=500)#双个词
    # feature =  bigram_words(text(),score_fn=BigramAssocMeasures.chi_sq,n=500)#单个词和双个词
    feature = jieba_feature(300)  # 结巴分词

    # print feature
    posFeatures = []
    for items in read_file('total_pos.txt'):
        a = {}
        for item in items:
            if item in feature.keys():
                a[item] = 'True'
        posWords = [a, 'pos']  # 为积极文本赋予"pos"
        posFeatures.append(posWords)

    negFeatures = []

    for items in read_file('total_neg.txt'):
        a = {}
        for item in items:
            if item in feature.keys():
                a[item] = 'True'
        negWords = [a, 'neg']  # 为消极文本赋予"neg"
        negFeatures.append(negWords)
    return posFeatures, negFeatures


posFeatures, negFeatures = build_features()  # 获得训练数据
# print negFeatures

from random import shuffle
shuffle(posFeatures)  # 把文本的排列随机化
shuffle(negFeatures)  # 把文本的排列随机化

train = posFeatures[3000:] + negFeatures[3000:]  # 训练集(80%)
test = posFeatures[:3000] + negFeatures[:3000]  # 预测集(验证集)(20%)
data, tag = zip(*test)  # 分离测试集合的数据和标签，便于验证和测试


def score(classifier):
    classifier = SklearnClassifier(classifier)  # 在nltk中使用scikit-learn的接口
    classifier.train(train)  # 训练分类器
    # print train
    pred = classifier.classify_many(data)  # 对测试集的数据进行分类，给出预测的标签
    n = 0
    s = len(pred)
    for i in range(0, s):
        if pred[i] == tag[i]:
            n = n + 1
    return float(n) / float(s)  # 对比分类预测结果和人工标注的正确结果，给出分类器准确度


import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

classifier = nltk.NaiveBayesClassifier.train(train)
print('nltk NaiveBayes accuracy is %f' %nltk.classify.accuracy(classifier, test))
print('BernoulliNB`s accuracy is %f' % score(BernoulliNB()))
print('MultinomiaNB`s accuracy is %f' % score(MultinomialNB()))
print('LogisticRegression`s accuracy is  %f' % score(LogisticRegression()))
print('SVC`s accuracy is %f' % score(SVC()))
print('LinearSVC`s accuracy is %f' % score(LinearSVC()))
print('NuSVC`s accuracy is %f' % score(NuSVC()))


