#coding=utf-8  
import random,nltk,sklearn
from nltk.corpus import names  
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def gender_features(word):  
    '''''提取每个单词的最后一个字母作为特征'''  
    return {'last_letter': word[-1]}  
# 先为原始数据打好标签  
labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])  
# 随机打乱打好标签的数据集的顺序，  
random.shuffle(labeled_names)  
# 从原始数据中提取特征（名字的最后一个字母， 参见gender_features的实现）  
featuresets = [(gender_features(name), gender) for (name, gender) in labeled_names]  
# 将特征集划分成训练集和测试集  
train_set, test_set = featuresets[500:], featuresets[:500]  
data, tag = zip(*test_set)
 

def score(classifier):
    classifier = SklearnClassifier(classifier)  # 在nltk中使用scikit-learn的接口
    classifier.train(train_set)  # 训练分类器
    pred = classifier.classify_many(data)  # 对测试集的数据进行分类，给出预测的标签
    n = 0
    s = len(pred)
    for i in range(0, s):
        if pred[i] == tag[i]:
            n = n + 1
    return float(n) / float(s)  # 对比分类预测结果和人工标注的正确结果，给出分类器准确度

# 使用训练集训练模型（核心就是求出各种后验概率）  
classifier = nltk.NaiveBayesClassifier.train(train_set)  
# 通过测试集来估计分类器的准确性  
print('nltk NaiveBayes accuracy is %f' %nltk.classify.accuracy(classifier, test_set))
print('BernoulliNB`s accuracy is %f' % score(BernoulliNB()))
print('MultinomiaNB`s accuracy is %f' % score(MultinomialNB()))
print('LogisticRegression`s accuracy is  %f' % score(LogisticRegression()))
print('SVC`s accuracy is %f' % score(SVC()))
print('LinearSVC`s accuracy is %f' % score(LinearSVC()))
print('NuSVC`s accuracy is %f' % score(NuSVC()))
