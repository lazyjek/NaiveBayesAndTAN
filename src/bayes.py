""" core class of Naive Bayes """
"""
Author: Jennifer Cao
Email: jennifer.cao@wisc.edu
"""
import numpy as np
class Bayes(object):
    def __init__(self, attributes, labels, traindata):
        """
            initialize parameters for naive bayes
            @param attrNames: names of attributes
            @param attrVals: all possible values of attributes
            @param labels: all possible labels (in this case, it is only a binary
                            classification task, so there are only two labels)
            @param x: training data
            @negative and positive: probability of negative and positive instances in training data
        """
        self.attrNames =  [item[0] for item in attributes]
        self.attrVals = [item[1] for item in attributes]
        self.labels = labels
        self.x = np.array(traindata, dtype = np.str)
        self.negative, self.positive = self.clsProbability()

    def smoothing(self, x, m, k):
        """
            laplacing smoothing function for x / m (pseudocounts of 1)
            x / m -> (x + 1) / (m + k)
            @param x: count of a specific class
            @param m: count of data
            @param k: count of classes of data
            @rparam: <float> probability of this class after laplacing smoothing
        """
        pseudocounts = 1
        return float(x + pseudocounts) / (m + k * pseudocounts)

    def clsProbability(self):
        """
            @rparam class probability: represents probabilities for negative and positibe datas respectively
        """
        m = self.x.shape[0]
        k = len(self.labels)
        negative = np.where(self.x[:,-1] == self.labels[0], 1, 0).sum()
        positive = np.where(self.x[:,-1] == self.labels[1], 1, 0).sum()
        return self.smoothing(negative, m, k), self.smoothing(positive, m, k)

    def probability(self, label, attrInd, attrVal):
        """
            @param label: actual class of x
            @param attrInd: feature index
            @param attVal: feature value
            @rparam prbability for P(attr | label)
        """
        attr = np.where((self.x[:,attrInd] == attrVal) & (self.x[:,-1] == label), 1, 0).sum()
        m = np.where(self.x[:,-1] == label, 1, 0).sum()
        k = len(self.attrVals[attrInd])
        return self.smoothing(attr, m, k)

    def classify(self, test):
        """
            @param test: a single instance of test data (attr1 attr2 ... attrN class)
            @rparam: predict class | actual class | posterior probability
        """
        negative, positive = self.negative, self.positive
        label = test[-1]
        x = test[:-1]
        for i, val in enumerate(x):
            negative *= self.probability(self.labels[0], i, val)
            positive *= self.probability(self.labels[1], i, val)
        predictClass, actualClass, posteriorProb = self.labels[0], label, negative / (negative + positive)
        if negative < positive:
            predictClass, posteriorProb = self.labels[1], positive / (negative + positive)
        return predictClass, actualClass, posteriorProb

    def printTree(self):
        """
            output format: variable name | 'class'
        """
        for name in self.attrNames:
            print '{} class'.format(name)

if __name__ == '__main__':
    from tool import naiveBayesOutput
    naiveBayesOutput('../lymph_train.arff', '../lymph_test.arff')
    naiveBayesOutput('../vote_train.arff', '../vote_test.arff')
