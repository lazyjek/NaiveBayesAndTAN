""" Tree-Augamented Naive Bayes """
"""
Author: Jennifer Cao
Email: jennifer.cao@wisc.edu
"""
import numpy as np
import math
from bayes import Bayes

class TAN(Bayes):
    def __init__(self, attributes, labels, traindata):
        Bayes.__init__(self, attributes, labels, traindata)
        self.edges = self.initializeTree()
        self.graph = self.findMaxSpanningTree()

    def mutuProbability(self, label, i, iVal, j, jVal, cross = False):
        """
            calculate mutul probability for cross probability (Xi, Xj)
            @param i, j: index for attributes i, j
            @param iVal, jVal: values for attributes i, j
            @param label: value for y
            @rparam: laplacian parameter k, probability for cross attributes(Xi, Xj)
        """
        attr = np.where((self.x[:,i] == iVal) & \
                (self.x[:,j] == jVal) & \
                (self.x[:,-1] == label), 1, 0).sum()
        k = len(self.attrVals[i])
        m = np.where((self.x[:,-1] == label) & (self.x[:,j] == jVal), 1, 0).sum()
        if cross:
            k *= len(self.attrVals[j])
            m = np.where(self.x[:,-1] == label, 1, 0).sum()
        return k, self.smoothing(attr, m, k)

    def mutulInformation(self, i, j):
        """
            calculate conditional mutul information between attributes (i, j)
            @param i: index of attribute i
            @param j: index of attribute j
            @rparam: mutul information (i, j) calculated from train data.
        """
        if i == j:
            return -1.0
        mi = 0.0
        total = self.x.shape[0]
        for label in self.labels:
            classTotal = np.where(self.x[:,-1] == label, 1, 0).sum()
            for iVal in self.attrVals[i]:
                iProbAttr = self.probability(label, i, iVal)
                for jVal in self.attrVals[j]:
                    jProbAttr = self.probability(label, j, jVal)
                    k, ijProbAttr = self.mutuProbability(label, i, iVal, j, jVal, True)
                    probAttr = ijProbAttr * (classTotal + k) / (total +
                            k * len(self.labels))
                    mi += probAttr * math.log(ijProbAttr/(iProbAttr*jProbAttr), 2)
        return mi

    def initializeTree(self):
        """
            initialize all directions graph, vertex is index of attributes, and edge is mi.
            @rparam: edges and empty graph
        """
        attrLen = len(self.attrNames)
        edges = [[0 for _ in range(attrLen)] for _ in range(attrLen)]
        for i in range(attrLen):
            for j in range(i, attrLen):
                edges[i][j] = edges[j][i] = self.mutulInformation(i, j)
        return edges

    def findMaxSpanningTree(self):
        """
            use prims algorithm to find maximum spanning tree.
            initialize:
                vertex: attribute. Initialize with attribute index 0. If there are ties in selecting maximum weight edges
            preference criteria:
                1. prefer edges emanating from variables listed earlier in the input file
                2. if there are multiple maximum weight edges emanating from the first such variable,
                   prefer edges going to variables listed earlier in the input file.
            pick the first variable in the input file as the root.
        """
        attrLen = len(self.attrNames)
        edges = []
        roots = set([0])
        while len(roots) <  attrLen:
            maxInform = -10 ** 31
            edge = [0, 1]
            for i in roots:
                for j in range(attrLen):
                    if j in roots:
                        continue
                    if self.edges[i][j] <= maxInform:
                        continue
                    maxInform = self.edges[i][j]
                    edge = [i, j]
            edges.append(edge)
            roots.add(edge[1])

        # graph = {parent: child}
        graph = {}
        for j, i in edges:
            graph[i] = j
        return graph

    def printTree(self):
        """
            output format: variable name | name of its parents | 'class'
        """
        print '{} class'.format(self.attrNames[0])
        pairs = sorted(self.graph.items(), key = lambda x:x[0])
        for i, j in pairs:
            print '{} {} class'.format(self.attrNames[i], self.attrNames[j])

    def classify(self, test):
        """
            @param test: a single instance of test data (attr1 attr2 ... attrN class)
            @rparam: predict class | actual class | posterior probability
        """
        positive, negative = self.positive, self.negative
        label = test[-1]
        x = test[:-1]
        for i, val in enumerate(x):
            if i == 0:
                negative *= self.probability(self.labels[0], i, val)
                positive *= self.probability(self.labels[1], i, val)
            else:
                negative *= self.mutuProbability(self.labels[0], i, val, self.graph[i], x[self.graph[i]])[1]
                positive *= self.mutuProbability(self.labels[1], i, val, self.graph[i], x[self.graph[i]])[1]
        predictClass, actualClass, posteriorProb = self.labels[0], label, negative / (negative + positive)
        if negative < positive:
            predictClass, posteriorProb = self.labels[1], positive / (negative + positive)
        return predictClass, actualClass, posteriorProb

def utMutulInformation(tan):
    """
        file: lymph_t_debug_output.txt
        Verbose output
        Conditional mutual information graph:
        (row,column) = w : The mutual information between rowth attribute and columnth attribute is w.
    """
    for items in tan.edges:
        print ' '.join(['{}'.format(item) for item in items])

def utPrims(tan):
    print tan.graph

def utPrintTree(tan):
    tan.printTree()

if __name__ == '__main__':
    from data_provider import data_provider
    attributes, labels, instances = data_provider('../lymph_train.arff')
    tan = TAN(attributes, labels, instances)
    utMutulInformation(tan)
    utPrims(tan)
    utPrintTree(tan)


