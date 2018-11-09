""" format output """
"""
Author: Jennifer Cao
Email: jennifer.cao@wisc.edu
"""
from data_provider import data_provider
from bayes import Bayes
from tan import TAN

def formatOutput(result):
    predictClass, actualClass, posteriorProb = result
    return '{} {} {:.12f}'.format(predictClass, actualClass, posteriorProb)

def modelOutput(trainFile, testFile, modelType):
    """
    output is:
        (naive bayes) variable name | 'class'
        (tan) variable name | name of its parents
    # empty
    followed by:
        predict class | actual class | posterior probability (12 digits after decimal point)
    # empty
    followed by:
        The number of the test-set examples that were correctly classified.
    """

    attributes, labels, instances = data_provider(trainFile)
    if modelType == 'n':
        model = Bayes(attributes, labels, instances)
    elif modelType == 't':
        model = TAN(attributes, labels, instances)
    else:
        import sys
        print >> sys.stderr, 'model type should be [n] or [t] !!!'
        sys.exit()
    attributes, labels, instances = data_provider(testFile)

    # format output part1: attribute name | 'class'
    model.printTree()
    print

    correctClassCnt = 0
    for test in instances:
        result = model.classify(test)
        if result[0] == result[1]:
            correctClassCnt += 1
        # format output part2: predict class | actual class | posterior probability
        print formatOutput(result)
    print

    # format output part3: correctly classified number of test instances
    print correctClassCnt
