import sys
from tool import modelOutput

if __name__ == '__main__':
    try:
        trainFileName = sys.argv[1] # train set
        testFileName = sys.argv[2]  # test set
        modelType = sys.argv[3]     # t | n (tan | naive bayes)
    except:
        print >> sys.stderr, '[ERROR] wrong input format! 3 inputs: [string] [string] [string]'
    modelOutput(trainFileName, testFileName, modelType)
