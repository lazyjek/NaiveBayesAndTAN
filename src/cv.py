"""implementation of cross validation"""
"""
Author: Jennifer Cao
Email: jennifer.cao@wisc.edu
"""
import random
random.seed(0)
from data_provider import data_provider

"""an implementation of cross validation
"""
class CrossValidate(object):
    def __init__(self, nfolds, instances, labels):
        """ labels refers to all the possible classifications of all the instances """
        self.nfolds = nfolds
        self.instances = instances
        self.labels = labels
        self.mergefolds = self.stratify()

    def group(self):
        """ divide instances into multiple groups, the number of group is the same as the number of classes """
        instances = self.instances
        groups = []
        for i in range(len(self.labels)):
            groups.append([instance for instance in instances if instance[-1] == self.labels[i]])
        return groups

    def sample(self, instances):
        """ randomly sample instances into kfolds """
        kfolds = [[] for i in range(self.nfolds)]
        grids = [(count, count + 1) for count in range(self.nfolds)]
        for instance in instances:
            rNum = random.uniform(0, self.nfolds)
            for i in range(len(grids)):
                grid = grids[i]
                if grid[0] <= rNum < grid[1]:
                    kfolds[i].append(instance)
        return kfolds

    def stratify(self):
        """ stratify and make final k folds data. """
        groups = self.group()
        folds = []
        for group in groups:
            folds.append(self.sample(group))
        return [sum([folds[j][i] for j in range(len(folds))], []) \
                for i in range(self.nfolds)]

    def fold(self, fold, shuffle = True):
        """ get n fold x, y
        @ param fold: the number of fold that you choose,
                     should be exactly between 1 ~ nfold.
        @ shuffle: if True, shuffle the training and testing sets.
        @ rparam: train and test sets
        """
        testData = self.mergefolds[fold]
        trainData = sum(self.mergefolds[:fold] +
                self.mergefolds[fold + 1:], [])
        if shuffle:
            random.shuffle(testData)
            random.shuffle(trainData)
        return trainData, testData

"""below is unit test part"""
def utStratify():
    attrNum, labels, instances = data_provider('../test.arff')
    cv = CrossValidate(6, instances, labels)
    print cv.mergefolds
    print len(cv.mergefolds)

def utFold():
    attrNum, labels, instances = data_provider('../sonar.arff')
    cv = CrossValidate(10, instances, labels)
    for i in range(10):
        print '>>>>>>>>>>>>>>>>>>> Fold',i,'<<<<<<<<<<<<<<<<<<<<<<'
        train, test = cv.fold(i)
        #print train[1], test[1]
        print train[1].shape[0] + test[1].shape[0]
        print train[0].shape, test[0].shape

if __name__ == '__main__':
    utFold()
#    utStratify()

