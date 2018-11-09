from data_provider import data_provider
from cv import CrossValidate
from bayes import Bayes
from tan import TAN

def nfoldTest(nfold = 10):
    attributes, labels, instances = data_provider('../chess-KingRookVKingPawn.arff')
    cv = CrossValidate(nfold, instances, labels)
    accuracy = [{'bayes':0, 'tan':0} for _ in range(nfold)]
    models = {'bayes':Bayes, 'tan':TAN}
    for i in range(nfold):
        train, test = cv.fold(i)
        iTotal = len(test)
        for key in models:
            model = models[key](attributes, labels, train)
            for instance in test:
                result = model.classify(instance)
                print result[0], result[1]
                if result[0] == result[1]:
                    accuracy[i][key] += 1
        for key in accuracy[i]:
            accuracy[i][key] = float(accuracy[i][key]) / iTotal
    print accuracy

    fileout = open('output.txt', 'w+')
    for i in range(len(accuracy)):
        fileout.write('fold{} bayes:{:.16f} tan:{:.16f}\n'.format(i, accuracy[i]['bayes'], accuracy[i]['tan']))
    fileout.close()

if __name__ == '__main__':
    nfoldTest()
