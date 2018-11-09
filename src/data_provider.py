#-*- coding: utf-8 -*-
import arff

def data_provider(filename):
    """ load data from arff files.
    @param filename[string]: input filename directory. string. example: 'sonar.arff'
    @rparam attrNum[integer]: number of attributes.
    @rparam labels[list]: all possible outputs of labels. for binary classification, there's only two items.
    @rparam instances[[list]]: instances, [[x1, x2, x3 .. xn label], .. ]
    """
    data = arff.load(open(filename, 'rb'))
    attributes = data['attributes'][:-1]
    labels = data['attributes'][-1][1]
    instances = data['data']
    return attributes, labels, instances

if __name__ == '__main__':
    import sys
    datas = data_provider(sys.argv[1])
