import numpy as np
def getRecall(o):
    return o['tp'] / (o['tp'] + o['fn'])

def getPrecision(o):
    return o['tp'] / (o['tp'] + o['fp'])

def getCorrectPercent(o):
    return (o['tp'] + o['tn'])/ (o['tp'] + o['tn'] + o['fp'] + o['fn'])

def getSpecificity(o):
    #TN/(FP + TN)
    return o['tn']/ (o['tn'] + o['fp'])

def getF1(prec, recall):
    return 2 * prec * recall / (prec + recall)

def getF2(prec, recall):
    return (1 + pow(2, 2)) * prec * recall / (pow(2, 2) * prec + recall)

def printMetrics(indx2metrics):
    for k in indx2metrics:
        o = indx2metrics[k]
        # correct%
        correctPercent = getCorrectPercent(o)
        print('correct%', k, correctPercent)
        # recall
        recall = getRecall(o)
        print('recall', k, recall)
        # specificity
        specificity = getSpecificity(o)
        print('specificity', k, specificity)
        # precision 
        precision = getPrecision(o)
        print('precision', k, precision)
        # f1 
        f1 = getF1(precision, recall)
        print('f1', k, f1)
        # f2
        f2 = getF2(precision, recall)
        print('f2', k, f2)
        print('')

def getIndx2Metrics(cm):
    indx2metrics = {}
    for i, predicts in enumerate(cm):
        indx2metrics[i] = {}
        indx2metrics[i]['tp'] = predicts[i]
        indx2metrics[i]['fn'] = sum(predicts) - predicts[i]
        indx2metrics[i]['fp'] = sum(cm[:, i]) - predicts[i]
        indx2metrics[i]['tn'] = sum(sum(cm)) - (indx2metrics[i]['tp'] + indx2metrics[i]['fn'] + indx2metrics[i]['fp'])
    return indx2metrics
'''
Task 1
Confusion matrix:
[[623   0   0   0]
 [620   0   0   0]
 [620   0   0   0]
 [624   0   0   0]]
Training time:           1341.3634989261627
Prediction time:         2.340301513671875
'''
cm = np.array([
    [623,   0,   0,   0],
    [620,   0,   0,   0],
    [620,   0,   0,   0],
    [624,   0,   0,   0],
])

print('task 1')
indx2metrics = getIndx2Metrics(cm)
printMetrics(indx2metrics)

'''
Task 2
[[250  57  50 266]
 [ 93 277  71 179]
 [ 60  19 421 120]
 [206  47 107 264]]
Training time:           1430.3466546535492
Prediction time:         2.4221107959747314
'''
cm = np.array([
    [250,  57,  50, 266],
    [ 93, 277,  71, 179],
    [ 60,  19, 421, 120],
    [206,  47, 107, 264],
])

print('task 2')
indx2metrics = getIndx2Metrics(cm)
printMetrics(indx2metrics)


'''
Task 3
[[177  94  47 305]
 [ 48 320  56 196]
 [ 42  79 364 135]
 [167 140  45 272]]
Training time:           1407.9878554344177
Prediction time:         2.4469852447509766
'''
cm = np.array([
 [177,  94,  47, 305],
 [ 48, 320,  56, 196],
 [ 42,  79, 364, 135],
 [167, 140,  45, 272],
])

print('task 3')
indx2metrics = getIndx2Metrics(cm)
printMetrics(indx2metrics)
