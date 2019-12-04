# members: pc38, bz30
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

    # For excel paste : )
    print('# For excel paste : )')
    for k in indx2metrics:
        o = indx2metrics[k]
        correctPercent = getCorrectPercent(o)
        recall = getRecall(o)
        specificity = getSpecificity(o)
        precision = getPrecision(o)
        f1 = getF1(precision, recall)
        f2 = getF2(precision, recall)
        print('{},{},{},{},{},{}'.format(correctPercent, recall, specificity, precision, f1, f2))

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
[[  0   0 623   0]
 [  0   0 620   0]
 [  0   0 620   0]
 [  0   0 624   0]]
Training time:           1413.9152915477753
Prediction time:         1.7894158363342285
'''
cm = np.array([
    [0, 0, 623, 0],
    [0, 0, 620, 0],
    [0, 0, 620, 0],
    [0, 0, 624, 0],
])

print('task 1')
indx2metrics = getIndx2Metrics(cm)
printMetrics(indx2metrics)

'''
Task 2
Confusion matrix:
 [[237 69 51 266]
 [126 292 39 163]
 [70 19 454 77]
 [227 75 52 270]]
Training time:           1611.98002409935
Prediction time:         2.4913928508758545
'''

cm = np.array([
    [237, 69, 51, 266],
    [126, 292, 39, 163],
    [70, 19, 454,  77],
    [227, 75, 52, 270],
])
print('task 2')
indx2metrics = getIndx2Metrics(cm)
printMetrics(indx2metrics)


'''
Task 3
Confusion matrix:
[[233  49  32 309]
 [ 95 312  26 187]
 [ 58  11 437 114]
 [229  65  56 274]]
Training time:           1597.7762842178345
Prediction time:         2.661710500717163
'''
cm = np.array([
 [233,  49,  32, 309],
 [ 95, 312,  26, 187],
 [ 58,  11, 437, 114],
 [229,  65,  56, 274],
])

print('task 3')
indx2metrics = getIndx2Metrics(cm)
printMetrics(indx2metrics)


'''
Task 4
Confusion matrix:
[[336  12  31 244]
 [ 27 524  45  24]
 [  3   0 573  44]
 [ 57  21  56 490]]
Training time:           2230.9548387527466
Prediction time:         2.569983959197998
'''
cm = np.array([
    [336, 12, 31, 244],
    [27, 524, 45, 24],
    [3, 0, 573, 44],
    [57, 21, 56, 490],
])
print('task 4')
indx2metrics = getIndx2Metrics(cm)
printMetrics(indx2metrics)
