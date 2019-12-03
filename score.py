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
    [0,   0,  623,    0],
    [0,   0,  620,    0],
    [0,   0,  620,    0],
    [0,   0,  624,    0],
])

print('task 1')
indx2metrics = getIndx2Metrics(cm)
printMetrics(indx2metrics)

'''
Task 2
Confusion matrix:
[[126 163  82 252]
 [ 39 323  65 193]
 [ 23  59 457  81]
 [ 96 135 166 227]]
Training time:           1427.547043800354
Prediction time:         2.491168737411499
'''
cm = np.array([
 [126, 163,  82, 252],
 [ 39, 323,  65, 193],
 [ 23,  59, 457,  81],
 [ 96, 135, 166, 227],
])

print('task 2')
indx2metrics = getIndx2Metrics(cm)
printMetrics(indx2metrics)


'''
Task 3
Confusion matrix:
[[270  62  40 251]
 [130 197  15 278]
 [ 85   8 344 183]
 [242  51  29 302]]
Training time:           1597.7762842178345
Prediction time:         2.661710500717163
'''
cm = np.array([
 [270,  62,  40, 251],
 [130, 197,  15, 278],
 [ 85,   8, 344, 183],
 [242,  51,  29, 302],
])

print('task 3')
indx2metrics = getIndx2Metrics(cm)
printMetrics(indx2metrics)


cm = np.array([
    [166, 68, 186, 203],
    [114, 242, 149, 115],
    [168, 56, 211, 185],
    [190, 78, 188, 168],
])
print('task 4')
indx2metrics = getIndx2Metrics(cm)
printMetrics(indx2metrics)

cm = np.array([
    [267, 68, 39, 249],
    [170, 233, 57, 160],
    [57, 18, 434, 111],
    [237, 73,  68, 246],
])
print('task 5')
indx2metrics = getIndx2Metrics(cm)
printMetrics(indx2metrics)
