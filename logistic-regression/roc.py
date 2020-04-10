from sklearn import metrics
import matplotlib.pyplot as plt
import math

def drawGraph(f, x, y, nome_graph):
    plt.plot(x, y)
    f.savefig(nome_graph)

def getROC(pred, y_test, nome_graph):
    pred_yTest = []

    for i in range(len(pred)):
        pred_yTest.append((pred[i], y_test[i]))

    pred_yTest = sorted( pred_yTest, key=lambda x: x[0] )

    aux = []
    x_roc = []
    y_roc = []
    best_threshold = []
    best_dist = 1

    while(pred_yTest != []):
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        
        for i in range(len(aux)):
            if(aux[i][1] == 0):
                TN += 1
            else:
                FN += 1

        for i in range(len(pred_yTest)):
            if(pred_yTest[i][1] == 1):
                TP += 1
            else:
                FP += 1
        
        TPR = TP / (TP+FN)
        FPR = FP / (FP+TN)
        
        # procurar melhor threshold pela distância entre (FPR, TPR) e o ponto (0,1)
        new_dist = math.sqrt( (0 - FPR)**2 + (1-TPR)**2 ) 
        if(new_dist < best_dist):
            best_dist = new_dist
            best_threshold = [best_dist, FPR, TPR, TP, FN, FP, TN]

        # adicionar coordenadas (FPR, TPR) às listas para a curva ROC
        x_roc.append( FPR )
        y_roc.append( TPR )
        
        aux.append( pred_yTest[0] )
        pred_yTest.pop(0)


    print(best_threshold)


    # --- Desenhar Curva ROC --- #
    g = plt.figure()
    drawGraph(g, x_roc, y_roc, nome_graph)

    # --- Calcular AUC --- #
    AUC = metrics.auc(x_roc, y_roc)
    print("AUC: " + str(AUC))

    # --- Calcular Precision, Recall e Accurary --- #
    precision = best_threshold[3] / (best_threshold[3] + best_threshold[5])
    recall = best_threshold[3] / (best_threshold[3] + best_threshold[4])
    accuracy = (best_threshold[3] + best_threshold[4]) / (best_threshold[3] + best_threshold[4] + best_threshold[5] + best_threshold[6])
    print("Precision: " + str(precision) + "\nRecall: " + str(recall) + "\nAccuracy: " + str(accuracy))