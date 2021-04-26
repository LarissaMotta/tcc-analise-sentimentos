import src.utils.common_util as util
import src.utils.graphic_util as graphic_util
import src.utils.imports_util as imports
import src.utils.file_util as file_util

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


# def testing(net, test_loader, batch_size, criterion):
def testing(net, df3, batch_size, criterion, csv_list):
    # dados de entrada para a rede
    test_loader = util.get_input_loader(df3, batch_size)

    # inicializando variaveis de teste
    test_losses, list_pred_test, list_label_test = [], [], []
    num_correct = 0

    h = net.init_hidden(batch_size)
    net.eval()

    for inputs, labels in test_loader:
        # step_t += 1

        h = tuple([each.data for each in h])
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        test_output, h = net(inputs, h)

        loss = criterion(test_output.squeeze(), labels.float())
        test_losses.append(loss.item())

        preds = torch.round(test_output.squeeze())
        # acuracia
        correct_tensor = preds.eq(labels.float().view_as(preds))
        if torch.cuda.is_available():
            correct = np.squeeze(correct_tensor.cuda())
        else:
            correct = np.squeeze(correct_tensor)
        num_correct += sum(correct)

        list_pred_test.append(preds.tolist()[0])
        list_label_test.append(labels.float().view_as(preds).tolist()[0])

    loss_teste = "{:.4f}".format(np.mean(test_losses))
    acc_teste = "{:.2f}".format(num_correct / len(test_loader.dataset))

    print("Test Loss: ", loss_teste)
    print("Test Accuracy: ", acc_teste)

    print("label como view")
    cd_mt = confusion_matrix(list_label_test, list_pred_test)
    cd_mt[0][0] = cd_mt[0][0] * batch_size
    cd_mt[1][0] = cd_mt[1][0] * batch_size
    cd_mt[1][1] = cd_mt[1][1] * batch_size
    cd_mt[0][1] = cd_mt[0][1] * batch_size
    print(cd_mt)
    # plotando matrix de confusao
    graphic_util.plot_confusion_matrix(cd_mt, imports.CONFUSION_MATRIX_PATH)

    TP = cd_mt[0][0]
    FP = cd_mt[1][0]
    TN = cd_mt[1][1]
    FN = cd_mt[0][1]

    metrics = precision_recall_fscore_support(list_label_test, list_pred_test)

    prec = metrics[0][0]
    rec = metrics[1][0]
    f1s = metrics[2][0]

    acc1 = accuracy_score(list_label_test, list_pred_test)  # mesmo que o medido
    acc2 = accuracy_score(list_label_test, list_pred_test, normalize=False)

    precision = __divisao(TP, (TP + FP))
    acc = __divisao((TP + TN), (TP + FP + FN + TN))
    recall = __divisao(TP, (TP + FN))
    IoU = __divisao(TP, (TP + FP + FN))
    f1score = __divisao((2 * precision * recall), (precision + recall))


    # terminando de adicionar as informacoes e salvando no arquivo
    csv_list.append(loss_teste)
    csv_list.append(acc_teste)
    csv_list.append(acc)
    csv_list.append(imports.POSITIVE_TWEETS_PATH)
    csv_list.append(imports.NEGATIVE_TWEETS_PATH)
    file_util.save_infos_in_csv(csv_list)

    # salvando as metricas
    # colocando valores num dicionario e salvando num csv
    dic = {
        'Acuracia_cod': acc_teste,
        'Acuracia_mc': acc,
        'precision': prec,
        'recall': rec,
        'f1score': f1s,
        'IoU': IoU
    }
    file_util.save_dict(dic, imports.METRICS_PATH)

    print("precision: ", prec, " // ", precision)
    print("recall: ", rec, " // ", recall)
    print("f1score: ", f1s, " // ", f1score)
    print("acc: ", acc)
    print("acc Normalizada: ", acc2)
    print("Iou: ", IoU)

    return 0

def __divisao(dividendo, divisor):
    if divisor == np.nan or divisor == 0:
        return 0
    return dividendo/divisor
