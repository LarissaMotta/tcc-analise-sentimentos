import src.utils.common_util as util

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


# def testing(net, test_loader, batch_size, criterion):
def testing(net, df3, batch_size, criterion):
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

    print("Test Loss: {:.4f}".format(np.mean(test_losses)))
    print("Test Accuracy: {:.2f}".format(num_correct / len(test_loader.dataset)))

    print("label como view")
    cd_mt = confusion_matrix(list_label_test, list_pred_test)
    print(cd_mt)

    # TODO Plotar matrix de confusao

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

    precision = TP / (TP + FP)
    acc = (TP + TN) / (TP + FP + FN + TN)
    recall = TP / (TP + FN)
    IoU = TP / (TP + FP + FN)
    f1score = (2 * precision * recall) / (precision + recall)

    print("precision: ", prec, " // ", precision)
    print("recall: ", rec, " // ", recall)
    print("f1score: ", f1s, " // ", f1score)
    print("acc: ", acc)
    print("acc Normalizada: ", acc2)
    print("Iou: ", IoU)

    return 0

