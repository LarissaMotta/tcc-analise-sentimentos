import src.core.process_data_w2v as process_file
import src.models.LSTM_model_w2v as LSTM_model
import src.utils.common_util as util
import src.utils.imports_util as imports
import src.utils.graphic_util as graphic

import numpy as np
import torch
from torch import nn
from torch import optim


def training(df, df2, df3):
    # Params variaveis
    p_n_hidden = 24
    p_n_layers = 3
    # p_optimizer = 'Adam'
    p_optimizer = 'RMSProp'
    p_loss = 'MSE'
    # p_loss = 'BCE'
    # p_loss = 'L1'
    p_lr = 0.005
    p_drop = 0.2
    # func_activation = 'sigmoid'
    func_activation = 'relu'
    p_n_epochs = 7 #1250
    p_batch_size = 20
    p_clip = 5
    # padding = 'same',
    # activation = 'relu'

    # pegando dados pre-processados
    matrix_embedding, df, df2, df3, seq_length = process_file.process_data_w2v(df, df2, df3)

    # definindo o lote de dados
    batch_size = p_batch_size

    # dados de entrada para a rede
    train_loader = util.get_input_loader(df, batch_size)
    valid_loader = util.get_input_loader(df2, batch_size)

    # definindo o tensor defaut, usando CPU ou não
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("GPU")
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        print("CPU")

    # instanciando o modelo e os hiperparametros
    n_vocab = len(matrix_embedding) + 1
    n_embed = 300
    n_hidden = p_n_hidden
    n_output = 1  # 1 ("positive") or 0 ("negative")
    n_layers = p_n_layers

    net = LSTM_model.SentimentLSTM(n_vocab, n_embed, n_hidden, n_output, n_layers, matrix_embedding, p_drop)

    # definindo loss e otimizador
    if p_loss == 'MSE':
        criterion = nn.MSELoss()
    elif p_loss == 'L1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.BCELoss()


    if p_optimizer == 'RMSProp':
        optimizer = optim.RMSprop(net.parameters(), lr=p_lr) #, momentum=0.6)
    else:
        optimizer = optim.Adam(net.parameters(), lr=p_lr)

    # inicialização de variáveis
    print_every = 200
    step = 0
    n_epochs = p_n_epochs  # validation loss increases from ~ epoch 3 or 4
    clip = p_clip  # for gradient clip to prevent exploding gradient problem in LSTM/RNN

    # Acuracia treinamento
    num_correct_train = 0

    # Listas para os graficos
    list_loss_train, list_loss_validation = [], []
    list_acc_train, list_acc_validation = [], []
    list_eixo_x = []

    net.train()
    for epoch in range(n_epochs):
        h = net.init_hidden(batch_size)

        for inputs, labels in train_loader:
            step += 1
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # making requires_grad = False for the latest set of h
            h = tuple([each.data for each in h])

            net.zero_grad()
            output, h = net(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            preds = torch.round(output.squeeze())
            correct_tensor = preds.eq(labels.float().view_as(preds))
            if torch.cuda.is_available():
                correct = np.squeeze(correct_tensor.cuda())
            else:
                correct = np.squeeze(correct_tensor)
            num_correct_train += correct.sum()

            if (step % print_every) == 0:
                ######################
                ##### VALIDATION #####
                ######################
                net.eval()
                valid_losses = []
                v_h = net.init_hidden(batch_size)
                # Accuracy
                v_num_correct = 0

                for v_inputs, v_labels in valid_loader:
                    if torch.cuda.is_available():
                        v_inputs, v_labels = inputs.cuda(), labels.cuda()

                    v_h = tuple([each.data for each in v_h])

                    v_output, v_h = net(v_inputs, v_h)
                    v_loss = criterion(v_output.squeeze(), v_labels.float())
                    valid_losses.append(v_loss.item())

                    v_preds = torch.round(v_output.squeeze())
                    v_correct_tensor = v_preds.eq(v_labels.float().view_as(v_preds))
                    if torch.cuda.is_available():
                        v_num_correct += np.squeeze(v_correct_tensor.cuda()).sum()
                    else:
                        v_num_correct += np.squeeze(v_correct_tensor).sum()

                # setando variaveis
                acc_train = num_correct_train / (step * batch_size)
                acc_validate = v_num_correct / len(valid_loader.dataset)

                # preenchendo a lista para plotar grafico
                list_loss_train.append(loss.item())
                list_loss_validation.append(np.mean(valid_losses))
                list_acc_train.append(acc_train)
                list_acc_validation.append(acc_validate)
                list_eixo_x.append("{}".format(step))

                print("Epoch: {}/{}".format((epoch + 1), n_epochs),
                      "Step: {}".format(step),
                      "Training Loss: {:.4f}".format(loss.item()),
                      "Training Accuracy: {:.4f}".format(acc_train),
                      "Validation Loss: {:.4f}".format(np.mean(valid_losses)),
                      "Validation Accuracy: {:.4f}".format(acc_validate))
                net.train()

    print("\n ------------------------------------------------------------- \n")

    loss_medio_val = "{:.6f}".format(sum(list_loss_validation) / len(list_loss_validation))
    acc_media_val = "{:.6f}".format(sum(list_acc_validation) / len(list_acc_validation))

    print("TREINAMENTO")
    print("Ultimo Loss: {:.6f}".format(list_loss_train[-1]),
          "Loss Máximo: {:.6f}".format(max(list_loss_train, key=float)),
          "Loss Médio: {:.6f}".format(sum(list_loss_train) / len(list_loss_train)))
    print("Ultima Acurácia: {:.6f}".format(list_acc_train[-1]),
          "Acurácia Máxima: {:.6f}".format(max(list_acc_train, key=float)),
          "Acurácia Média: {:.6f}".format(sum(list_acc_train) / len(list_acc_train)))

    print("VALIDAÇÃO")
    print("Ultimo Loss: {:.6f}".format(list_loss_validation[-1]),
          "Loss Máximo: {:.6f}".format(max(list_loss_validation, key=float)),
          "Loss Médio: ", loss_medio_val)
    print("Ultima Acurácia: {:.6f}".format(list_acc_validation[-1]),
          "Acurácia Máxima: {:.6f}".format(max(list_acc_validation, key=float)),
          "Acurácia Média: ", acc_media_val)

    # plotando os graficos
    ## All loss
    graphic.create_plot_multiple(list_eixo_x, list_loss_train, list_loss_validation, "Loss", "Epocas", "Loss Train",
                                 "Loss Val", imports.GRAPHIC_LOSS_PATH)
    ## All Acc
    graphic.create_plot_multiple(list_eixo_x, list_acc_train, list_acc_validation, "Acurácia", "Epocas", "Acc Train",
                                 "Acc Val", imports.GRAPHIC_ACC_PATH)

    csv_list = [p_n_hidden, p_n_layers, p_optimizer, p_loss, p_lr, p_drop, func_activation, p_n_epochs,
                p_batch_size,seq_length,p_clip, loss_medio_val, acc_media_val]

    return net, df3, batch_size, criterion, csv_list


def training_again(df, df2, df3, mt_embedding, seq_length):
    # Params variaveis
    p_n_hidden = 24
    p_n_layers = 3
    # p_optimizer = 'Adam'
    p_optimizer = 'RMSProp'
    p_loss = 'MSE'
    # p_loss = 'BCE'
    # p_loss = 'L1'
    p_lr = 0.005
    p_drop = 0.2
    # func_activation = 'sigmoid'
    func_activation = 'relu'
    p_n_epochs = 7 #1250
    p_batch_size = 20
    p_clip = 5
    # padding = 'same',
    # activation = 'relu'

    # pegando dados pre-processados
    matrix_embedding, df, df2, df3, seq_length = process_file.process_data_w2v(df, df2, df3)

    # definindo o lote de dados
    batch_size = p_batch_size

    # dados de entrada para a rede
    train_loader = util.get_input_loader(df, batch_size)
    valid_loader = util.get_input_loader(df2, batch_size)

    # definindo o tensor defaut, usando CPU ou não
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("GPU")
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        print("CPU")

    # instanciando o modelo e os hiperparametros
    n_vocab = len(matrix_embedding) + 1
    n_embed = 300
    n_hidden = p_n_hidden
    n_output = 1  # 1 ("positive") or 0 ("negative")
    n_layers = p_n_layers

    net = LSTM_model.SentimentLSTM(n_vocab, n_embed, n_hidden, n_output, n_layers, matrix_embedding, p_drop)

    # definindo loss e otimizador
    if p_loss == 'MSE':
        criterion = nn.MSELoss()
    elif p_loss == 'L1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.BCELoss()


    if p_optimizer == 'RMSProp':
        optimizer = optim.RMSprop(net.parameters(), lr=p_lr) #, momentum=0.6)
    else:
        optimizer = optim.Adam(net.parameters(), lr=p_lr)

    # inicialização de variáveis
    print_every = 200
    step = 0
    n_epochs = p_n_epochs  # validation loss increases from ~ epoch 3 or 4
    clip = p_clip  # for gradient clip to prevent exploding gradient problem in LSTM/RNN

    # Acuracia treinamento
    num_correct_train = 0

    # Listas para os graficos
    list_loss_train, list_loss_validation = [], []
    list_acc_train, list_acc_validation = [], []
    list_eixo_x = []

    net.train()
    for epoch in range(n_epochs):
        h = net.init_hidden(batch_size)

        for inputs, labels in train_loader:
            step += 1
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # making requires_grad = False for the latest set of h
            h = tuple([each.data for each in h])

            net.zero_grad()
            output, h = net(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            preds = torch.round(output.squeeze())
            correct_tensor = preds.eq(labels.float().view_as(preds))
            if torch.cuda.is_available():
                correct = np.squeeze(correct_tensor.cuda())
            else:
                correct = np.squeeze(correct_tensor)
            num_correct_train += correct.sum()

            if (step % print_every) == 0:
                ######################
                ##### VALIDATION #####
                ######################
                net.eval()
                valid_losses = []
                v_h = net.init_hidden(batch_size)
                # Accuracy
                v_num_correct = 0

                for v_inputs, v_labels in valid_loader:
                    if torch.cuda.is_available():
                        v_inputs, v_labels = inputs.cuda(), labels.cuda()

                    v_h = tuple([each.data for each in v_h])

                    v_output, v_h = net(v_inputs, v_h)
                    v_loss = criterion(v_output.squeeze(), v_labels.float())
                    valid_losses.append(v_loss.item())

                    v_preds = torch.round(v_output.squeeze())
                    v_correct_tensor = v_preds.eq(v_labels.float().view_as(v_preds))
                    if torch.cuda.is_available():
                        v_num_correct += np.squeeze(v_correct_tensor.cuda()).sum()
                    else:
                        v_num_correct += np.squeeze(v_correct_tensor).sum()

                # setando variaveis
                acc_train = num_correct_train / (step * batch_size)
                acc_validate = v_num_correct / len(valid_loader.dataset)

                # preenchendo a lista para plotar grafico
                list_loss_train.append(loss.item())
                list_loss_validation.append(np.mean(valid_losses))
                list_acc_train.append(acc_train)
                list_acc_validation.append(acc_validate)
                list_eixo_x.append("{}".format(step))

                print("Epoch: {}/{}".format((epoch + 1), n_epochs),
                      "Step: {}".format(step),
                      "Training Loss: {:.4f}".format(loss.item()),
                      "Training Accuracy: {:.4f}".format(acc_train),
                      "Validation Loss: {:.4f}".format(np.mean(valid_losses)),
                      "Validation Accuracy: {:.4f}".format(acc_validate))
                net.train()

    print("\n ------------------------------------------------------------- \n")

    loss_medio_val = "{:.6f}".format(sum(list_loss_validation) / len(list_loss_validation))
    acc_media_val = "{:.6f}".format(sum(list_acc_validation) / len(list_acc_validation))

    print("TREINAMENTO")
    print("Ultimo Loss: {:.6f}".format(list_loss_train[-1]),
          "Loss Máximo: {:.6f}".format(max(list_loss_train, key=float)),
          "Loss Médio: {:.6f}".format(sum(list_loss_train) / len(list_loss_train)))
    print("Ultima Acurácia: {:.6f}".format(list_acc_train[-1]),
          "Acurácia Máxima: {:.6f}".format(max(list_acc_train, key=float)),
          "Acurácia Média: {:.6f}".format(sum(list_acc_train) / len(list_acc_train)))

    print("VALIDAÇÃO")
    print("Ultimo Loss: {:.6f}".format(list_loss_validation[-1]),
          "Loss Máximo: {:.6f}".format(max(list_loss_validation, key=float)),
          "Loss Médio: ", loss_medio_val)
    print("Ultima Acurácia: {:.6f}".format(list_acc_validation[-1]),
          "Acurácia Máxima: {:.6f}".format(max(list_acc_validation, key=float)),
          "Acurácia Média: ", acc_media_val)

    # plotando os graficos
    ## All loss
    graphic.create_plot_multiple(list_eixo_x, list_loss_train, list_loss_validation, "Loss", "Epocas", "Loss Train",
                                 "Loss Val", imports.GRAPHIC_LOSS_PATH)
    ## All Acc
    graphic.create_plot_multiple(list_eixo_x, list_acc_train, list_acc_validation, "Acurácia", "Epocas", "Acc Train",
                                 "Acc Val", imports.GRAPHIC_ACC_PATH)

    csv_list = [p_n_hidden, p_n_layers, p_optimizer, p_loss, p_lr, p_drop, func_activation, p_n_epochs,
                p_batch_size,seq_length,p_clip, loss_medio_val, acc_media_val]

    return net, df3, batch_size, criterion, csv_list
