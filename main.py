import torch
import torch.nn as nn
import os

from data_preprocessing import get_vocab_size, get_word_2_index, get_params, get_files, get_files_linear, labels_distribution
from train_test import train_linear, test_linear, create_iterator, run_train, evaluate
from linear_model import Linear
from cnn_model import CNN
from lstm_model import LSTM
from mlp_model import MLP

if __name__ == "__main__":

    # placing the tensors on the GPU if one is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    path = 'C:/Users/Yaron/PycharmProjects/Sentiment_Analyzer/Word_Based'
    path_data = os.path.join(path, "data")

    # parameters
    model_type = "LSTM"
    data_type = "morph" # or: "token"

    char_based = True
    if char_based:
        tokenizer = lambda s: list(s) # char-based
    else:
        tokenizer = lambda s: s.split() # word-based


    # hyper-parameters:
    lr = 1e-4
    batch_size = 50
    dropout_keep_prob = 0.5
    embedding_size = 300
    max_document_length = 100  # each sentence has until 100 words
    dev_size = 0.8 # split percentage to train\validation data
    max_size = 5000 # maximum vocabulary size
    seed = 1
    num_classes = 3

    # dropout_keep_prob, embedding_size, batch_size, lr, dev_size, vocabulary_size, max_document_length, input_size, hidden_size, output_dim, n_filters, filter_sizes, num_epochs = get_params(model_type)
    train_data, valid_data, test_data, Text, Label = get_files(path_data, dev_size, max_document_length, seed, data_type, tokenizer)

    # Build_vocab : It will first create a dictionary mapping all the unique words present in the train_data to an
    # index and then after it will use word embedding (random, Glove etc.) to map the index to the corresponding word embedding.
    Text.build_vocab(train_data, max_size=max_size)
    Label.build_vocab(train_data)
    vocab_size = len(Text.vocab)

    train_iterator, valid_iterator, test_iterator = create_iterator(train_data, valid_data, test_data, batch_size, device)

    # loss function
    loss_func = nn.CrossEntropyLoss()

    if (model_type == "Linear"):
        # dropout_keep_prob, embedding_size, batch_size, lr, dev_size, vocabulary_size, max_document_length, input_size, hidden_size, output_dim, n_filters, filter_sizes, num_epochs = get_params(model_type)

        num_epochs = 10
        hidden_size = 100

        to_train = True

        linear_model = Linear(max_document_length, hidden_size, num_classes) # input size when there is no embedding layer is max_doc_size and with embedding is the vocabulary size

        # optimization algorithm
        optimizer = torch.optim.Adam(linear_model.parameters(), lr=lr)
        # train and evaluation
        if (to_train):
            # train and evaluation
            run_train(num_epochs, linear_model, train_iterator, valid_iterator, optimizer, loss_func, model_type)

        # load weights
        linear_model.load_state_dict(torch.load(os.path.join(path, "saved_weights_Linear.pt")))
        # predict
        test_loss, test_acc = evaluate(linear_model, test_iterator, loss_func)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    if (model_type == "CNN"):
        # dropout_keep_prob, embedding_size, batch_size, lr, dev_size, vocabulary_size, max_document_length, input_size, hidden_size, output_dim, n_filters, filter_sizes, num_epochs = get_params(model_type)

        hidden_size = 128
        pool_size = 2
        n_filters = 128
        filter_sizes = [3, 8]
        num_epochs = 5

        to_train = True

        cnn_model = CNN(vocab_size, embedding_size, n_filters, filter_sizes, pool_size, hidden_size, num_classes, dropout_keep_prob)

        # optimization algorithm
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
        # train and evaluation
        if (to_train):
            # train and evaluation
            run_train(num_epochs, cnn_model, train_iterator, valid_iterator, optimizer, loss_func, model_type)

        # load weights
        cnn_model.load_state_dict(torch.load(os.path.join(path, "saved_weights_CNN.pt")))
        # predict
        test_loss, test_acc = evaluate(cnn_model, test_iterator, loss_func)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


    if (model_type == "LSTM"):

        num_hidden_nodes = 93
        hidden_dim2 = 128
        num_layers = 2  # LSTM layers
        bi_directional = True
        num_epochs = 7

        to_train = True
        pad_index = Text.vocab.stoi[Text.pad_token]

        # Build the model
        lstm_model = LSTM(vocab_size, embedding_size, num_hidden_nodes, hidden_dim2 , num_classes, num_layers,
                       bi_directional, dropout_keep_prob, pad_index)

        # optimization algorithm
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)

        # train and evaluation
        if (to_train):
            # train and evaluation
            run_train(num_epochs, lstm_model, train_iterator, valid_iterator, optimizer, loss_func, model_type)

            # load weights
        lstm_model.load_state_dict(torch.load(os.path.join(path, "saved_weights_LSTM.pt")))
        # predict
        test_loss, test_acc = evaluate(lstm_model, test_iterator, loss_func)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    if (model_type == "MLP"):

        hidden_size1 = 256
        hidden_size2 = 128
        hidden_size3 = 64
        num_epochs = 6

        to_train = True

        # Build the model
        mlp_model = MLP(vocab_size, embedding_size, hidden_size1, hidden_size2, hidden_size3,  num_classes, dropout_keep_prob, max_document_length)

        # optimization algorithm
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr)

        if(to_train):
            # train and evaluation
            run_train(num_epochs, mlp_model, train_iterator, valid_iterator, optimizer, loss_func, model_type)

        # load weights
        mlp_model.load_state_dict(torch.load(os.path.join(path, "saved_weights_MLP.pt")))
        # predict
        test_loss, test_acc = evaluate(mlp_model, test_iterator, loss_func)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')





