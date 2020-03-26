import torch
from torch.autograd import Variable
from torchtext import data
import matplotlib.pyplot as plt

from data_preprocessing import get_batch


def train_linear(model, num_epochs, train_text, train_label, batch_size, optimizer, criterion, vec_length, word2index):
    # Train the Model
    for epoch in range(num_epochs):
        total_batch = int(len(train_text) / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = get_batch(train_text, train_label, i, batch_size, vec_length, word2index)
            text = Variable(torch.FloatTensor(batch_x))
            labels = Variable(torch.LongTensor(batch_y))

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = model(text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 4 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_text) // batch_size, loss.data.item()))
            return model

def test_linear(model, test_text, test_label,vec_length, word2index):
    # Test the Model
    correct = 0
    total = 0
    total_test_data = len(test_label)
    batch_x_test, batch_y_test = get_batch(test_text, test_label, 0, total_test_data, vec_length, word2index)
    text = Variable(torch.FloatTensor(batch_x_test))
    labels = torch.LongTensor(batch_y_test)
    outputs = model(text)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

    print('Accuracy of the network on the %d test sentences is: %d %%' % (test_label.shape[0], 100 * correct / total))


def accuracy(probs, target):
  winners = probs.argmax(dim=1)
  corrects = (winners == target)
  accuracy = corrects.sum().float() / float(target.size(0))
  return accuracy

######################################## Using torchText ######################################

def create_iterator(train_data, valid_data, test_data, batch_size, device):
    #  BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    # by setting sort_within_batch = True.
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
        batch_size = batch_size,
        sort_key = lambda x: len(x.text), # Sort the batches by text length size
        sort_within_batch = True,
        device = device)
    return train_iterator, valid_iterator, test_iterator


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    for batch in iterator:
        optimizer.zero_grad()
        # retrieve text and no. of words
        text, text_lengths = batch.text

        predictions = model(text, text_lengths)
        loss = criterion(predictions, batch.labels.squeeze())

        acc = accuracy(predictions, batch.labels)

        # perform backpropagation
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.labels)

            acc = accuracy(predictions, batch.labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def run_train(epochs, model, train_iterator, valid_iterator, optimizer, criterion, model_type):
    best_valid_loss = float('inf')

    for epoch in range(epochs):

        # train the model
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

        # evaluate the model
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights'+'_'+model_type+'.pt')

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


def plot_loss_and_accuracy(history):
    fig, axs = plt.subplots(1, 2, sharex=True)

    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title('Model Loss')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    axs[1].plot(history.history['acc'])
    axs[1].plot(history.history['val_acc'])
    axs[1].set_title('Model Accuracy')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    fig.tight_layout()
    plt.show()

