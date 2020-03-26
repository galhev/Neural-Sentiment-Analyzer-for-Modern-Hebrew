import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, n_filters, filter_sizes, pool_size, hidden_size, num_classes,
                 dropout):
        super().__init__()

        # embedding and convolution layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Specify convolutions with filters of different sizes (fs)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1,
                                              out_channels=n_filters,
                                              kernel_size=(fs, embed_size))
                                    for fs in filter_sizes])


        #self.maxpool1 = nn.MaxPool1d(filter_sizes[0])
        #self.maxpool2 = nn.MaxPool1d(filter_sizes[1])

        #self.wool=nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=(3, embed_size))
        self.max_pool1 = nn.MaxPool1d(pool_size) # pool size = 2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(95*n_filters, hidden_size, bias=True)  # dense  # TODO: fixed_length to dynamic batch - more efficient in calculations
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)  # dense layer

    def forward(self, text, text_lengths):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1) # Conv1d gets 4 dimension
        # embedded = [batch size, 1, sent len, emb dim]
        # Perform convolutions and apply Pooling layer to reduce dimensionality

        # conv shape: 50, 93, 128, filter_size = 8
        # conv shape: 50, 98, 128, filter_size = 3
        # convs shpae: [50, 49, 128] [50, 46,128]
        # conv1d dimension input is: [batch_size, channels, length]
        convolution = [conv(embedded) for conv in self.convs]
        #max1 = self.maxpool1(convolution[1].squeeze())
        #max2 = self.maxpool2(convolution[0].squeeze())
        # shape after maxpool = [batch size, n_filters]
        #max1 = F.max_pool1d(convolution[0].squeeze(), kernel_size=8)
        #max2 = F.max_pool1d(convolution[1].squeeze(), kernel_size=3)

        max1 = self.max_pool1(convolution[0].squeeze()) # pooling size = 2
        max2 = self.max_pool1(convolution[1].squeeze())

        #max1_c = max1.view(max1.shape[0], max1.shape[2], -1)
        #max2_c = max2.view(max2.shape[0], max2.shape[2], -1)

        cat = torch.cat((max1, max2), dim=2)
        #cat shpae: [50, 95 ,128]
        x = cat.view(cat.shape[0], -1)  # x = Flatten()(x)
        x = self.fc1(self.relu(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # We don’t have to manually apply a log_softmax layer after our final layer because nn.CrossEntropyLoss does that for us. However, we need to apply log_softmax for our validation and testing.
        #preds = F.softmax(x, 1)
        #preds = preds.argmax(dim=1).unsqueeze(0)
        return x


