import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size2, hidden_size3, hidden_size4, output_dim, dropout, max_document_length):
        super().__init__()

        # embedding and convolution layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_size*max_document_length, hidden_size2)  # dense layer
        self.fc2 = nn.Linear(hidden_size2, hidden_size3)  # dense layer
        self.fc3 = nn.Linear(hidden_size3, hidden_size4)  # dense layer
        self.fc4 = nn.Linear(hidden_size4, output_dim)  # dense layer

    def forward(self, text, text_lengths):
        # text shape = (batch_size, num_sequences)
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        x = embedded.view(embedded.shape[0], -1)  # x = Flatten()(x)
        #embedded = embedded.unsqueeze(1) # fc gets 4 dimension

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        preds = self.fc4(x)
        # preds = F.softmax(preds, 1)
        return preds


