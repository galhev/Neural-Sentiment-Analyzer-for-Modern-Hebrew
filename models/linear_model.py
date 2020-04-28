
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, text, text_lengths):
        text = text.float() # dense layer deals just with float type data
        x = self.fc1(text)
        preds = self.fc2(x)
        # preds = F.softmax(preds,1) # nn.softmax
        return preds

# In nn.BCELoss, latest layer need a sigmoid function.
# In nn.BCEWithLogitsLoss, not need a sigmoid function in latest layer.
# In nn.CrossEntropyLoss, not need a nn.Softmax(dim=1) in latest layer, because the loss funtion already include softmax function.

