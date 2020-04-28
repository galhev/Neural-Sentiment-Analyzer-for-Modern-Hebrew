from torchtext import data
import random
import matplotlib.pyplot as plt
import seaborn as sns


def labels_distribution(y):
    fig = plt.figure(figsize=(8,5))
    ax = sns.barplot(x=y.unique(),y=y.value_counts());
    ax.set(xlabel='Labels')
    print(y.value_counts())

def get_files(path, dev_size, max_document_length, seed, data_type, tokenizer):
    # include_lengths = True - This will cause batch.text to now be a tuple with the first element being our sentence (a numericalized tensor that has been padded) and the second element being the actual lengths of our sentences.
    Text = data.Field(tokenize=tokenizer, batch_first=True, include_lengths=True, fix_length=max_document_length) # fix_length - make the sentences padded in the same lengths for all the batches
    Label = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

    # All files:
    fields = [('text', Text), ('labels', Label)]

    train_data, test_data = data.TabularDataset.splits(
        path=path,
        train= data_type + '_train.tsv',
        test= data_type + '_test.tsv',
        format='tsv',
        fields=fields,
        skip_header=False
    )

    train_data, valid_data = train_data.split(split_ratio=dev_size, random_state=random.seed(seed))
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')
    return train_data, valid_data, test_data, Text, Label

