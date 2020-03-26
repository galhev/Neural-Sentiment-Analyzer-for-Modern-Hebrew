# Neural-Sentiment-Analyzer-for-Modern-Hebrew

This code is a neural sentiment analysis for Modern Hebrew which was written in pytorch and is based on the article: 
[Representations and Architectures in Neural Sentiment Analysis for
Morphologically Rich Languages: A Case Study from Modern Hebrew](https://www.aclweb.org/anthology/C18-1190.pdf).


## Results

Accuracy results (percentage of correct label predictions) for all architecture and representation
choices on the test set; for the string-based vocabulary, and the char-based vocabulary.

### String-based Vocabulary

Architecture | Linear | MLP | CNN | LSTM | BiLSTM | 
--- | --- | --- | --- |--- |--- |
 Token-Based| 64.08 | 79.31 | 86.92 | 84.42 | 85.96 |
 Morpheme-Based| 62.38 | 78.19 | 85.73 | 82.19 | 85.88 | 

### Char-based Vocabulary

Architecture | Linear | MLP | CNN | LSTM | BiLSTM | 
--- | --- | --- | --- |--- |--- |
 Token-Based| 66.15 | 75.42 | 84.77 | 74.42 | 79.35 |
 Morpheme-Based| 68.42 | 73.5 | 82.42 | 74.65 | 78.08 | 
