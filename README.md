# Neural-Sentiment-Analyzer-for-Modern-Hebrew

This code is a neural sentiment analysis for Modern Hebrew which was written in pytorch and is based on the article: 
[Representations and Architectures in Neural Sentiment Analysis for
Morphologically Rich Languages: A Case Study from Modern Hebrew](https://www.aclweb.org/anthology/C18-1190.pdf).


## Results

Accuracy results (percentage of correct label predictions) for all architecture and representation
choices on our test set; for the string-based vocabulary, and the char-based vocabulary.

### String-based Vocabulary

Architecture | Linear | MLP | CNN | LSTM | BiLSTM | 
--- | --- | --- | --- |--- |--- |
 Token-Based| 301 | 283 | 290 | 286 | 289 |
 Morpheme-Based| 301 | 283 | 290 | 286 | 289 | 

### Char-based Vocabulary

Architecture | Linear | MLP | CNN | LSTM | BiLSTM | 
--- | --- | --- | --- |--- |--- |
 Token-Based| 301 | 283 | 290 | 286 | 289 |
 Morpheme-Based| 301 | 283 | 290 | 286 | 289 | 
