# Fake News Detection: Machine Learning and Deep Learning

A university data-mining project comparing five traditional machine learning classifiers and a convolutional neural network (CNN) on a labelled news-article dataset.

**Authors:** Chukwu John Okike and Muhammad Bala (as credited in the final project report).

## Implementation

- `classifier_comparison.py`: TF-IDF, multinomial Naive Bayes, logistic regression, linear SVM, random forest and an entropy-based decision tree.
- `cnn_softmax_fake_news.py`: tokenization, embedding, 1D convolution and softmax binary classification.
- `prepare_dataset.py`: creates a compatible ARFF file from separately obtained Fake.csv and True.csv files.

## Reproduce the experiment

Use a Python environment compatible with TensorFlow (Python 3.11 is a suitable starting point). Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Obtain the original Fake and Real News Dataset separately under its applicable terms. Place the original Fake.csv and True.csv beside the scripts. Dataset files, trained model weights and personal coursework reports are intentionally not distributed in the archive.

```bash
python prepare_dataset.py
python classifier_comparison.py
python cnn_softmax_fake_news.py
```

The original scripts use a stratified 70/30 split with random_state=42 and compare accuracy, precision, recall, F1 and confusion matrices. Reconstructed data and dependency versions may affect results. The CNN is not a general-purpose fact-checking system; classification metrics on this dataset do not establish reliable detection of misinformation in new settings.

The original experiment used the public Fake and Real News Dataset distributed via Kaggle; consult the dataset provider and original publication for access and terms. The source archive is provided for scholarly citation and reference, with All Rights Reserved for the original project software. See LICENSE.

Student IDs and coursework files have been excluded.
