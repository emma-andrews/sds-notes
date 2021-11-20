# 11 - Naive Bayes

## Text Classification
- **input** - a document $d$ and a fixed set of classes $C=\{c_1,c_2,...,c_j\}$
- **output** - a predicted class $c \in C$
- *classification methods* - **hand-coded rules**
  - rules based on combinations of words or other features
  - accuracy can be high, if rules carefully refined by expert
  - but building and maintaining these rules is *expensive*
- *classification *methods* - **supervised machine learning**
  - **input** - a document $d$, a fixed set of classes $C=\{c_1,c_2,...,c_j\}$, and a training set of $m$ hand-labeled documents $(d_1,c_1),...,(d_m,c_m)$
  - **output** - a learned classifier $\gamma :d\rArr c$
  - any kind of classifier
    - naive bayes
    - logistic regression
    - neural networks
    - k-nearest neighbors
    - etc

## Naive Bayes Classifier
- **intuition for Naive Bayes** - simple classification method based on Bayes rule
  - relies on very simple representation of document, *bag of words*
- **Bayes' Rule** - for a document $d$ and a class $c$, $P(c|d)=\frac{P(d|c)P(c)}{P(d)}$
- **Naive Bayes Classifier**
  - MAP = most likely class
  - $c_{MAP}=\argmax {(c\in C)}{P(c|d)}$
  - using Bayes' Rule, can simplify to
  - $c_{MAP}=\argmax {(c\in C)}{P(d|c)P(c)}$
  - where $P(d|c)$ is the *likelihood* and $P(c)$ is the *prior*
  - $d$ can be represented as features $x_1,...,x_n$
  - $O(|X|^n\times |C|)$ parameters
    - could only be estimated if a very, very large number of training examples was available
    - we can just count the relative frequencies in a corpus
    - 