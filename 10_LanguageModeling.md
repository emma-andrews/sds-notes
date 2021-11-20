# 10 - Language Modeling

## Probabilistic Language Models
- **goal** - assign a probability to a sentence
  - *machine translation* - P(high winds tonight) > P(large winds tonight)
  - *spell correction* - the office is about fifteen minuets from my house
    - P(about fifteen minutes from) > P(about fifteen minuets from)
  - *speech recognition* - P(I saw a van) >> P(eyes awe of an)
  - summarization, question-answering, etc
  - $P(W)=P(w_1,w_2,w_3,w_4,w_5)$
- **related task** - probability of an upcoming word
  - $P(w_5|w_1,w_2,w_3,w_4)$
- **language model (LM)** - model that computes either of the two formulas
  - also called *grammar*
- how do we compute P(W)?
  - rely on **Chain Rule of Probability**

## Chain Rule
- definitions of conditional probabilities
  - $P(B|A) = \frac{P(A,B)}{P(A)}$
  - $P(A,B)=P(A)P(B|A)$
- **general equation**
  - $P(x_1,x_2,x_3,...,x_n)=P(x_1)P(x_2|x_1)P(x_3|x_1,x_2)...P(x_n|x_1,...,x_{n-1})$

## Applied Chain Rule
- **applied** to *joint probability of words* in a sentence
  - $P(w_1w_2...w_n)=\Pi{i}\ {P(w_i|w_1w_2...w_{i-1})}$
  - sidenote: latex sucks so if you see $\Pi {i}$ that means i is the bound, not multiplying the rest of the stuff by i
- example: P("its water is so transparent")
  - $=P(its)\times P(water|its)\times P(so|its\ water\ is)\times P(transparent|its\ water\ is\ so)$
- **naive estimation** - count and divide
  - $P(the|its\ water\ is\ so\ transparent\ that)=\frac{Count(its\ water\ is\ so\ transparent\ that\ the)}{Count(its\ water\ is\ so\ transparent\ that)}$
  - but there are *way too many* possible sentences
  - never see enough data for estimating

## Markov Assumption
- *simplify* assumption
- approximate each component in the product
  - $P(w_i|w_1w_2...w_{i-1}\approx P(w_i|w_{i-k}...w_{i-1})$
- **unigram model** - simplest case
  - $P(w_1w_2...w_n)\approx \Pi{i}{\ P(w_i)}$
- **bigram model** - condition on the previous word
  - $P(w_i|w_1w_2...w_{i-1})\approx P(w_i|w_{i-1})$
- **n-gram models** - can extend to trigrams, 4-grams, etc
  - in general this is an *insufficient model of language* because language has **long-distance dependencies**
    - words that have meaning tied with another part of the sentence may be many many words separated
  - we can often get away with n-gram models though

## Estimating Bigram Probabilities
- **maximum likelihood estimate**
  - count abbreviated to c in following formulas
  - $P(w_i|w_{i-1})=\frac{c(w_{i-1},w_i)}{c(w_{i-1})}$