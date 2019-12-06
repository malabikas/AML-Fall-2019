# AML-Fall-2019
# Project page for CS5824/ECE5424 - Advanced Machine Learning 

This page lists down the observations and results performed as a part of the project under **CS5824/ECE5424 Advanced Machine Learning** in Fall 2019 under *Prof. Dr. Bert Huang*. This project attempts to reproduce a finding from a Machine Learning Research study and present the results of the same.

---

## ABCNN: Attention-Based Convolutional Neural Network for Modelling Sentence Pairs
The [paper](https://arxiv.org/pdf/1512.05193.pdf) by Yin et al. assert that Attention-Based Convolutional Neural Networks (CNNs) perform better than CNNs without attention mechanisms. The paper integrated attention into CNNs for sentence pair modelling. Sentence pair modelling is a critical task in many Natural Language Processing (NLP) tasks such as Answer Selection, Paraphrase Identification and Textual Entailment. The experiments were performed for these three tasks. 

For this project, we are going to focus on Answer Selection.

---

## Background

Modeling a pair of sentences is a critical task for many NLP applications. 
The following example illustrates it, which is taken from the base paper - 

```Question:  how much did Waterboy gross?```

```Candidate Answer 1: the movie earned $161.5 million```

```Candidate Answer 2: this was Jerry Reed’s final film appearance```

To correctly answer the question, attention is required on _gross_ - the first candidate answer contains a corresponding unit _earned_ while the second one does not.

---

## Setup

The experiments were run on a Windows Machine using Python with the following specifications -

1. Windows 10 Predator PH315-52/64-bit/i7/32GB RAM 
2. Python 3.7.4
3. tensorflow == 1.14.0
4. numpy == 1.17.4
5. sklearn == 0.21.3
5. [WikiQA Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52419)

---

## Experiment
In this experiment we will be comparing performance of Attention based models with other base line models in NLP tasks such as Answer selection. For this we will be using WikiQA data-set.
Baselines which are considered are WordCnt; WgtWordCnt; CNN-Cnt (the state-of-theart system): combination of CNN with WordCnt and WgtWordCnt. Apart from the baselines considered we have also considered two LSTM baselines Addition and A-LSTM 
The models which we are replicating are Bi-CNN, ABCNN1, ABCNN2, ABCNN3.

For every models we have used 2 types of classifiers, Linear Regression and Support Vector Machines along with Adam as optimizer to further improve the results.

The main focus on using Attention based model is to show its wide variety of applications in different types of NLP operations and its effectiveness. Attention-based DL systems are now applied to NLP after their success in computer vision and speech recognition.

---

## Procedure
We ran the experiment on the same corpus that is used by the authors for Answer Selection which is WikiQA, an open-domain question answer dataset. 

### Pre-processing:

We used Word2Vec model to initialize the words using 300-dimensional word2vec embeddings. If a word is not present in the embedding, we use the randomly initialized embedding which is passed for all unknown words by uniform sampling from \[-.01,.01].

A sample training data from the corpus contains the following line - 

```how are glacier caves formed ?	A glacier cave is a cave formed within the ice of a glacier .	1```

The first part of the line is the question, followed by the candidate answer and ending with a label which is either ```0``` or ```1```, where ```0``` indicates wrong answer and ```1``` indicates the right answer.

While reading the corpus, both in training and testing mode, we perform the following steps -
1. Maintain a list of questions.
2. Maintain a list of candidate answers, tokenized to first 40 letters.
3. Maintain a list of labels
4. Maintin a list of word counts for all the words which occur in the _question sentence_ and is _not a stopword_ (we use the standard NLTK's English stopwords), and which also occur in the _candidate answer_
5. Maintain a list of Inverted Document Frequency (IDF) for all the words in the candidate answers.
6. The word count calculated above for a given question-answer pair, weighted word count calculated using IDF, the length of the question and the length of the answer all together form the features for the pair.

---

### BCNN and applying Attention

![Flowchart](https://github.com/malabikas/AML-Fall-2019/blob/master/Attention.PNG)
_Taken from Yin et al's ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs_

The above diagram shows the working of BCNN - a basic Bi-CNN with the layers as explained below -

#### Input Layer 
The Input layer contains the 2 sentences - question sentence _s0_ and candidate answer sentence _s1_ as a feature map initialized as a matrix of the embedded word vector of 300 dimensions times the no of words in _s0_ and _s1_
Therefore the input layer contains 2 matrices of size 300 x _len(s0)_ and 300 x _len(s1)_

#### Wide Convolution Layer
We use _tanh_ function in the convolution layer to generate a phrase, the size of which is equal to the filter width.

#### Average Pooling Layer

For BCNN, a pooling layer is used alone while in ABCNN models, the pooling layer is used in conjuntion with weighing attention.

The paper employs 2 pooling layers - _w-ap_ and _all-ap_ which has been found to extract robust features.

For non-final convolution layers, we do average pooling where the convolution layer transforms the inpput feature map of _s_ columns into new feature map of _s+w-1_ columns, where _w_ is the filter width. When averaged over, the pooling transforms the columns back to _s_ columns. 
Using this architecture, each consecutive layer gets more and more features staring from words in the bottom layer to phrases in the next and so on. Each level is able to generate more abstract features of higher granularity.

Yin et al found that performance is significantly increased if the output of all pooling layers is provided as an input to the output layer. Thus, before forwarding the result to the output layer, we perform an _all-p_ over all the columns.

#### Final Layer/Logistic Regression(LR) Layer

The last layer is an output layer which is chosen according to the tasks, eg. Logistic Regression for binary classification. 

#### Employing Attention

##### ABCNN-1
![ABCNN-1](https://github.com/malabikas/AML-Fall-2019/blob/master/ABCNN-1.PNG)
_Taken from Yin et al's ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs_

1. In ABCNN-1, attention is introduced before the convolution operation. The input representation feature map for both sentences _s0_ and _s1_, are “matched” to arrive at the Attention Matrix.
2. Every cell in the attention matrix, _Aij_, represents the attention score between the _ith_ word in _s0_ and _jth_ word in _s1_. In the paper this score is the Euclidean distance.
3. The attention matrix is transformed back to original dimension using weights.
4. The convolution operation is now performed on the resulting model which is both the input representation and the attention feature map which was calculated above.

##### ABCNN-2
![ABCNN-2](https://github.com/malabikas/AML-Fall-2019/blob/master/ABCNN-2.PNG)
_Taken from Yin et al's ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs_

1. In ABCNN-2, attention matrix is generated using the output of convolution operation.
2. This matrix is used to derive attention weights by summing the values in a row for _s0_ and summing the values in a column for _s1_.
3. This attention weight is used to re-calculate the convolution feature map columns. Every column is calculated as the attention weighted sum of the _w_ conv feature map columns that are being pooled.

##### ABCNN-3
![ABCNN-3](https://github.com/malabikas/AML-Fall-2019/blob/master/ABCNN3.PNG)
_Taken from Yin et al's ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs_

ABCNN-3, is the combination of the above 2 which is applying attention to the input of convolution and to the output of convolution while pooling.

---

## Metrics Used

We used Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR) to measure the quality of the results. We compared it against the baselines of word count (WordCnt) and Weighted word count (WgtWordCnt) as measured by Yang et al.

---

## Results
The below tables show the final results that were obtained after the models were trained uing 2 different tyes of classifiers:
LR - Linear Regression, SVM - Support Vector Machines

In the below results, the layer number represents the number of convolution-pooling blocks used.

- BCNN

    |               |          |   MAP   |   MRR   |
    |:-------------:|:--------:|:-------:|:-------:|
    | BCNN(1 layer) |    LR    |  0.6498 |  0.6592 |
    |               |   SVM    |  0.6507 |  0.6581 |
    | BCNN(2 layer) |  LR      |  0.6510 |  0.6641 |
    |               | SVM      |  0.6493 |  0.6622 |

- ABCNN-1

    |                  |          |   MAP   |   MRR   |
    |:----------------:|:--------:|:-------:|:-------:|
    | ABCNN-1(1 layer) |  LR |  0.6557 |  0.6685 |
    |                  | SVM |  0.6503 |  0.6638 |
    | ABCNN-1(2 layer) |  LR |  0.6796 |  0.6884 |
    |                  | SVM |  0.6757 |  0.6827 |

- ABCNN-2

    |                  |          |   MAP   |   MRR   |
    |:----------------:|:--------:|:-------:|:-------:|
    | ABCNN-2(1 layer) |  LR |  0.6976 |  0.6957 |
    |                  | SVM |  0.6995 |  0.6987 |
    | ABCNN-2(2 layer) |  LR |  0.7095 |  0.7065 |
    |                  | SVM |  0.7024 |  0.7076 |

- ABCNN-3

    |                  |          |   MAP   |   MRR   |
    |:----------------:|:--------:|:-------:|:-------:|
    | ABCNN-3(1 layer) |  LR |  0.7126 |  0.7108 |
    |                  | SVM |  0.7099 |  0.7116 |
    | ABCNN-3(2 layer) |  LR |  0.7193 |  0.7165 |
    |                  | SVM |  0.7129 |  0.7126 |
    
Below are examples of the results we obtained for a few questions in our test case for ABCNN3 with 2 convolution layer

```Question: who is st patty ?```

```Answer: it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .```

---
```Question: when was pokemon first started ?```

```Answer: is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .```

---
```Question: who are the members of the climax blues band ?```

```Answer: the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; drummer george newsome ; and lead vocalist and saxophonist colin cooper .```

---
```Question: when did proof die ?```

```Answer: deshaun dupree holton ( october 2 , 1973 – april 11 , 2006 ) , better known by his stage name proof , was an american rapper and actor from detroit , michigan.```

---
## Conclusion
The author of paper concludes that in Answer Selection the non-attention network BCNN already performs better than the baselines. Attention-based CNNs perform better than CNNs without attention mechanisms. For CNNs, they have test one (one-conv) and two (two-conv)
convolution-pooling blocks. The ABCNN-2 generally outperforms the  ABCNN-1 and the ABCNN3 surpasses both because, when combine the ABCNN-1 and the ABCNN-2 to form the ABCNN-3, as ability to take attention of finer-grained granularity into consideration in each convolution-pooling block. But, Due to the limited size of training data, increasing the number of convolutional layers did not show any significant improvement in the performance.

From our results we can conclude that attention based definitely provide better results in an NLP model we had data from many different papers to compare the performance of attention-based models with other models like LSTM, Addition and a few other baseline models. Also we have used a different optimizer that is ADAM instead of AdaGrad which was used in the paper.

Comparing performance of Attention based models with other baselines.

    |               |  MAP   |   MRR   |
    |:-------------:|:------:|:-------:|
    |  ABCNN3       | 0.7193 |  0.7165 |
    |  A-LSTM       | 0.6381 |  0.6537 |
    |  WordCnt      | 0.4891 |  0.4924 |
    |  WgtWordCnt   | 0.5099 |  0.5132 |
    |  CNN-Cnt      | 0.6520 |  0.6652 |
    
We were sucessfully able to replicate the reulsts from proposed paper moreover,we can also see that ABCNN models outperforms all other baseline models which indicates that ABCNNs are generally strong NN systems.

## Papers

1. Chunshui Cao, Xianming Liu, Yi Yang, Yinan Yu, Jiang Wang, Zilei Wang, Yongzhen Huang, Liang Wang, Chang Huang, Wei Xu, Deva Ramanan, and Thomas S. Huang. 2015. Look and think twice: Capturing top-down visual attention with feedback convolutional neural networks. In Proceedings of ICCV, pages 2956–2964.

2. Tianjun Xiao, Yichong Xu, Kuiyuan Yang, Jiaxing Zhang, Yuxin Peng, and Zheng Zhang. 2015. The
application of two-level attention models in deep convolutional neural network for fine-grained image classification. In Proceedings of CVPR, pages 842–850. 

3. Yi Yang, Wen-tau Yih, and Christopher Meek. 2015. WIKIQA: A challenge dataset for open-domain question answering. In Proceedings of EMNLP, pages 2013–2018.

4. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by jointly learning to align and translate. In Proceedings of ICLR.

5. Minh-Thang Luong, Hieu Pham, and Christopher D Manning. 2015. Effective approaches to attentionbased neural machine translation. In Proceedings of EMNLP, pages 1412–1421.


