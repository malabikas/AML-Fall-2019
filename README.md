# AML-Fall-2019
Project repository/page for CS5824/ECE5424 - Advanced Machine Learning project. 

This page lists down the observations and results performed as a part of the project under **CS5824/ECE5424 Advanced Machine Learning** in Fall 2019 under *Prof. Dr. Bert Huang*. This project attempts to reproduce a finding from a Machine Learning Research study and present the results of the same.

## ABCNN: Attention-Based Convolutional Neural Network for Modelling Sentence Pairs
The [paper](https://arxiv.org/pdf/1512.05193.pdf) asserts that Attention-Based Convolutional Neural Networks (CNNs) perform better than CNNs without attention mechanisms. The paper integrated attention into CNNs for sentence pair modelling. Sentence pair modelling is a critical task in many Natural Language Processing (NLP) tasks such as Answer Selection, Paraphrase Identification and Textual Entailment. The experiments were performed for these three tasks. For this project, we are going to focus on Answer Selection.

## Background

Modeling a pair of sentences is a critical task for many NLP applications. 
<need to edit> 
s0 how much did Waterboy gross?
s1+ the movie earned $161.5 million
s1- this was Jerry Reed’s final film appearance
For AS, correctly answering s0 requires attention on “gross”: s1+ contains a corresponding unit (“earned”) while s1- does not.



## Setup

The experiments were run on a Windows Machine using Python with the following specifications -

1. Windows 10 Predator PH315-52/64-bit/i7/32GB RAM 
2. Python 3.7.4
3. tensorflow == 1.14.0
4. numpy == 1.17.4
5. sklearn == 0.21.3
5. [WikiQA Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52419)


## Experiment
The paper referred concludes that in Answer Selection the non-attention network BCNN already performs better than the baselines. Attention-based CNNs perform better than CNNs without attention mechanisms. For CNNs, they have test one (one-conv) and two (two-conv)
convolution-pooling blocks. The ABCNN-2 generally outperforms the  ABCNN-1 and the ABCNN3 surpasses both because, when combine the ABCNN-1 and the ABCNN-2 to form the ABCNN-3, as ability to take attention of finer-grained granularity into consideration in each convolution-pooling block. But, Due to the limited size of training data, increasing the number of convolutional layers did not show any significant improvement in the performance.

Baselines which thy have considered, the first three are by Yang et al.: WordCnt; WgtWordCnt; CNN-Cnt (the state-of-theart system): combine CNN with WordCnt and WgtWordCnt. Apart from the baselines considered by Yang, they have also considered two LSTM baselines Addition and A-LSTM 

In addition, linguistic features contribute in all three tasks: improvements by 0.0321 (MAP) and 0.0338 (MRR).
They concluded by stating Attention based models can be used to build really strong neural networks.





## Results

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
    | ABCNN-3(2 layer) |  LR |  0.6571 |  0.6722 |
    |                  | SVM |  0.7193 |  0.7165 |
    
Below are a few examples results from different networks.

### ABCNN3 - 1 layer LR

Q- who is st patty ?

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ?

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ?

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; 
drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- deshaun dupree holton ( october 2 , 1973 – april 11 , 2006 ) , better known by his stage name proof , was an american rapper and actor from detroit , michigan.

max MAP: 0.7126145192917447, max MRR: 0.7108062618199071

### ABCNN3 - 2 layer LR

Q- who is st patty ? 

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ? 

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ? 

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; 
drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- deshaun dupree holton ( october 2 , 1973 – april 11 , 2006 ) , better known by his stage name proof , was an american rapper and actor from detroit , michigan.

max MAP: 0.7193757702010519, max MRR: 0.7165993562907142


### ABCNN3 - 1 layer SVM

Q- who is st patty ? 

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ? 

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ? 

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- deshaun dupree holton ( october 2 , 1973 – april 11 , 2006 ) , better known by his stage name proof , was an american rapper and actor from detroit , michigan .

max MAP: 0.7099585687272065, max MRR: 0.7116442579236981

### ABCNN3 - 2 layer SVM

Q- who is st patty ?

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ? 

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ? 

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- deshaun dupree holton ( october 2 , 1973 – april 11 , 2006 ) , better known by his stage name proof , was an american rapper and actor from detroit , michigan .

max MAP: 0.7129132538700438, max MRR: 0.7126593833692598

### ABCNN2 - 1 layer LR

Q- who is st patty ? 

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ? 

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ?

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; 
drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- deshaun dupree holton ( october 2 , 1973 – april 11 , 2006 ) , better known by his stage name proof , was an american rapper and 
actor from detroit , michigan .

max MAP: 0.6976783762533840, max MRR: 0.6957261930673619

### ABCNN2 - 2 layer LR

Q- who is st patty ? 

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ? 

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ? 

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; 
drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- deshaun dupree holton ( october 2 , 1973 – april 11 , 2006 ) , better known by his stage name proof , was an american rapper and actor from detroit , michigan .

max MAP: 0.709527190475839, max MRR: 0.706527103659273

### ABCNN2 - 1 layer SVM

Q- who is st patty ? 

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ? 

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ? 

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; 
drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- deshaun dupree holton ( october 2 , 1973 – april 11 , 2006 ) , better known by his stage name proof , was an american rapper and 
actor from detroit , michigan .

max MAP: 0.6995279173549926, max MRR: 0.6987253940575274

### ABCNN2 - 2 layer SVM

Q- who is st patty ? 

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ? 

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ? 

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- deshaun dupree holton ( october 2 , 1973 – april 11 , 2006 ) , better known by his stage name proof , was an american rapper and actor from detroit , michigan .

max MAP: 0.702410004825818, max MRR: 0.7076416394056357

### ABCNN1 - 1 layer LR

Q- who is st patty ? 

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ? 

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ? 

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; 
drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- in 2006 , proof was shot and killed during an altercation at the ccc nightclub in detroit .

max MAP: 0.6557968939731829, max MRR: 0.668525113432521

### ABCNN1 - 1 layer SVM

Q- who is st patty ? 

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ? 

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ? 

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; 
drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- in 2006 proof was shot and killed during an altercation at the ccc nightclub in detroit .

max MAP: 0.6503264699070797, max MRR: 0.663860310928212

### ABCNN1 - 2 layer LR

Q- who is st patty ? 

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ? 

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ?

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; 
drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- deshaun dupree holton ( october 2 , 1973 – april 11 , 2006 ) , better known by his stage name proof , was an american rapper and 
actor from detroit , michigan .

max MAP: 0.6796591410508728, max MRR: 0.6884243022157253

### ABCNN1 - 2 layer SVM

Q- who is st patty ? 

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ? 

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ? 

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; 
drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die

A- deshaun dupree holton ( october 2 , 1973 – april 11 , 2006 ) , better known by his stage name proof , was an american rapper and 
actor from detroit , michigan .

max MAP: 0.6757270541529802, max MRR: 0.6827977463779933

### BCNN - 2 layer LR

Q- who is st patty ? 

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ? 

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ? 

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; 
drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- deshaun dupree holton ( october 2 , 1973 – april 11 , 2006 ) , better known by his stage name proof , was an american rapper and 
actor from detroit , michigan .

max MAP: 0.6510566209234175, max MRR: 0.6641459551771827

### BCNN - 2 layer SVM

Q- who is st patty ? 

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ? 

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ? 

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; 
drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- deshaun dupree holton ( october 2 , 1973 – april 11 , 2006 ) , better known by his stage name proof , was an american rapper and 
actor from detroit , michigan .

max MAP: 0.6493646812360265, max MRR: 0.6622141804240569

### BCNN - 1 layer LR

Q- who is st patty ? 

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started 

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ? 

A- some authors also define an alkali as a base that dissolves in water .

Q- who are the members of the climax blues band ? 

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; 
drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- in 2006 , proof was shot and killed during an altercation at the ccc nightclub in detroit .

max MAP: 0.6498818245318789, max MRR: 0.6592214712585085

### BCNN - 1 layer SVM

Q- who is st patty ? 

A- it is named after saint patrick ( ad 385–461 ) , the most commonly recognised of the patron saints of ireland .

Q- when was pokemon first started

A- is a media franchise published and owned by japanese video game company nintendo and created by satoshi tajiri in 1996 .

Q- what does alkali do to liquids ?

A- some authors also define an alkali as a base that dissolves in water . 1 -0.9740697249362577

Q- who are the members of the climax blues band ? 

A- the original members were guitarist/vocalist peter haycock , guitarist derek holt ; keyboardist arthur wood ; bassist richard jones ; 
drummer george newsome ; and lead vocalist and saxophonist colin cooper .

Q- when did proof die 

A- deshaun dupree holton ( october 2 , 1973 – april 11 , 2006 ) , better known by his stage name proof , was an american rapper and 
actor from detroit , michigan .

max MAP: 0.6507638844777246, max MRR: 0.6581023019809028



    
## Conclusion

## Papers



