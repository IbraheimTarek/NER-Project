# Pizza semantic parsing implementation

The dataset
(https://github.com/amazon-science/pizza-semantic-parsing-dataset/tree/main)

The paper
(https://arxiv.org/abs/2212.00265)

## Project Document

## Project Pipeline

We begin our model by identifying the entities and labeling them using BIO taging techique.
After labeling the data we use an embedding layer to extract the feature for the bidirectional LSTMS.

The pipeline consists of 2 models one for identifying the order sequences and the existing orders inside the given input and another one for identifying the existing entities for the discovered order like topping, sizes and numbers and drinks.
The 2 networks each one of them consists of 1 embedding layer, 2 biLSTM layers and one fully connected layer for the output. Using adam optimizer with 0.001 learning rate and drop out of 0.25 to prevent overfitting and cross entropy loss to update weights.

![Test 1 Image](https://github.com/IbraheimTarek/NER-Project/blob/main/Images/nlp_pipeline.png)

## Final results

private public
0.48657 0.711564

private data was never seen before

## Streamlit Application

![Test 3 Image](https://github.com/IbraheimTarek/NER-Project/blob/main/Images/app1.jpg)
![Test 4 Image](https://github.com/IbraheimTarek/NER-Project/blob/main/Images/app2.jpg)

## Contributors

- [Ibraheim Tarek](https://github.com/IbraheimTarek)
- [Youssef Hagag](https://github.com/Youssef-Hagag)
- [Mahmoud Sobhy](https://github.com/MoSobhy01)
- [Youssef Rabie](https://github.com/YoussefMoRabie)
