# It Looks Fair but It's Not
Repository of the paper "It Looks Fair, but It’s Not: From Binary to Multiclass and Multigroup Fairness in Graph Neural Network-Based Models for User Profiling" by Erasmo Purificato, Ludovico Boratto and Ernesto William De Luca.

## Abstract
User profiling is a key topic in many applications, mainly information retrieval systems and social networks.
To assess how effective a user profiling approach is, its capability to classify personal characteristics (e.g. the gender, age or consumption grade of the users) is evaluated. 
Due to the fact that some of the attributes to predict are multiclass (e.g. age is non-binary), assessing *fairness* in user profiling becomes a challenge, since most of the related metrics work with binary attributes.
As a workaround, the original multiclass attributes are usually binarised to meet standard fairness metrics definitions where both the target class and sensitive attribute (such as gender or age) are binary. However, this alters the original conditions, and fairness is evaluated on classes that differ from those used in the classification.
In this paper, we extend the definitions of four existing *fairness metrics* (related to disparate impact and disparate mistreatment) from binary to multiclass scenarios, considering different settings where either the target class or the sensitive attribute are non-binary.
Our work is an endeavour to bridge the gap between formal definitions and real use cases in the field of bias detection.
The results of the experiments conducted on two real-world datasets by leveraging two state-of-the-art graph neural network-based models for user profiling show that the proposed generalisation of fairness metrics can lead to a more effective and fine-grained comprehension of disadvantaged sensitive groups and, in some cases, to a better analysis of machine learning models originally deemed to be fair.

## Requirements
The code has been executed under **Python 3.8.1**, with the dependencies listed below.

### CatGCN
```
metis==0.2a5
networkx==2.6.3
numpy==1.22.0
pandas==1.3.5
scikit_learn==1.0.2
scipy==1.7.3
texttable==1.6.4
torch==1.10.1+cu113
torch_geometric==2.0.3
torch_scatter==2.0.9
tqdm==4.62.3
```

### RHGN
```
dgl==0.6.1
dgl_cu113==0.7.2
fasttext==0.9.2
fitlog==0.9.13
hickle==4.0.4
matplotlib==3.5.1
numpy==1.22.0
pandas==1.3.5
scikit_learn==1.0.2
scipy==1.7.3
torch==1.10.1+cu113
tqdm==4.62.3
```
Notes:
* the file `requirements.txt` installs all dependencies for both models;
* the dependencies including `cu113` are meant to run on **CUDA 11.3** (install the correct package based on your version of CUDA).

## Datasets
The preprocessed files required for running each model are included as a zip file within the related folder.

The raw datasets are available at:
* **Alibaba**: [link](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56)
* **JD**: [link](https://github.com/guyulongcs/IJCAI2019_HGAT)

## Multiclass and Multigroup Fairness Metrics
ADD DEFINITION

## Run the code
Test runs for each combination of model-dataset.

### CatGCN - Alibaba dataset
```
$ cd CatGCN
$ 
```

### CatGCN - JD dataset
```
$ cd CatGCN
$ 
```

### RHGN - Alibaba dataset
```
$ cd RHGN
$ 
```

### RHGN - JD dataset
```
$ cd RHGN
$ 
```

## Contact
<!-- Erasmo Purificato (erasmo.purificato@ovgu.de) -->
erasmo.purificato@ovgu.de