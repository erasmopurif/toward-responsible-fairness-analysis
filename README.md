[![Python](https://img.shields.io/badge/Python-3.9.18-%233776AB?logo=Python)](https://www.python.org/)

# Toward a Responsible Fairness Analysis
Repository of the paper *"Toward a Responsible Fairness Analysis: From Binary to Multiclass and Multigroup Assessment in Graph Neural Network-Based User Modeling Tasks"* by Erasmo Purificato, Ludovico Boratto and Ernesto William De Luca, published at Minds and Machines Journal for Artificial Intelligence, Philosophy and Cognitive Science (Special Issue on "Interdisciplinary Perspectives on the (Un)fairness of Artificial Intelligence").

## Abstract
User modeling is a key topic in many applications, mainly social networks and information retrieval systems.
To assess the effectiveness of a user modeling approach, its capability to classify personal characteristics (e.g., the gender, age, or consumption grade of the users) is evaluated. 
Due to the fact that some of the attributes to predict are multiclass (e.g., age usually encompasses multiple ranges), assessing \textit{fairness} in user modeling becomes a challenge since most of the related metrics work with binary attributes.
As a workaround, the original multiclass attributes are usually binarized to meet standard fairness metrics definitions where both the target class and sensitive attribute (such as gender or age) are binary. However, this alters the original conditions, and fairness is evaluated on classes that differ from those used in the classification.
In this article, we extend the definitions of four existing fairness metrics (related to disparate impact and disparate mistreatment) from binary to multiclass scenarios, considering different settings where either the target class or the sensitive attribute includes more than two groups.
Our work endeavors to bridge the gap between formal definitions and real use cases in bias detection.
The results of the experiments, conducted on four real-world datasets by leveraging two state-of-the-art graph neural network-based models for user modeling, show that the proposed generalization of fairness metrics can lead to a more effective and fine-grained comprehension of disadvantaged sensitive groups and, in some cases, to a better analysis of machine learning models originally deemed to be fair.

## Requirements
The code has been executed under **Python 3.9.18**, with the dependencies listed below.

### CatGCN
```
metis==0.2a5
networkx==2.6.3
numpy==1.22.0
pandas==1.3.5
scikit_learn==1.1.2
scipy==1.7.3
texttable==1.6.4
torch==1.10.1+cu113
torch_geometric==2.0.3
torch_scatter==2.0.9
tqdm==4.62.3
```

### RHGN
```
dgl==0.9.1
dgl_cu113==0.7.2
hickle==4.0.4
matplotlib==3.5.1
numpy==1.22.0
pandas==1.3.5
scikit_learn==1.1.2
scipy==1.7.3
torch==1.10.1+cu113
```
Notes:
* the file `requirements.txt` installs all dependencies for both models;
* the dependencies including `cu113` are meant to run on **CUDA 11.3** (install the correct package based on your version of CUDA).

## Datasets
The preprocessed files required for running each model are included as a zip file within the related folder.

The raw datasets are available at:
* [**Alibaba**](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56)
* [**JD**](https://github.com/guyulongcs/IJCAI2019_HGAT)
* [**Pokec**](https://github.com/EnyanDai/FairGNN/tree/main/dataset/pokec)
* [**NBA**](https://github.com/EnyanDai/FairGNN/tree/main/dataset/NBA)

## Multiclass and Multigroup Fairness Metrics
The repository implements the generalised **Multiclass and Multigroup Fairness Metrics** presented in the paper.

Let:
* $M$ be the number of *classes*;
* $N$ be the number of *demographic groups*;
* $y \in \lbrace 0, ..., M-1 \rbrace$ be the *target class*;
* $\hat{y} \in \lbrace 0, ..., M-1 \rbrace$ be the *predicted class*;
* $s \in \lbrace 0, ..., N-1 \rbrace$ be the *sensitive attribute*.

The score of each of the metrics displayed below should be equal across every class and group:

### **Multiclass and multigroup statistical parity**
$$
P(\hat{y} = m | s = n), \forall m \in \lbrace 0,...,M-1 \rbrace \land \forall n \in \lbrace 0,...,N-1 \rbrace
$$

### **Multiclass and multigroup equal opportunity**
$$
P(\hat{y} = m | y = m, s = n), \forall m \in \lbrace 0,...,M-1 \rbrace \land \forall n \in \lbrace 0,...,N-1 \rbrace
$$

### **Multiclass and multigroup overall accuracy equality**
$$
\sum_{m=0}^{M-1} P(\hat{y} = m | y = m, s = n), \forall n \in \lbrace 0,...,N-1 \rbrace
$$

### **Multiclass and multigroup treatment equality**
$$
\frac{P(\hat{y} = m | y \neq m, s = n)}{P(\hat{y} \neq m | y = m, s = n)}, \forall m \in \lbrace 0,...,M-1 \rbrace \land \forall n \in \lbrace 0,...,N-1 \rbrace
$$

## Run the code
Example test runs for each combination of model-dataset.

### CatGCN - Alibaba dataset
```
$ cd CatGCN
$ python3 main.py --seed 11 --gpu 0 --learning-rate 0.1 --weight-decay 1e-5 \
--dropout 0.1 --diag-probe 1 --graph-refining agc --aggr-pooling mean --grn-units 64 \
--bi-interaction nfm --nfm-units none --graph-layer pna --gnn-hops 1 --gnn-units none \
--aggr-style sum --balance-ratio 0.7 --edge-path ./input/ali_data/user_edge.csv \
--field-path ./input_ali_data/user_field.npy --target-path ./input_ali_data/user_buy.csv \
--labels-path ./input_ali_data/user_labels.csv --sens-attr age --label buy
```

### CatGCN - JD dataset
```
$ cd CatGCN
$ python3 main.py --seed 11 --gpu 0 --learning-rate 1e-2 --weight-decay 1e-5 \
--dropout 0.1 --diag-probe 39 --graph-refining agc --aggr-pooling mean --grn-units 64 \
--bi-interaction nfm --nfm-units none --graph-layer pna --gnn-hops 1 --gnn-units none \
--aggr-style sum --balance-ratio 0.7 --edge-path ./input_jd_data/user_edge.csv \
--field-path ./input_jd_data/user_field.npy --target-path ./input_jd_data/user_expense.csv \
--labels-path ./input_jd_data/user_labels.csv --sens-attr bin_age --label expense
```

### CatGCN - Pokec dataset
```
$ cd CatGCN
$ python3 main.py --seed 11 --gpu 0 --learning-rate 1e-3 --weight-decay 1e-5 \
--dropout 0.7 --diag-probe 1 --graph-refining agc --aggr-pooling mean --grn-units 64 \
--bi-interaction nfm --nfm-units none --graph-layer pna --gnn-hops 1 --gnn-units none \
--aggr-style sum --balance-ratio 0.1 --edge-path ./input_pokec_data/edges.csv \
--field-path ./input_pokec_data/categories.npy --target-path ./input_pokec_data/user_workfield.csv \
--labels-path ./input_pokec_data/users.csv --sens-attr bin_age --label work_field
```

### CatGCN - NBA dataset
```
$ cd CatGCN
$ python3 main.py --seed 3 --gpu 0 --learning-rate o.1 --weight-decay 1e-4 \
--dropout 0.9 --diag-probe 39 --graph-refining agc --aggr-pooling mean --grn-units 64 \
--bi-interaction nfm --nfm-units none --graph-layer pna --gnn-hops 1 --gnn-units 64 \
--aggr-style sum --balance-ratio 0.7 --edge-path ./input_nba_data/edges.csv \
--field-path ./input_nba_data/points.npy --target-path ./input_nba_data/user_bin_salary.csv \
--labels-path ./input_nba_data/users.csv --sens-attr bin_age --label bin_salary
```

### RHGN - Alibaba dataset
```
$ cd RHGN
$ python3 ali_main.py --seed 42 --gpu 0 --model RHGN --data_dir ./input_ali_data/ \
--graph G --max_lr 0.1 --n_hid 32 --clip 2 --n_epoch 100 \
--label bin_buy --sens_attr bin_age
```

### RHGN - JD dataset
```
$ cd RHGN
$ python3 jd_main.py --seed 3 --gpu 0 --model RHGN --data_dir ./input_jd_data/ \
--graph G --max_lr 1e-3 --n_hid 64 --clip 1 --n_epoch 100 \
--label bin_exp --sens_attr bin_age
```

### RHGN - Pokec dataset
```
$ cd RHGN
$ python3 pokec_main.py --seed 11 --gpu 0 --model RHGN --data_dir ./input_pokec_data/ \
--graph G --max_lr 1e-3 --n_hid 64 --clip 2 --n_epoch 100 \
--label bin_work_field --sens_attr age
```

### RHGN - NBA dataset
```
$ cd RHGN
$ python3 nba_main.py --seed 11 --gpu 0 --model RHGN --data_dir ./input_nba_data/ \
--graph G --max_lr 0.1 --n_hid 32 --clip 1 --n_epoch 100 \
--label salary --sens_attr age
```

## Contact
<!-- Erasmo Purificato (erasmo.purificato@ovgu.de) -->
erasmo.purificato@ovgu.de
