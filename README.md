# VWS-PR
  * Source code for EACL 2021 paper **Variational Weakly Supervised Sentiment Analysis with Posterior Regularization**

## Requirements
  * Python3
  * tensorflow-gpu>=1.14.0 
  * tqdm
  * sklearn

## Usage
* VWS-PR
```bash
python yelp.py
python imdb.py
python amazon.py
```

* VWS
```bash
python yelp.py --beta 0.0
python imdb.py --beta 0.0
python amazon.py --beta 0.0
```

## Contextualized Eembeddings
We also implement a model which uses contextualized embeddings (BERT) as the input of CNN.
```bash
cd Contextualized
```




