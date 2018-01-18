# Paraphrase Identification with multi-perspective CNN and Dynamic k-max/min pooling

This repo contains the Torch implementation of multi-perspective convolutional neural networks with dynamic k-max/min pooling for identifying paraphrase on various dataset including SICK, MSRVID, [MSRP](https://aclweb.org/aclwiki/Paraphrase_Identification_(State_of_the_art)) and [**Quora question pair**](https://www.kaggle.com/quora/question-pairs-dataset) in Kaggle.

Dependencies
------------
- [Torch](https://github.com/torch/distro) 
- INTEL MKL library
- Glove embeddings (run fetech_and_preprocess.sh)

Running for Quora question pair
------------
```
chmod +x quora_script_train.sh
./quora_script_train.sh
```

Experiments
-------------
We designed 4 different models, those are the following:
- **Model_kmax** : applies the K-Max pooling instead of the simple max-pooling, and then we stack one more convolutional layer and use another max pooling layer to maintain the same dimension.
- **Model_kmax2** : adding another convolutional layer before choosing maximum k elements in K-Max pooling.
- **Model_kmaxmin** : Adding the min pooling on the **model_kmax**
- **Model_kmaxmin2** : Adding the min pooling on the **model_kmax2**


Results
-------------
We are presenting the result of our works based on the MSRVID dataset.
| Models  | Model_orig | Model_kmax | Model_kmax2 | Model_kmaxmin| Model_kmaxmin2|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Test score | 0.9075  | 0.9100 | 0.9088 | 0.9105 | 0.9097|

Ackowledgement
-------------
We thank provider of authors of the implementation codes for [Multi-Perspective Convolutional Neural Networks for Modeling Textual Similarity](https://github.com/hohoCode/textSimilarityConvNet) and also author of

+ Hua He, Kevin Gimpel, and Jimmy Lin. [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks.](http://aclweb.org/anthology/D/D15/D15-1181.pdf) *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP 2015)*, pages 1576-1586.

We also thank the public data providers and Torch developers for deep learning libaries and specially for implementation of Dynamic K Max Pooling as described in the paper:
+ Blunsom, Phil, Edward Grefenstette, and Nal Kalchbrenner. [A convolutional neural network for modelling sentences](https://arxiv.org/abs/1404.2188) Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics. Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics, 2014.
