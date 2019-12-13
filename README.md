# TDSA Comparisons

This code base explores how advancements in and outside of TDSA have improved TDSA models generally. At the moment we only focus on English datasets. 

## Datasets
All of the data is stored in a private folder within `./data`. The datasets that we will use are the following:
1. Twitter Election dataset from [Wang et al. 2017](https://www.aclweb.org/anthology/E17-1046/) 
2. [SemEval 2014 task 4 subtask 2 Laptop](http://alt.qcri.org/semeval2014/task4/) of which the training data can found [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-train-data-v20-annotation-guidelines/683b709298b811e3a0e2842b2b6a04d7c7a19307f18a4940beef6a6143f937f0/) and the test data [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-test-data-gold-annotations/b98d11cec18211e38229842b2b6a04d77591d40acd7542b7af823a54fb03a155/)
3. SemEval 2014 task 4 subtask 2 Restaurant of which the training and test can be found in the same place as the Laptop.

Only the SemEval datasets need downloading into the `./data` folder, the Election dataset is automatically downloaded through the code. To create the training, validaton and test splits for each of the datasets run the following bash script (it uses the dataset splitter from [target_extraction code base](https://github.com/apmoore1/target-extraction/blob/master/create_splits.py)):
``` bash
./tdsa_comparisons/splitting_data/create_splits.sh
```
The Election, Laptop, and Restaurant dataset splits can be found in there respective folders: `./data/election_dataset`, `./data/laptop_dataset`, and `./data/restaurant_dataset`. Each of the targets in each text are ordered so that the first target that occurs in the sentence is also the first target within the TargetText object. This re-ordering is done so that methods that rely on this ordering can be used such as the model from [Hazarika et al. 2018](https://www.aclweb.org/anthology/N18-2043/) which encodes the aspect/target representation in sequential order using an LSTM.

## Word Embeddings
For all of the experiments the [840 billion token 300 dimension GloVe word vectors](https://nlp.stanford.edu/projects/glove/) will be used, these word vectors are stored within the `resources` directory under `./resources/embeddings/glove.840B.300d.txt`. The only time these word vectors will not be used is during the Contextualised Word Representation (CWR) experiments where the CWR will be used instead.

The CWR that are Transformer ELMo models are stored in the directory `./resources/CWR/`, of which each dataset has their own CWR due to domain specific CWR being shown to outperform non-domain specific CWR by a large margin.

## Experiments
In all of the experiments we are going to use the following 3 models:
1. **TDLSTM**
2. **IAN**
3. **InterAE** (model from [Hazarika et al. 2018](https://www.aclweb.org/anthology/N18-2043/)) -- The baseline version of this is without the Inter-Aspect LSTM which is the AE model from [Wang et al. 2016](https://www.aclweb.org/anthology/D16-1058.pdf) with attention applied after the sentence LSTM. Each time we refer to this model's baseline version it will be called **AE**.

The 4 main experiments are the following:
1. Baseline -- TDLSTM, IAN, and AE as is without modification using the GloVe word vectors.
2. Inter-Aspect -- TDLSTM, IAN, and InterAE with Inter-Aspect modelling. To incorporate Inter Aspect modeling we will adopt the method of [Hazarika et al. 2018](https://www.aclweb.org/anthology/N18-2043/) which uses an LSTM.
3. Position -- Run the IAN, and AE models with position of the target encoded. TDLSTM is not used in this experiment as the model already encodes position information via the network architecture.
4. CWR -- Replace the GloVe vectors with domain specific CWR for TDLSTM, IAN, and AE.

All of the default training configurations for each of the 3 models can be found [here](./resources/model_configs/).

### Baseline Experiments

{'metadata':{'predictions':{'target_sentiment_{word_vector}_{position}_{}}}

## Results
All of the results which have been some what anonymised (`text` data from the dataset is removed) from these models are released in `JSON` format nearly identical to there original format but without the `text` as the SemEval dataset do not allow re-distribution of the data. The results from all of the experiments can be found in the following folders:
1. [`./saved_results/restaurant`](./saved_results/restaurant)
2. [`./saved_results/laptop`](./saved_results/laptop)
3. [`./saved_results/election`](./saved_results/election)

Where each folder contains a `test.json` and `val.json` files that represent the test and validation results for the associated dataset.

To create the anonymised results run the following python script, which takes the results and other data from the associated dataset folders within `./data` and stores them in the `./saved_results` folder in the correct format:
``` bash
python anonymise_dataset_folder.py ./data ./saved_results
```





