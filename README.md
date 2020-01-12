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

The CWR that are Transformer ELMo models are stored in the directory `./resources/CWR/`, of which each dataset has their own CWR due to [Rietzler et al. 2019](https://arxiv.org/abs/1908.11860) showing that domain specific CWR outperform non-domain specific CWR by a large margin.

## Experiments
In all of the experiments we are going to use the following 3 TDSA models:
1. [**TDLSTM**](https://www.aclweb.org/anthology/C16-1311.pdf)
2. [**IAN**](https://www.ijcai.org/proceedings/2017/0568.pdf)
3. **AE** (in the thesis/paper it is called *AE-Attention*) -- A model that is the same as the *AE* model from [Wang et al. 2016](https://www.aclweb.org/anthology/D16-1058.pdf) but with an attention layer after the LSTM enocder. This model is also the same as the inter-aspect model (from now on called **Inter-AE**) from [Hazarika et al. 2018](https://www.aclweb.org/anthology/N18-2043/) but without the LSTM aspect encoder (phase 2 in figure 1) that models other targets from the same context/text.

The 4 main experiments are the following:
1. Baseline -- TDLSTM, IAN, and AE as is without modification using the GloVe word vectors. Also included in these experiments is a baseline CNN text classifier which does not take into account the target.   
2. Inter-Aspect -- TDLSTM, IAN, and InterAE with Inter-Aspect modelling. To incorporate Inter Aspect modeling we will adopt the method of [Hazarika et al. 2018](https://www.aclweb.org/anthology/N18-2043/) which uses an LSTM hence why the AE model is now called the InterAE model.
3. Position -- Run the IAN, and AE models with position of the target encoded. TDLSTM is not used in this experiment as the model already encodes position information via the network architecture.
4. CWR -- Replace the GloVe vectors with domain specific CWR for TDLSTM, IAN, and AE.

All of the default training configurations for each of the 3 models can be found [here](./resources/model_configs/).

Furthermore all the experiments ran within the experiment section all use bash scripts of which these scripts all take two arguments:
1. The number of times to run each model. In all of experiments this is `8`
2. The directory to store a saved model of each model ran on each dataset.

### Baseline Text classification experiments
Before performing all of the experiments on the Target based models we want to set a benchmark on these datasets using standard text classification models that have no knowledge of the target. In these experiments we have one CNN based model from [Kim 2014](https://www.aclweb.org/anthology/D14-1181/) which takes as input word embeddings and then passes those through 3 filters (3, 4, and 5 window filter) each with a filter map of 100. This model is going to have two versions:
1. Trained on all of the sentences from the TDSA datasets where the sentiment for sentences with multiple targets and sentiments is going to be associated with the most frequent sentiment (ties decided by random choice).
2. Trained on only sentences from the TDSA datasets where the sentence has only one sentiment associated with it.

The two versions from now on will be called *CNN(average)* and *CNN(single)* respectively. The metadata associated from the results of these baselines will be the same as those from the Target based methods, of which the metadata is described in the [results section](#anonymise-the-results). The only extra metadata add for these experiments is the following within the `predicted_target_sentiment_key` dictionary: `data-trained-on` which can only have two values `single` and `average` this is to represent the two different model versions *CNN(single)* and *CNN(average)*.

Before training these two model versions we need to create two new training and validation datasets based on the different sentiment labels (single and average). To do this easily we create new data directories for each of the datasets (Election, Restaurant, and Laptop). Of which these data directories can be found in `./data/text_classification/single` and `./data/text_classification/average` for the single and average sentiment labels respectively. To create these data directories run the following bash script:
``` bash
./tdsa_comparisons/splitting_data/text_classification_dataset_creator.sh
```

This bash script if ran multiple times will not change the data but will provide you with the data statistics for the training and validation datasets text sentiment label. These dataset statistics are better shown for the training dataset through this [notebook](./analysis/Baseline_non_target_results.ipynb). The main difference with these dataset directories is that they will have an extra validation dataset called `train_val.json` which will be used for early stopping for the text classification models. However when predicting for the TDSA task this will be done like all of the other experiments on the `val.json` and `test.json` TDSA data. 

Run the following bash script to re-create the CNN text classification experiments: 

``` bash
./tdsa_comparisons/experiments/non_target_baseline.sh 8 ./saved_models/
```

The results from the CNN text classification can be seen through the associated [notebook](./analysis/Baseline_non_target_results.ipynb) of which it was found that the *CNN (average)* was better on 2 (3) of the 3 datasets for the accuracy (macro f1) metric across both validation and test splits. Thus the *CNN (average)* will be used as the baseline to compare against the TDSA methods.

### Baseline Experiments

To run all 3 TDSA methods for the baseline experiments on the 3 datasets run the following bash script:

``` bash
./tdsa_comparisons/experiments/baseline.sh 8 ./saved_models/
```

### Inter-Aspect Experiments

To run all 3 TDSA methods for the inter-aspect experiments on the 3 datasets run the following bash script:

``` bash
./tdsa_comparisons/experiments/sequential_inter_aspect.sh 8 ./saved_models/
```

### Position Embedding Experiments

To run the 2 TDSA methods for the position embedding experiments on the 3 datasets run the following:

``` bash
./tdsa_comparisons/experiments/position_embeddings.sh 8 ./saved_models/
```

### Position Weighting and Inter-Aspect Experiments
To run the 2 TDSA methods for the position embedding experiments on the 3 datasets run the following:

``` bash
./tdsa_comparisons/experiments/sequential_inter_aspect_and_position_weighting.sh 8 ./saved_models/
```

### CWR Experiments

To run the 3 TDSA methods for the CWR experiments on the 3 datasets run the following:
``` bash
./tdsa_comparisons/experiments/cwr.sh 8 ./saved_models/
```

We also ran the *CNN (average)* model with CWR to see the difference between the TDSA and Text classification methods when CWR are used. To run the *CNN* model:
``` bash
./tdsa_comparisons/experiments/non_target_cwr_baseline.sh 8 ./saved_models/
```
The results are saved to their own directory `./data/text_classification/average`, later we show how to merge these results with the TDSA results.
{'metadata':{'predictions':{'target_sentiment_{word_vector}_{position}_{}}}

## Results
In the results section the predictions generated from all of the experiments are examined through their respective notebooks which can be found in the [analysis folder.](./analysis). The predictions from the experiments are saved to the original data within the data folder (./data) and is then released annoymised (no text) within this github repository in the [saved results folder.](./saved_results) 

Thus before any analysis is performed we first annoymise the results so the analysis can be performed, we then explain how the predictions are stored and the metadata, and lastly we state which notebooks show what analysis.

### Anonymise the results
All of the results which have been some what anonymised (`text` data from the dataset is removed) from these models are released in `JSON` format nearly identical to there original format. The results from all of the experiments can be found in the following folders for the Target Based and best performing non-target baseline:
1. [`./saved_results/main/restaurant`](./saved_results/main/restaurant)
2. [`./saved_results/main/laptop`](./saved_results/main/laptop)
3. [`./saved_results/main/election`](./saved_results/main/election)

For the results between the two CNN version non-target baselines (*CNN(single)* and *CNN(average)*) they can be found within the following two folders both containing additional sub-folders for each dataset:
1. [`./saved_results/non_target_baselines/single`](./saved_results/non_target_baselines/single)
2. [`./saved_results/non_target_baselines/average`](./saved_results/non_target_baselines/average)

Where each dataset folder contains a `test.json`, `val.json`, and `train.json` files that represent the test and validation results for the associated dataset, as well as the training dataset so that error analysis can be performed.

Before the anonymisation we want to merge the results from the best performing CNN text classifier (*CNN (average)*) and the *CNN* using CWR with the TDSA results. This is done so that the analysis is easier. To do this run the following script before the anonymisation:
``` bash
python merge_text_and_tdsa_results.py ./data/text_classification/average ./data
```


To create the anonymised results for the Target Based and best performing non-target baselines run the following:
``` bash
python anonymise_dataset_folder.py ./data ./saved_results/main
```

To create the anonymised results for the CNN version non-target baselines run the following:
``` bash
python anonymise_dataset_folder.py ./data/text_classification/single ./saved_results/non_target_baselines/single
python anonymise_dataset_folder.py ./data/text_classification/average ./saved_results/non_target_baselines/average
```

### Result/Prediction Data and Metadata
Each validation and test `.json` result file (this is for all data within `./saved_results` and `./data`) contain metadata which is stored on the last line of the `.json` file of which the metadata contains the following keys:
1. `name` -- Name of the dataset in this case this is either `Laptop`, `Restaurant`, or `Election`
2. `split` -- The dataset split this is either `Validation` or `Test`
3. `predicted_target_sentiment_key` -- This contains a dictionary of dictionaries where each dictionary key links to a predicted sentiment key in each sample e.g. `predicted_target_sentiment_IAN_GloVe_None_None` this key then has a dictionary as value describing the model that generated those predictions. This dictionary has the following keys:
    * `CWR` -- If the model used Contextualised Word Representations, if `False` then GloVe vectors were used.
    * `Inter-Aspect` -- Whether the model toke into account inter aspect/target modelling if so then the name of this modelling would be the value else `False`. The valid names are the following `sequential` for [Hazarika et al. 2018](https://www.aclweb.org/anthology/N18-2043/) LSTM method.
    * `Position` -- Whether or not target position weighting or embedding were used if not this would be `False`. The valid names that can appear here are `Weighting` or `Embedding`
    * `Model` -- The name of the TDSA model used. Valid names that can appear are the following: `AE`, `TDLSTM`, and `IAN`

Example of this metadata is shown below:
```json
{"name": "Laptop", 
 "split": "Test", 
 "predicted_target_sentiment_key": 
   {"predicted_target_sentiment_IAN_GloVe_None_None": 
      {"CWR": false, "Position": false, 
      "Inter-Aspect": false, "Model": "IAN"}
   }
}
```

The metadata requires you to know the prediction key to find the associated metadata. The prediction keys do have a structure which is the following:
``` 
predicted_target_sentiment_$ Model Name $_$ Word Representation $_$ Position Encoding $_$ Inter Aspect Encoding $
``` 

Where the values within the dollar signs can be the following:
1. Model Name -- `IAN`, `AE`, `TDLSTM`, `CNN`
2. Word Representation -- `CWR` or `GloVe`.
3. Position Encoding -- `None`, `Weighted`, or `Embedding`. Where None represents no position encoding. 
4. Inter Aspect Encoding -- `None`, or `sequential`. Where None represents no inter aspect encoding.

Thus the following prediction key is for an `IAN` model that has been trained with GloVe and no position or inter apsect encodings: `predicted_target_sentiment_IAN_GloVe_None_None`

### Baseline Results
The following notebook shows the Baseline results



