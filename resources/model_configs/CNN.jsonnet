{
    "dataset_reader": {
      "type": "text_sentiment",
      "label_name": "text_sentiment"
    },
    "model": {
      "type": "basic_classifier",
      "dropout": 0.5,
      "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
      "seq2vec_encoder": {
        "type": "cnn",
        "ngram_filter_sizes": [3,4,5],
        "num_filters": 100
      }
    },
    "iterator": {
      "type": "basic",
      "batch_size": 32
    },
    "trainer": {
      "optimizer": {
        "type": "adam"
      },
      "shuffle": true,
      "patience": 10,
      "num_epochs": 200,
      "cuda_device": 0,
      "validation_metric": "+accuracy"
    }
  }