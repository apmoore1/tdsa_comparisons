{
  "dataset_reader": {
    "type": "target_sentiment"
  },
  "model": {
    "type": "atae_classifier",
    "dropout": 0.5,
    "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
    "AE": true,
    "AttentionAE": false,
    "context_encoder": {
      "type": "lstm",
      "input_size": 600,
      "hidden_size": 300,
      "bidirectional": false,
      "num_layers": 1
    },
    "target_encoder": {
        "type": "lstm",
        "input_size": 300,
        "hidden_size": 300,
        "bidirectional": false,
        "num_layers": 1
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