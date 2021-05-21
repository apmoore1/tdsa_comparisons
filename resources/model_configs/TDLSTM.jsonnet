{
    "dataset_reader": {
      "type": "target_sentiment",
      "incl_target": true,
      "left_right_contexts": true,
      "reverse_right_context": true
    },
    "model": {
      "type": "split_contexts_classifier",
      "dropout": 0.5,
      "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
      "left_text_encoder": {
        "type": "lstm",
        "input_size": 300,
        "hidden_size": 300,
        "bidirectional": false,
        "num_layers": 1
      },
      "right_text_encoder": {
        "type": "lstm",
        "input_size": 300,
        "hidden_size": 300,
        "bidirectional": false,
        "num_layers": 1
      }
    },
    "iterator": {
      "type": "basic",
      "batch_size": 8
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