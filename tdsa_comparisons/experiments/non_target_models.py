import argparse
from collections import Counter
from pathlib import Path
from typing import Optional
import tempfile

from allennlp.common.params import Params
from target_extraction.data_types import TargetTextCollection
from target_extraction.allen.allennlp_model import AllenNLPModel

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def add_cwr_indexer(config_params: Params) -> Params:
    elmo_indexer = {"elmo": {"type": "elmo_characters", "token_min_padding_length": 1}}
    if 'token_indexers' not in config_params['dataset_reader']:
        config_params['dataset_reader']['token_indexers'] = elmo_indexer
    else:
        config_params['dataset_reader']['token_indexers']['elmo'] = elmo_indexer['elmo']
    return config_params

def add_token_indexer(config_params: Params) -> Params:
    token_indexer = {"tokens": {"type": "single_id", "lowercase_tokens": True,
                                "token_min_padding_length": 1}}
    if 'token_indexers' not in config_params['dataset_reader']:
        config_params['dataset_reader']['token_indexers'] = token_indexer
    else:
        config_params['dataset_reader']['token_indexers']['tokens'] = token_indexer['tokens']
    return config_params 

def add_word_embedding(config_params: Params, pre_trained_fp: Path) -> Params:
    pre_trained_path = str(pre_trained_fp.resolve())
    embedding_dict = {"type": "embedding", "embedding_dim": 300, 
                      "pretrained_file": f"{pre_trained_path}",
                      "trainable": False}
    if 'text_field_embedder' not in config_params['model']:
        config_params['model']['text_field_embedder'] = {'tokens': embedding_dict}
    else:
        config_params['model']['text_field_embedder']['tokens'] = embedding_dict
    return config_params

def add_elmo_embedding(config_params: Params, pre_trained_fp: Path) -> Params:
    pre_trained_path = str(pre_trained_fp.resolve())
    embedding_dict = {"type": "bidirectional_lm_token_embedder",
                      "archive_file": f"{pre_trained_path}",
                      "bos_eos_tokens": ["<S>", "</S>"],
                      "remove_bos_eos": True,
                      "requires_grad": False}
    if 'text_field_embedder' not in config_params['model']:
        config_params['model']['text_field_embedder'] = {'elmo': embedding_dict}
    else:
        config_params['model']['text_field_embedder']['elmo'] = embedding_dict
    return config_params

def text_classification_prediction(model: AllenNLPModel, 
                                   dataset: TargetTextCollection, 
                                   prediction_key: str) -> None:
    for value in model._predict_iter(dataset.dict_iterator(), 
                                     yield_original_target=True):
        prediction_object, target_object = value
        predicted_sentiment = prediction_object['label']
        true_sentiment = target_object['target_sentiments']
        number_sentiments = len(true_sentiment)
        predicted_sentiment = [predicted_sentiment] * number_sentiments
        text_id = target_object['text_id']
        if prediction_key not in dataset[text_id]:
            dataset[text_id][prediction_key] = []
        dataset[text_id][prediction_key].append(predicted_sentiment)


def run_model(train_fp: Path, train_val_fp: Path, val_fp: Path, test_fp: Path, 
              config_fp: Path, number_runs: int, prediction_key: str,
              save_dir: Optional[Path] = None, 
              only_produce_model: bool = False) -> None:
    '''
    :param train_fp: Path to file that contains JSON formatted training data
    :param train_val_fp: Path to file that contains JSON formatted validation data
                         that will be used to train the model (used for early 
                         stopping).
    :param val_fp: Path to file that contains JSON formatted validation data 
                   that will be predicted on and used to evaluate the text 
                   classification model on the target sentiment task
    :param test_fp: Path to file that contains JSON formatted testing data
                    for target sentiment evaluation.
    :param config_fp: Path to file that contains the models configuration
    :param number_runs: Number of times to run the model
    :param prediction_key: The key to save the predictions to within the 
                           validation and test data
    :param save_dir: Path to save the model to.
    :param only_produce_model: Whether or not to run the model after all the 
                               predictions have been made to save a model to 
                               the `save_dir`
    '''
    # Test if all the predictions have already been made
    temp_test_data = TargetTextCollection.load_json(test_fp)
    temp_test_value = next(temp_test_data.dict_iterator())
    predictions_left = number_runs
    if prediction_key in temp_test_value:
        number_runs_done = len(temp_test_value[prediction_key])
        predictions_left = predictions_left - number_runs_done
        if number_runs_done >= number_runs and not only_produce_model:
            print('Predictions have already been made')
            return
    train_data = TargetTextCollection.load_json(train_fp)
    train_val_data = TargetTextCollection.load_json(train_val_fp)
    val_data = TargetTextCollection.load_json(val_fp)
    test_data = TargetTextCollection.load_json(test_fp)

    if only_produce_model:
        model = AllenNLPModel('model', config_fp, save_dir=save_dir, 
                              predictor_name='target-sentiment')
        model.fit(train_data, train_val_data, test_data)

    for run in range(predictions_left):
        print(f'Run number {run}')
        if run == 0 and predictions_left == number_runs:
            model = AllenNLPModel('model', config_fp, save_dir=save_dir,
                                  predictor_name='target-sentiment')
        else:
            model = AllenNLPModel('model', config_fp, 
                                  predictor_name='target-sentiment')
        model.fit(train_data, train_val_data, test_data)
        text_classification_prediction(model, val_data, prediction_key)
        text_classification_prediction(model, test_data, prediction_key)
        #for value in model._predict_iter(val_data.dict_iterator(), yield_original_target=True):
        #    prediction_object, target_object = value
        #    predicted_sentiment = prediction_object['label']
        #    true_sentiment = target_object['target_sentiments']
        #    number_sentiments = len(true_sentiment)
        #    predicted_sentiment = [predicted_sentiment] * number_sentiments
        #    text_id = target_object['text_id']
        #    if prediction_key not in val_data[text_id]:
        #        val_data[text_id][prediction_key] = []
        #    val_data[text_id][prediction_key].append(predicted_sentiment)
        #model.predict_into_collection(val_data, key_mapping=key_mappings)
        #model.predict_into_collection(test_data, key_mapping=key_mappings)
    val_data.to_json_file(val_fp, include_metadata=True)
    test_data.to_json_file(test_fp, include_metadata=True)

if __name__=='__main__':
    dataset_dir_help = ('File Path to directory that contains the train, '
                        'val, and test data')
    average_help = ("If False it only trains on texts that have one unique "
                    "sentiment within the text else it finds the average sentiment "
                    "in the multi sentiment sentences")
    model_save_fp_help = ("File Path to save the model. NOTE if N>1 then the "
                          "first model will be saved. This needs to be a directory"
                          " that does not exist currently.")
    run_to_get_saved_model_help = ('This is to be used if you want to create a '
                                   'saved model without saving the predictions')
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=parse_path, 
                        help=dataset_dir_help)
    parser.add_argument("config_fp", type=parse_path, 
                        help='File Path to the models config file')
    parser.add_argument("N", type=int, 
                        help='Number of times to run the model')
    parser.add_argument("domain", type=str, choices=['Laptop', 'Restaurant', 'Election'])
    parser.add_argument("model_name", type=str, choices=['CNN'])
    parser.add_argument("model_dir", type=parse_path, 
                        help='Top level directory to save all of models to')
    parser.add_argument("model_save_name", type=str, 
                        help='Name to save the model to')
    parser.add_argument("--cwr", action='store_true')
    parser.add_argument("--glove", action='store_true')
    parser.add_argument("--position", type=str)
    parser.add_argument("--inter_aspect", type=str, choices=['sequential'])
    parser.add_argument("--run_to_get_saved_model", action='store_true', 
                        help=run_to_get_saved_model_help)
    parser.add_argument("--average", action="store_true", help=average_help)
    args = parser.parse_args()

    average_sentiment = True if args.average else False

    config_params = Params.from_file(args.config_fp)

    embedding_name = 'GloVe'
    # Add relevant token indexers
    if args.cwr:
        add_cwr_indexer(config_params)
        embedding_name = 'CWR' 
    if args.glove:
        add_token_indexer(config_params)
    # Add relevant word embedding contexts
    embedding_dir = Path('.', 'resources', 'embeddings').resolve()
    if args.glove:
        glove_embedding_fp = Path(embedding_dir, 'glove.840B.300d.txt')
        add_word_embedding(config_params, glove_embedding_fp)
    # Add relevant CWR
    cwr_dir = Path('.', 'resources', 'CWR').resolve()
    if args.cwr:
        cwr_fp = Path(cwr_dir, f'{args.domain.lower()}_model.tar.gz')
        add_elmo_embedding(config_params, cwr_fp)
    word_rep_dim = 0
    if args.cwr:
        word_rep_dim += 1024
    if args.glove:
        word_rep_dim += 300

    config_params['model']['seq2vec_encoder']['embedding_dim'] = word_rep_dim
    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model_dir_save_dir = Path(model_dir, args.model_save_name)

    position_name = 'None'
    if args.position:
        position_name = args.position
    inter_aspect_name = 'None'
    if args.inter_aspect:
        inter_aspect_name = args.inter_aspect
    
    data_trained_on_name = 'single'
    if average_sentiment:
        data_trained_on_name = 'average'
    
    prediction_key = f'predicted_target_sentiment_{data_trained_on_name}_{embedding_name}'
    train_fp = Path(args.dataset_dir, 'train.json')
    train_val_fp = Path(args.dataset_dir, 'train_val.json')
    val_fp = Path(args.dataset_dir, 'val.json')
    test_fp = Path(args.dataset_dir, 'test.json')
    with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
        config_params.to_file(temp_file.name)
        only_produce_model = True if args.run_to_get_saved_model else False
        run_model(train_fp, train_val_fp, val_fp, test_fp, Path(temp_file.name), 
                  args.N, prediction_key, save_dir=model_dir_save_dir,
                  only_produce_model=only_produce_model)
    
    # Add metadata and ensure that the datasets have there correct name 
    # associated to them
    for split_name, data_fp in [('Validation', val_fp), ('Test', test_fp)]:
        data = TargetTextCollection.load_json(data_fp, name=args.domain)
        metadata = data.metadata if data.metadata else {}
        metadata['split'] = split_name
        sentiment_key_metadata = {}
        sentiment_key_metadata_key = 'predicted_target_sentiment_key'
        if 'predicted_target_sentiment_key' in metadata:
            sentiment_key_metadata = metadata['predicted_target_sentiment_key']
        key_metadata = {}
        key_metadata['CWR'] = True if args.cwr else False
        key_metadata['Position'] = True if args.position else False
        key_metadata['Inter-Aspect'] = args.inter_aspect if args.inter_aspect else False
        key_metadata['Model'] = args.model_name
        key_metadata['data-trained-on'] = data_trained_on_name
        sentiment_key_metadata[prediction_key] = key_metadata
        metadata['predicted_target_sentiment_key'] = sentiment_key_metadata
        data.to_json_file(data_fp, include_metadata=True)

    