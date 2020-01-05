import argparse
from pathlib import Path
import tempfile
from typing import Optional

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
    if 'context_field_embedder' not in config_params['model']:
        config_params['model']['context_field_embedder'] = {'tokens': embedding_dict}
    else:
        config_params['model']['context_field_embedder']['tokens'] = embedding_dict
    return config_params

def add_elmo_embedding(config_params: Params, pre_trained_fp: Path) -> Params:
    pre_trained_path = str(pre_trained_fp.resolve())
    embedding_dict = {"type": "bidirectional_lm_token_embedder",
                      "archive_file": f"{pre_trained_path}",
                      "bos_eos_tokens": ["<S>", "</S>"],
                      "remove_bos_eos": True,
                      "requires_grad": False}
    if 'context_field_embedder' not in config_params['model']:
        config_params['model']['context_field_embedder'] = {'elmo': embedding_dict}
    else:
        config_params['model']['context_field_embedder']['elmo'] = embedding_dict
    return config_params

def model_specific_rep_params(config_params: Params, model_name: str, 
                              word_rep_dim: int) -> Params:
    if 'target_encoder' in config_params['model']:
        target_params = config_params['model']['target_encoder']
        if 'hidden_size' in target_params:
            target_encoder_out = target_params['hidden_size']
    if model_name == 'IAN':
        config_params['model']['context_encoder']['input_size'] = word_rep_dim
        config_params['model']['target_encoder']['input_size'] = word_rep_dim
    if model_name == 'TDLSTM':
        config_params['model']['left_text_encoder']['input_size'] = word_rep_dim
        config_params['model']['right_text_encoder']['input_size'] = word_rep_dim
    if model_name == 'InterAE':
        config_params['model']['context_encoder']['input_size'] = word_rep_dim + target_encoder_out
        config_params['model']['target_encoder']['input_size'] = word_rep_dim
    if model_name == 'AE':
        config_params['model']['context_encoder']['input_size'] = word_rep_dim + target_encoder_out
        config_params['model']['target_encoder']['input_size'] = word_rep_dim
    return config_params

def add_inter_aspect(config_params: Params, model_name: str, 
                     inter_aspect_encoder: str):
    allowed_inter_aspect_encoders = ['sequential']
    if inter_aspect_encoder == 'sequential':
        model_input_size = 600
        if model_name == 'AE':
            model_input_size = 300
        inter_params = {"type": "sequence_inter_target",
                        "sequence_encoder": 
                        { "type": "lstm", "input_size": model_input_size,
                          "hidden_size": 300, "bidirectional": False,
                          "num_layers": 1}}
        config_params['model']['inter_target_encoding'] = inter_params
    else:
        raise ValueError('The inter aspect encoder has to be one of the '
                         f'following {allowed_inter_aspect_encoders}\n'
                         f'and not {inter_aspect_encoder}')

def add_position(config_params: Params, model_name: str, 
                 position_model_type: str):
    if position_model_type == 'Weighting':
        config_params['dataset_reader']['position_weights'] = True
        config_params['model']['target_position_weight'] = {"type": "relative_target_position_weight"}
    elif position_model_type == 'Embedding':
        config_params['dataset_reader']['position_embeddings'] = True
        position_embedding = {'position_tokens': {'type': 'embedding', 'embedding_dim': 30, 'trainable': True}}
        config_params['model']['target_position_embedding'] = position_embedding
        context_encoder_input_dim = config_params['model']['context_encoder']['input_size']
        context_encoder_input_dim += 30
        config_params['model']['context_encoder']['input_size'] = context_encoder_input_dim
    else:
        raise ValueError('Can only add position weighting or embedding')

def run_model(train_fp: Path, val_fp: Path, test_fp: Path, 
              config_fp: Path, number_runs: int, prediction_key: str,
              save_dir: Optional[Path] = None, 
              only_produce_model: bool = False) -> None:
    '''
    :param train_fp: Path to file that contains JSON formatted training data
    :param val_fp: Path to file that contains JSON formatted validation data
    :param test_fp: Path to file that contains JSON formatted testing data
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
    val_data = TargetTextCollection.load_json(val_fp)
    test_data = TargetTextCollection.load_json(test_fp)

    key_mappings = {'sentiments': prediction_key}

    if only_produce_model:
        model = AllenNLPModel('model', config_fp, save_dir=save_dir, 
                              predictor_name='target-sentiment')
        model.fit(train_data, val_data, test_data)

    for run in range(predictions_left):
        print(f'Run number {run}')
        if run == 0 and predictions_left == number_runs:
            model = AllenNLPModel('model', config_fp, save_dir=save_dir,
                                  predictor_name='target-sentiment')
        else:
            model = AllenNLPModel('model', config_fp, 
                                  predictor_name='target-sentiment')
        model.fit(train_data, val_data, test_data)
        model.predict_into_collection(val_data, key_mapping=key_mappings)
        model.predict_into_collection(test_data, key_mapping=key_mappings)
    val_data.to_json_file(val_fp, include_metadata=True)
    test_data.to_json_file(test_fp, include_metadata=True)

if __name__=='__main__':
    model_save_fp_help = "File Path to save the model. NOTE if N>1 then the "\
                         "first model will be saved. This needs to be a directory"\
                         " that does not exist currently."
    run_to_get_saved_model_help = ('This is to be used if you want to create a '
                                   'saved model without saving the predictions')
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=parse_path, 
                        help='File Path to directory that contains the train, val, and test data')
    parser.add_argument("config_fp", type=parse_path, 
                        help='File Path to the models config file')
    parser.add_argument("N", type=int, 
                        help='Number of times to run the model')
    parser.add_argument("domain", type=str, choices=['Laptop', 'Restaurant', 'Election'])
    parser.add_argument("model_name", type=str, choices=['IAN', 'TDLSTM', 'AE'])
    parser.add_argument("model_dir", type=parse_path, 
                        help='Top level directory to save all of models to')
    parser.add_argument("model_save_name", type=str, 
                        help='Name to save the model to')
    parser.add_argument("--cwr", action='store_true')
    parser.add_argument("--glove", action='store_true')
    parser.add_argument("--position", type=str, choices=['Weighting', 'Embedding'])
    parser.add_argument("--inter_aspect", type=str, choices=['sequential'])
    parser.add_argument("--run_to_get_saved_model", action='store_true', 
                        help=run_to_get_saved_model_help)
    args = parser.parse_args()

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
    # Need to change model specific parameters based on the word representation
    # dimensions
    model_specific_rep_params(config_params, args.model_name, word_rep_dim)
    split_context_models = ['TDLSTM']
    if args.cwr and args.model_name not in split_context_models:
        config_params['model']['use_target_sequences'] = True
        config_params['dataset_reader']['target_sequences'] = True
    if args.cwr and args.model_name in split_context_models:
        config_params['iterator']['batch_size'] = 8
    # if inter aspect
    if args.inter_aspect:
        add_inter_aspect(config_params, args.model_name, args.inter_aspect)
    # position weighting
    if args.position:
        if args.model_name in split_context_models:
            raise ValueError('Cannot perform position weighting or embedding on '
                             f'this split context model {args.model_name} ')
        else:
            add_position(config_params, args.model_name, args.position)
    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model_dir_save_dir = Path(model_dir, args.model_save_name)

    
    position_name = 'None'
    if args.position:
        position_name = args.position
    inter_aspect_name = 'None'
    if args.inter_aspect:
        inter_aspect_name = args.inter_aspect
    
    prediction_key = f'predicted_target_sentiment_{args.model_name}_{embedding_name}_{position_name}_{inter_aspect_name}'
    train_fp = Path(args.dataset_dir, 'train.json')
    val_fp = Path(args.dataset_dir, 'val.json')
    test_fp = Path(args.dataset_dir, 'test.json')
    # Check if we have all the predictions required
    test_collection = TargetTextCollection.load_json(test_fp)
    test_value = next(test_collection.dict_iterator())
    with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
        config_params.to_file(temp_file.name)
        only_produce_model = True if args.run_to_get_saved_model else False
        run_model(train_fp, val_fp, test_fp, Path(temp_file.name), 
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
        key_metadata['Position'] = args.position if args.position else False
        key_metadata['Inter-Aspect'] = args.inter_aspect if args.inter_aspect else False
        key_metadata['Model'] = args.model_name
        sentiment_key_metadata[prediction_key] = key_metadata
        metadata['predicted_target_sentiment_key'] = sentiment_key_metadata
        data.to_json_file(data_fp, include_metadata=True)