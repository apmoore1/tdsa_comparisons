from collections import defaultdict
import argparse
from pathlib import Path

from target_extraction.data_types import TargetTextCollection

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    '''
    This will add the Text CNN classifier results and meta data from the 
    `./data/text_classification/average` directory and add it to each dataset 
    and splits result in `./data` when these are two arguments to this script.

    The CNN text classifier results will be saved in the following key:
    `predicted_target_sentiment_CNN_GloVe_None_None` and the meta data 
    associated to this key will be the following:
    {"predicted_target_sentiment_CNN_GloVe_None_None": 
      {"CWR": false, "Position": false, 
      "Inter-Aspect": false, "Model": "CNN"}
    } 
    '''
    new_data_dir_help = ('Directory that contains the data to merge with the'
                     ' existing directory')
    existing_data_dir_help = ('Directory that contains the data that is to'
                              ' contain the new data')
    parser = argparse.ArgumentParser()
    parser.add_argument("new_data_dir", type=parse_path, help=new_data_dir_help)
    parser.add_argument("existing_data_dir", type=parse_path, 
                        help=existing_data_dir_help)
    args = parser.parse_args()

    new_data_dir = args.new_data_dir
    existing_data_dir = args.existing_data_dir

    dataset_names = ['election', 'laptop', 'restaurant']
    split_names = ['val', 'test']
    prediction_keys_to_copy = ['predicted_target_sentiment_average_GloVe', 
                               'predicted_target_sentiment_average_CWR']
    prediction_key_rename = {'predicted_target_sentiment_average_GloVe': 'predicted_target_sentiment_CNN_GloVe_None_None',
                             'predicted_target_sentiment_average_CWR': 'predicted_target_sentiment_CNN_CWR_None_None'}
    for dataset_name in dataset_names:
        new_dataset_result_folder = Path(new_data_dir, f'{dataset_name}_dataset')
        existing_dataset_result_folder = Path(existing_data_dir, f'{dataset_name}_dataset')

        for split_name in split_names:
            new_split_data = Path(new_dataset_result_folder, f'{split_name}.json')
            new_target_collection = TargetTextCollection.load_json(new_split_data)
            exist_split_data = Path(existing_dataset_result_folder, f'{split_name}.json')
            exist_target_collection = TargetTextCollection.load_json(exist_split_data)

            id_results = defaultdict(dict)
            for text_id, target_text in new_target_collection.items():
                for prediction_key in prediction_keys_to_copy:
                    cnn_result = target_text[prediction_key]
                    id_results[text_id][prediction_key] = cnn_result

            len_err = ('The number of keys that the merge predictions are'
                       f' associated with {len(id_results)} are not the same '
                       'number of keys as the existing predictions dataset '
                       f'contains {len(exist_target_collection)}.')
            assert len(id_results) == len(exist_target_collection), len_err

            for text_id, key_cnn_predictions in id_results.items():
                for prediction_key, cnn_predictions in key_cnn_predictions.items():
                    new_key_name = prediction_key_rename[prediction_key]
                    exist_target_collection[text_id][new_key_name] = cnn_predictions
            
            for prediction_key in prediction_keys_to_copy:
                related_metadata = new_target_collection.metadata['predicted_target_sentiment_key'][prediction_key]
                del related_metadata['data-trained-on']
                new_key_name = prediction_key_rename[prediction_key]
                exist_target_collection.metadata['predicted_target_sentiment_key'][new_key_name] = related_metadata
            exist_target_collection.to_json_file(exist_split_data, include_metadata=True)