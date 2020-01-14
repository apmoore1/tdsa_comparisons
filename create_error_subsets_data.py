from pathlib import Path
from typing import List

import pandas as pd
from target_extraction.data_types import TargetTextCollection
from target_extraction.analysis.util import overall_metric_results
from target_extraction.analysis.sentiment_error_analysis import (ERROR_SPLIT_SUBSET_NAMES,
                                                                 error_split_df)
from target_extraction.analysis import sentiment_metrics

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=parse_path, 
                        help='Directory that contain results for each dataset')
    parser.add_argument('save_fp', type=parse_path, 
                        help='File path to save the results too')
    args = parser.parse_args()
    results_dir = args.results_dir
    save_fp = args.save_fp
    # Get the data
    data_splits = ['test', 'val']
    dataset_names = ['election', 'laptop', 'restaurant']
    index_keys = ['prediction key', 'run number']

    all_dfs: List[pd.DataFrame] = []
    training_datasets = {}
    for dataset_name in dataset_names:
        data_fp = Path(results_dir, f'{dataset_name}_dataset', 'train.json')
        training_datasets[dataset_name] = TargetTextCollection.load_json(data_fp)

    formatted_data_split = {'test': 'Test', 'val': 'Validation'}

    import time
    overall_time = time.time()
    for data_split in data_splits:
        print(f'Data Split {data_split}')
        for dataset_name in dataset_names:
            one_time = time.time()
            print(f'Dataset {dataset_name}')
            data_fp = Path(results_dir, f'{dataset_name}_dataset', 
                        f'{data_split}.json')
            test_collection = TargetTextCollection.load_json(data_fp)
            metric_df = overall_metric_results(test_collection, 
                                            true_sentiment_key='target_sentiments')
            prediction_keys = metric_df['prediction key'].unique().tolist() 
            metric_df = metric_df.reset_index().set_index(index_keys)
            
            train_collection = training_datasets[dataset_name]
            subset_metric_df = error_split_df(train_collection, test_collection, 
                                              prediction_keys, 'target_sentiments', 
                                              ERROR_SPLIT_SUBSET_NAMES, 
                                              sentiment_metrics.accuracy, 
                                              assert_number_labels=3)
            subset_metric_df = subset_metric_df.reset_index().set_index(index_keys)
            # Combine the overall metrics with the subset metrics
            metric_df = pd.concat([metric_df, subset_metric_df], axis=1, sort=False)
            metric_df = metric_df.reset_index()
            metric_df['Dataset'] = dataset_name.capitalize()
            metric_df['Data Split'] = formatted_data_split[data_split]
            all_dfs.append(metric_df)
            print(time.time() - one_time)
    print(f'total time {time.time() - overall_time}')

    all_df_results = pd.concat(all_dfs, axis=0, sort=False, ignore_index=True)
    all_df_results.to_csv(save_fp, sep='\t')