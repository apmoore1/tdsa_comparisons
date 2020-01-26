from pathlib import Path

import pandas as pd
from target_extraction.data_types import TargetTextCollection
from target_extraction.analysis.sentiment_error_analysis import (ERROR_SPLIT_SUBSET_NAMES,
                                                                 error_split_df, PLOT_SUBSET_ABBREVIATION,
                                                                 error_analysis_wrapper, 
                                                                 subset_name_to_error_split)
from target_extraction.analysis.sentiment_metrics import accuracy
from target_extraction.analysis.util import add_metadata_to_df

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

    tssr_func = error_analysis_wrapper('TSSR')
    ds_func = error_analysis_wrapper('DS')
    nt_func = error_analysis_wrapper('NT')
    all_relevant_error_funcs = [tssr_func, ds_func, nt_func]

    splits = ['test', 'val']
    dataset_names = ['Laptop', 'Restaurant', 'Election']

    all_dfs = []
    relevant_prediction_keys = ['predicted_target_sentiment_AE_GloVe_None_None',
                                'predicted_target_sentiment_CNN_GloVe_None_None',
                                'predicted_target_sentiment_IAN_GloVe_None_None',
                                'predicted_target_sentiment_TDLSTM_GloVe_None_None']
    nt_error_names = ERROR_SPLIT_SUBSET_NAMES['NT']
    ds_error_names = ERROR_SPLIT_SUBSET_NAMES['DS']
    tssr_error_names = ERROR_SPLIT_SUBSET_NAMES['TSSR']
    reduced_collection_subset_names = ds_error_names + tssr_error_names
    nt_split_subsets = {'NT': ERROR_SPLIT_SUBSET_NAMES['NT']}

    import time
    overall_time = time.time()
    for dataset_name in dataset_names:
        print(f'Dataset {dataset_name}')
        for split in splits:
            one_time = time.time()
            print(f'Data Split {split}')
            data_fp = Path(results_dir, f'{dataset_name.lower()}_dataset', 
                           f'{split}.json')
            dataset = TargetTextCollection.load_json(data_fp)
            for error_func in all_relevant_error_funcs:
                error_func(None, dataset, True)
            for reduced_collection_subset_name in reduced_collection_subset_names:
                temp_df = error_split_df(None, dataset, relevant_prediction_keys, 
                                            'target_sentiments', nt_split_subsets, accuracy,
                                            {'ignore_label_differences': True}, 
                                            include_dataset_size=True,
                                            collection_subsetting=[[reduced_collection_subset_name]],
                                            table_format_return=False)
                temp_df = add_metadata_to_df(temp_df, dataset, 
                                            'predicted_target_sentiment_key')
                temp_df['reduced collection subset'] = reduced_collection_subset_name
                temp_df['Dataset'] = dataset_name
                temp_df['Split'] = split.capitalize()
                all_dfs.append(temp_df)
            print(time.time() - one_time)
    
    print(f'total time {time.time() - overall_time}')
    all_dfs = pd.concat(all_dfs, 0, ignore_index=True)
    temp_dfs = all_dfs.copy(deep=True)
    all_subset_names = [name for subset_names in ERROR_SPLIT_SUBSET_NAMES.values() 
                        for name in subset_names]
    temp_dfs['Reduced Error Split'] = temp_dfs.apply(lambda x: subset_name_to_error_split(x['reduced collection subset']), 1)
    temp_dfs['Metric'] = temp_dfs['Metric'] * 100
    temp_dfs = temp_dfs.rename(columns={'Metric': 'Accuracy'})
    temp_dfs['NT'] = temp_dfs.apply(lambda x: PLOT_SUBSET_ABBREVIATION[x['subset names']], 1)
    temp_dfs['Subset By'] = temp_dfs.apply(lambda x: PLOT_SUBSET_ABBREVIATION[x['reduced collection subset']], 1)
    temp_dfs = temp_dfs.drop(columns=['reduced collection subset', 'subset names'])
    temp_dfs.to_csv(save_fp, sep='\t')