from pathlib import Path

from target_extraction.data_types import TargetTextCollection

data_dir = Path(__file__, '..', 'data').resolve()
dataset_names = ['election', 'laptop', 'restaurant']
split_names = ['train', 'val', 'test']

data_name_mapper = {'election': 'Election', 'laptop': 'Laptop', 
                    'restaurant': 'Restaurant'}
split_name_mapper = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}

all_dataset_names = []
all_split_names = []
targets_in_split = []

for dataset_name in dataset_names:
    dataset_dir = data_dir / f'{dataset_name}_dataset'
    total_targets_datasets = 0
    for split_name in split_names:
        data_fp = dataset_dir / f'{split_name}.json'
        total_targets_datasets += TargetTextCollection.load_json(data_fp).number_targets()
    for split_name in split_names:
        data_fp = dataset_dir / f'{split_name}.json'
        num_targets = TargetTextCollection.load_json(data_fp).number_targets()
        num_targets = f'{num_targets} ({(num_targets/total_targets_datasets)*100:.2f}%)'
        targets_in_split.append(num_targets)
        all_split_names.append(split_name_mapper[split_name])
        all_dataset_names.append(data_name_mapper[dataset_name])
    targets_in_split.append(total_targets_datasets)
    all_split_names.append('Total')
    all_dataset_names.append(data_name_mapper[dataset_name])
import pandas as pd
stats_df = pd.DataFrame({'Dataset': all_dataset_names, 'Data Split': all_split_names, 
                         'values': targets_in_split})
stats_df = pd.pivot_table(data=stats_df, index='Dataset', columns='Data Split', 
                          values='values', aggfunc=lambda x: x)
stats_df = stats_df.reindex(['Train', 'Validation', 'Test', 'Total'], axis=1)
print(stats_df)
