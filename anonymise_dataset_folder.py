import argparse
from pathlib import Path

from target_extraction.data_types import TargetTextCollection

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    save_dir_help = ('File Path to directory where the anonymised results '
                     'will be saved.')
    results_dir_help = ('File path to the directory that currently stores all '
                        'results')
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=parse_path, help=results_dir_help)
    parser.add_argument("save_dir", type=parse_path, help=save_dir_help)
    args = parser.parse_args()

    save_dir = args.save_dir
    results_dir = args.results_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = ['election', 'laptop', 'restaurant']
    split_names = ['train', 'val', 'test']
    for dataset_name in dataset_names:
        dataset_result_folder = Path(results_dir, f'{dataset_name}_dataset')
        save_dataset_folder = Path(save_dir, f'{dataset_name}_dataset')
        save_dataset_folder.mkdir(parents=True, exist_ok=True)
        for split_name in split_names:
            split_fp = Path(dataset_result_folder, f'{split_name}.json')
            split_dataset = TargetTextCollection.load_json(split_fp)
            split_dataset: TargetTextCollection
            split_dataset.anonymised = True
            save_fp = Path(save_dataset_folder, f'{split_name}.json')
            split_dataset.to_json_file(save_fp, include_metadata=True)