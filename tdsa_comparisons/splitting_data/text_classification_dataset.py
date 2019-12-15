import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

from target_extraction.data_types import TargetTextCollection

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':
    dataset_dir_help = ('File Path to directory that contains the train, '
                        'val, and test data')
    average_help = ("If False it creates a training dataset that only contains"
                    " texts that have one unique sentiment within it else it "
                    "finds the average sentiment in the multi sentiment sentences")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=parse_path, 
                        help=dataset_dir_help)
    parser.add_argument("--save_dir", type=parse_path, 
                        help="Directory to save the dataset to")
    parser.add_argument("--average", action="store_true", help=average_help)
    args = parser.parse_args()

    average_sentiment = True if args.average else False
    save_dir = args.save_dir
    if save_dir.exists():
        save_dir = False
        print('Not overwriting the data')

    # Process the training, validation, and test data
    split_names = ['train', 'val', 'test']
    split_name_dataset = []
    train_size = 0
    text_sentiment_distribution = Counter()
    target_sentiment_distribution = Counter()
    for split_name in split_names:
        data_fp = Path(args.dataset_dir, f'{split_name}.json')
        dataset = TargetTextCollection.load_json(data_fp)
        if split_name == 'train':
            print(dataset.number_targets())
            print(len(dataset))
            for target_text in dataset.values():
                target_sentiment_distribution.update(target_text['target_sentiments'])
            for key, value in target_sentiment_distribution.items():
                target_sentiment_distribution[key] = round((value / dataset.number_targets()), 2) * 100

        if split_name == 'train' and average_sentiment:
            dataset.one_sentiment_text('target_sentiments', average_sentiment)
        elif split_name == 'train' and not average_sentiment:
            dataset.one_sentiment_text('target_sentiments', average_sentiment)
            ids_to_remove = []
            for text_id, target_text in dataset.items():
                if 'text_sentiment' not in target_text:
                    ids_to_remove.append(text_id)
            for id_to_remove in ids_to_remove:
                del dataset[id_to_remove]
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_fp = Path(save_dir, f'{split_name}.json')
            dataset.to_json_file(save_fp)
            print(f'Saved dataset to {save_fp}')

        if split_name == 'train':
            if args.save_dir and save_dir == False:
                existing_dataset_fp = Path(args.save_dir, 'train.json')
                dataset = TargetTextCollection.load_json(existing_dataset_fp)
            train_size = len(dataset)
            for target_text in dataset.values():
                text_sentiment_distribution.update([target_text['text_sentiment']])
            for key, value in text_sentiment_distribution.items():
                text_sentiment_distribution[key] = round((value / train_size), 2) * 100
        
        split_name_dataset.append((split_name, dataset))
    print(f'Training dataset size for text classification {train_size}')
    print(f'Target sentiment distribution for training data {target_sentiment_distribution}')
    print(f'Text sentiment distribution for training data {text_sentiment_distribution}')
