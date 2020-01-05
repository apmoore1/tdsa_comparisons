import argparse
from pathlib import Path

from target_extraction.data_types import TargetTextCollection

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=parse_path)
    args = parser.parse_args()

    for split_name in ['train.json', 'val.json', 'test.json']:
        split_fp = Path(args.data_dir, split_name)
        collection = TargetTextCollection.load_json(split_fp)
        for value in collection.values():
            try:
                value.re_order(keys_not_to_order=['pos_tags', 'tokenized_text', 
                                               'category_sentiments', 'categories'])
            except:
                print(value)
        collection.re_order(keys_not_to_order=['pos_tags', 'tokenized_text', 
                                               'category_sentiments', 'categories'])
        assert collection.in_order()
        collection.to_json_file(split_fp, include_metadata=True)