#!/bin/bash
python_path="$(which python)"
# Election dataset
election_dir="./data/election_dataset"
$python_path ./tdsa_comparisons/splitting_data/re_order.py "${election_dir}"

# Laptop and Restaurant datasets
laptop_dir="./data/laptop_dataset"
$python_path ./tdsa_comparisons/splitting_data/re_order.py "${laptop_dir}"

restaurant_dir="./data/restaurant_dataset"
$python_path ./tdsa_comparisons/splitting_data/re_order.py "${restaurant_dir}"