#!/bin/bash
python_path="$(which python)"

# Election dataset
echo "Election Dataset"
echo "Single sentiment dataset split"
election_dir="./data/election_dataset"
election_single_dir="./data/text_classification/single/election_dataset"
$python_path ./tdsa_comparisons/splitting_data/text_classification_dataset.py "${election_dir}" --save_dir "${election_single_dir}"
echo "Average sentiment dataset split"
election_average_dir="./data/text_classification/average/election_dataset"
$python_path ./tdsa_comparisons/splitting_data/text_classification_dataset.py "${election_dir}" --save_dir "${election_average_dir}" --average

# Laptop dataset
echo "Laptop Dataset"
echo "Single sentiment dataset split"
laptop_dir="./data/laptop_dataset"
laptop_single_dir="./data/text_classification/single/laptop_dataset"
$python_path ./tdsa_comparisons/splitting_data/text_classification_dataset.py "${laptop_dir}" --save_dir "${laptop_single_dir}"
echo "Average sentiment dataset split"
laptop_average_dir="./data/text_classification/average/laptop_dataset"
$python_path ./tdsa_comparisons/splitting_data/text_classification_dataset.py "${laptop_dir}" --save_dir "${laptop_average_dir}" --average

# Restaurant dataset
echo "Restaurant Dataset"
echo "Single sentiment dataset split"
restaurant_dir="./data/restaurant_dataset"
restaurant_single_dir="./data/text_classification/single/restaurant_dataset"
$python_path ./tdsa_comparisons/splitting_data/text_classification_dataset.py "${restaurant_dir}" --save_dir "${restaurant_single_dir}"
echo "Average sentiment dataset split"
restaurant_average_dir="./data/text_classification/average/restaurant_dataset"
$python_path ./tdsa_comparisons/splitting_data/text_classification_dataset.py "${restaurant_dir}" --save_dir "${restaurant_average_dir}" --average

echo "Done"