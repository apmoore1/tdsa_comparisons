#!/bin/bash
python_path="$(which python)"
# Election dataset
election_dir="./data/election_dataset"
$python_path ./tdsa_comparisons/splitting_data/create_splits.py "sentiment" "none" "none" "election_twitter" "${election_dir}/train.json" "${election_dir}/val.json" "${election_dir}/test.json" --in_target_order

# Laptop and Restaurant datasets
semeval_train_dir="./data/SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines"
semeval_test_dir="./data/ABSA_Gold_TestData"

laptop_train="${semeval_train_dir}/Laptop_Train_v2.xml" 
laptop_test="${semeval_test_dir}/Laptops_Test_Gold.xml"
laptop_dir="./data/laptop_dataset"
$python_path ./tdsa_comparisons/splitting_data/create_splits.py "sentiment" "${laptop_train}" "${laptop_test}" "semeval_2014" "${laptop_dir}/train.json" "${laptop_dir}/val.json" "${laptop_dir}/test.json" --in_target_order

restaurant_train="${semeval_train_dir}/Restaurants_Train_v2.xml" 
restaurant_test="${semeval_test_dir}/Restaurants_Test_Gold.xml"
restaurant_dir="./data/restaurant_dataset"
$python_path ./tdsa_comparisons/splitting_data/create_splits.py "sentiment" "${restaurant_train}" "${restaurant_test}" "semeval_2014" "${restaurant_dir}/train.json" "${restaurant_dir}/val.json" "${restaurant_dir}/test.json" --in_target_order