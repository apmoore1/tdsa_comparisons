#!/bin/bash
number_runs=$1
model_dir=$2

python_path="$(which python)"

declare -a domain_arr=("Laptop" "Election" "Restaurant")
config_fp="./resources/model_configs/CNN.jsonnet"


for domain in "${domain_arr[@]}"
do
    echo "${domain}"
    lower_domain="${domain,,}"
    data_dir="./data/text_classification/average/"$lower_domain"_dataset"
    model_save_name="CNN_"$domain"_average_cwr"
    $python_path ./tdsa_comparisons/experiments/non_target_models.py "${data_dir}" "${config_fp}" $number_runs $domain "CNN" $model_dir $model_save_name --cwr --average
        
done