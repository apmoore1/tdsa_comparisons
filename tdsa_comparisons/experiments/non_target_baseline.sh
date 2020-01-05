#!/bin/bash
number_runs=$1
model_dir=$2

python_path="$(which python)"

declare -a average_arr=("single" "average")
declare -a domain_arr=("Laptop" "Election" "Restaurant")
config_fp="./resources/model_configs/CNN.jsonnet"

for average_value in "${average_arr[@]}"
do
    echo "${average_value}"
    for domain in "${domain_arr[@]}"
    do
        echo "${domain}"
        lower_domain="${domain,,}"
        data_dir="./data/text_classification/"$average_value"/"$lower_domain"_dataset"
        model_save_name="CNN_"$domain"_"$average_value
        if [ "$average_value" == "single" ]; then
            echo "Running single"
            $python_path ./tdsa_comparisons/experiments/non_target_models.py "${data_dir}" "${config_fp}" $number_runs $domain "CNN" $model_dir $model_save_name --glove
        else
            echo "Running average option"
            $python_path ./tdsa_comparisons/experiments/non_target_models.py "${data_dir}" "${config_fp}" $number_runs $domain "CNN" $model_dir $model_save_name --glove --average
        fi
        
    done
done