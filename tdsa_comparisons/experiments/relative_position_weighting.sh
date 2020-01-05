#!/bin/bash
number_runs=$1
model_dir=$2

python_path="$(which python)"

declare -a model_arr=("IAN" "AE")
declare -a domain_arr=("Laptop" "Election" "Restaurant")
config_dir="./resources/model_configs"

for model in "${model_arr[@]}"
do
    echo "${model}"
    for domain in "${domain_arr[@]}"
    do
        echo "${domain}"
        lower_domain="${domain,,}"
        config_fp="$config_dir/$model.jsonnet"
        data_dir="./data/"$lower_domain"_dataset"
        model_save_name=$model"_"$domain"_relative_position_weighting"
        $python_path ./tdsa_comparisons/experiments/flexi_run_models.py "${data_dir}" "${config_fp}" $number_runs $domain $model $model_dir $model_save_name --glove --position Weighting
    done
done