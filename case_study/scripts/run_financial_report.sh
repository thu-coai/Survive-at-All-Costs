#!/bin/bash

echo "Running Financial Report"
cd ../agent

model_names=(
    "template"
)


for model_name in "${model_names[@]}"; do
    echo "Run ${model_name}"
    echo "Reset environment"
    mkdir -p ../data/financial_report/financial_raw_data
    rm ../data/financial_report/financial_raw_data/*
    cp ../data/financial_report/financial_raw_data_src/* ../data/financial_report/financial_raw_data/
    python main.py \
        --config_file configs/financial_report/${model_name}.yaml \
        --mode autonomous \
        --save_conversation_file ../results/financial_report/${model_name}.json
done
