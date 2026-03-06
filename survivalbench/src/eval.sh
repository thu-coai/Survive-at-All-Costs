#!/bin/bash

model_name_or_path=<model_name_or_path>
model_name=${model_name_or_path##*/}
mode=api # api or local
api_key=<api_key> # required for api mode
question_file=../data/survivalbench.jsonl
output_dir=../results/${model_name}

echo "Start evaluation for ${model_name}..."
mkdir -p $output_dir

CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model-name-or-path $model_name_or_path \
    --mode $mode \
    --api-key $api_key \
    --question-file $question_file \
    --output-file $output_dir/raw_responses.jsonl

echo "Start extracting choices for ${model_name}..."

python extract_choice.py \
    --api-key $api_key \
    --input-file $output_dir/raw_responses.jsonl \
    --output-file $output_dir/extracted_choices.jsonl

echo "Start counting results for ${model_name}..."

python count_results.py \
    --input-file $output_dir/extracted_choices.jsonl \
    --output-file $output_dir/count_results.json

echo "All processes completed. Results saved in $output_dir/count_results.json"


## cot evaluation

# python cot_evaluation.py \
#     --model-name $model_name \
#     --api-key $api_key \
#     --input-file $output_dir/raw_responses.jsonl \
#     --output-file $output_dir/cot_evaluation.jsonl