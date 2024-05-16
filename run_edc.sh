#!/bin/bash

# Run Target Alignment
python run.py \
    --oie_llm gpt-3.5-turbo \
    --oie_few_shot_example_file_path ./few_shot_examples/chemotherapy/oie_few_shot_examples.txt \
    --sd_llm gpt-3.5-turbo \
    --sd_few_shot_example_file_path ./few_shot_examples/chemotherapy/sd_few_shot_examples.txt \
    --sc_llm gpt-3.5-turbo \
    --input_text_file_path ./datasets/chemotherapy.txt \
    --target_schema_path ./schemas/chemotherapy_schema.csv \
    --output_dir ./output/chemotherapy_target_alignment \
    --ee_llm gpt-3.5-turbo \
    --ee_few_shot_example_file_path ./few_shot_examples/chemotherapy/ee_few_shot_examples.txt

# Run Schema Canonicalization
python run.py \
    --oie_llm gpt-3.5-turbo \
    --oie_few_shot_example_file_path ./few_shot_examples/chemotherapy/oie_few_shot_examples.txt \
    --sd_llm gpt-3.5-turbo \
    --sd_few_shot_example_file_path ./few_shot_examples/chemotherapy/sd_few_shot_examples.txt \
    --sc_llm gpt-3.5-turbo \
    --input_text_file_path ./datasets/chemotherapy.txt \
    --enrich_schema \
    --output_dir ./output/chemotherapy_schema_canonicalization \
    --ee_llm gpt-3.5-turbo \
    --ee_few_shot_example_file_path ./few_shot_examples/chemotherapy/ee_few_shot_examples.txt
