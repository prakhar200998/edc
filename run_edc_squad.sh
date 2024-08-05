#!/bin/bash

# Run Schema Canonicalization
python run.py \
    --oie_llm gpt-4o-mini \
    --oie_few_shot_example_file_path ./few_shot_examples/webnlg/oie_few_shot_examples.txt \
    --sd_llm gpt-4o-mini \
    --sd_few_shot_example_file_path ./few_shot_examples/webnlg/sd_few_shot_examples.txt \
    --sc_llm gpt-4o-mini \
    --input_text_file_path ./datasets/selected_squad_contexts.txt \
    --enrich_schema \
    --output_dir ./output/squad_schema_canonicalization \
    --ee_llm gpt-4o-mini \
    --ee_few_shot_example_file_path ./few_shot_examples/webnlg/ee_few_shot_examples.txt
