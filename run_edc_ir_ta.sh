#!/bin/bash

# Target Alignment with Iterative Refinement
python run.py \
    --oie_llm gpt-4o-mini \
    --oie_few_shot_example_file_path ./few_shot_examples/webnlg/oie_few_shot_examples.txt \
    --sd_llm gpt-4o-mini \
    --sd_few_shot_example_file_path ./few_shot_examples/webnlg/sd_few_shot_examples.txt \
    --sc_llm gpt-4o-mini \
    --input_text_file_path ./datasets/webnlg.txt \
    --target_schema_path ./schemas/webnlg_schema.csv \
    --output_dir ./output/webnlg_IR_target_alignment \
    --sr_adapter_path ./trained_model \
    --oie_refine_few_shot_example_file_path ./few_shot_examples/webnlg/oie_few_shot_refine_examples.txt \
    --ee_llm gpt-4o-mini \
    --ee_few_shot_example_file_path ./few_shot_examples/webnlg/ee_few_shot_examples.txt \
    --refinement_iterations 1
