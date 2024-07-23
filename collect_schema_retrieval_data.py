import pandas as pd
import os
from tqdm import tqdm
import csv
import json
import random
import edc.utils.llm_utils as llm_utils
import ast
from collections import Counter
from argparse import ArgumentParser
from datasets import Dataset, DatasetDict


def read_tekgen(tekgen_path):
    json_dict_list = []
    with open(tekgen_path, "r") as f:
        lines = f.readlines()
        for l in tqdm(lines):
            line_json_dict = json.loads(l)
            triples = line_json_dict["triples"]
            text = line_json_dict["sentence"]

            skip_flag = False
            for triple in triples:
                # skip quadruples
                if len(triple) != 3:
                    skip_flag = True
                else:
                    subject = triple[0]
                    relation = triple[1]
                    object = triple[2]

                    # Check if subject and object are present in text
                    if subject.lower() not in text.lower() or object.lower() not in text.lower():
                        # print(f"{triple} not explicitly in {text}")
                        skip_flag = True
            if not skip_flag:
                json_dict_list.append(line_json_dict)
    return json_dict_list

#AD

def crawl_relation_definitions(json_dict_list, result_csv_path):
    schema_definition_prompt_template = open("./prompt_templates/sd_template.txt").read()
    schema_definition_few_shot_examples = open("./few_shot_examples/default/sd_few_shot_examples.txt").read()

    collected_relations = set()

    if not os.path.exists(result_csv_path):
        result_csv = open(result_csv_path, "w")
        csv_writer = csv.writer(result_csv)
        csv_writer.writerow(["text", "triplets", "relations", "definitions"])
    else:
        result_csv = open(result_csv_path, "a")
        csv_writer = csv.writer(result_csv)

    progress_bar = tqdm(total=5000)
    for json_dict in json_dict_list:
        if len(collected_relations) >= 10:
            break
        triples = json_dict["triples"]
        skip_flag = False
        for triple in triples:
            # skip quadruples
            if len(triple) != 3:
                skip_flag = True
            relation = triple[1]
            if relation in collected_relations:
                # This is already collected, skip
                skip_flag = True
        if skip_flag:
            continue
        else:
            for triple in triples:
                relation = triple[1]
                if relation not in collected_relations:
                    collected_relations.add(relation)
                    progress_bar.update()
            text = json_dict["sentence"]
            triples = json_dict["triples"]
            present_relations = list(set([t[1] for t in triples]))

            filled_first_prompt = schema_definition_prompt_template.format_map(
                {
                    "few_shot_examples": schema_definition_few_shot_examples,
                    "text": text,
                    "triples": triples,
                    "relations": present_relations,
                }
            )

            # print(filled_first_prompt)
            output = llm_utils.openai_chat_completion(
                "gpt-4o-mini",
                system_prompt=None,
                history=[{"role": "user", "content": filled_first_prompt}],
            )
            csv_writer.writerow([text, triples, present_relations, output])
            result_csv.flush()


def collect_samples(df, dataset_size):
    # entries: list of dicts containing text and triples
    # relation_definitions: dict from relation to definitions
    collected_samples = []

    relation_definition_dict_list = []
    aggregated_relation_definition_dict = {}

    for idx, row in df.iterrows():
        raw_definitions = row["definitions"]
        relation_definition_dict = llm_utils.parse_relation_definition(raw_definitions)
        relation_definition_dict_list.append(relation_definition_dict)
        for relation, definition in relation_definition_dict.items():
            if relation not in aggregated_relation_definition_dict:
                aggregated_relation_definition_dict[relation] = [definition]
            else:
                aggregated_relation_definition_dict[relation].append(definition)

    for row_idx, row in df.iterrows():
        text = row["text"]
        triples = ast.literal_eval(row["triplets"])

        positive_relations = set()

        relation_triple_dict = {}

        for triple in triples:
            subject = triple[0]
            relation = triple[1]
            object = triple[2]

            # Check if subject and object are present in text
            if subject.lower() not in text.lower() or object.lower() not in text.lower():
                print(f"{triple} not explicitly in {text}")
                continue

            if relation in relation_definition_dict_list[row_idx]:
                positive_relations.add(relation)
                if relation not in relation_triple_dict:
                    relation_triple_dict[relation] = [triple]
                else:
                    relation_triple_dict[relation].append(triple)
        # print(len(aggregated_relation_definition_dict))
        negative_relations = set(aggregated_relation_definition_dict.keys()) - positive_relations
        # print(positive_relations)
        # print(negative_relations)
        negative_relations = random.sample(list(negative_relations), len(positive_relations))

        positive_relations = list(positive_relations)
        negative_relations = list(negative_relations)

        assert len(positive_relations) == len(negative_relations)

        for idx in range(len(negative_relations)):
            if idx >= 2:
                # Max 3 samples per sentence to ensure diversity of datasets
                break
            sample = {
                "sentence": text,
                "positive": f"{positive_relations[idx]}: {relation_definition_dict_list[row_idx][positive_relations[idx]]}",
                "negative": f"{negative_relations[idx]}: {random.choice(aggregated_relation_definition_dict[negative_relations[idx]])}",
                "positive_relation": positive_relations[idx],
                "negative_relation": negative_relations[idx],
                "positive_triple": relation_triple_dict[positive_relations[idx]],
            }
            # print(sample)
            collected_samples.append(sample)
            print(sample)
            if len(collected_samples) >= dataset_size:
                return collected_samples
    return collected_samples

def check_csv_content(csv_path):
    """Check if the CSV file exists and is not empty."""
    try:
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            with open(csv_path, 'r') as file:
                # Check if there's more than just the header
                has_content = next(csv.reader(file), None) and next(csv.reader(file), None)
                return has_content
        return False
    except IOError:
        return False

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tekgen_path", help="Path to tekgen path")
    parser.add_argument("--relation_definition_csv_path", help="Output path of relation definition of tekgen")
    parser.add_argument("--dataset_size", default=50000, type=int)
    parser.add_argument("--output_path", default="./schema_retriever_dataset")

    args = parser.parse_args()

    tekgen_path = args.tekgen_path
    relation_definition_csv_path = args.relation_definition_csv_path
    dataset_size = args.dataset_size
    output_path = args.output_path

    entries = read_tekgen(tekgen_path)
    print("Number of entries read:", len(entries))


    if not check_csv_content(relation_definition_csv_path):
        crawl_relation_definitions(entries, relation_definition_csv_path)
        print("Crawl completed. Check the CSV for entries.")
    else:
        print("CSV file already exists and contains data.")


    collected_samples = collect_samples(pd.read_csv(relation_definition_csv_path), dataset_size)
    print("Number of samples collected:", len(collected_samples))
    
    data = Dataset.from_list(collected_samples)

    train_test_split = data.train_test_split()
    test_valid = train_test_split["test"].train_test_split(test_size=0.5)
    train_test_valid_dataset = DatasetDict(
        {"train": train_test_split["train"], "test": test_valid["test"], "valid": test_valid["train"]}
    )

    train_test_valid_dataset.save_to_disk(output_path)
