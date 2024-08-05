from tqdm import tqdm
import json

def read_tekgen(tekgen_path):
    json_dict_list = []
    skipped_due_to_length = 0
    skipped_due_to_content = 0

    with open(tekgen_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line_json_dict = json.loads(line)
            triples = line_json_dict["triples"]
            text = line_json_dict["sentence"]

            skip_flag = False
            for triple in triples:
                if len(triple) != 3:
                    skipped_due_to_length += 1
                    skip_flag = True
                    break  # skip this entry entirely if any triple isn't length 3
                if triple[0].lower() not in text.lower() or triple[2].lower() not in text.lower():
                    skipped_due_to_content += 1
                    skip_flag = True
                    break  # skip this entry if subject/object not in text
            if not skip_flag:
                json_dict_list.append(line_json_dict)

    print(f"Skipped {skipped_due_to_length} entries due to length issues.")
    print(f"Skipped {skipped_due_to_content} entries due to content mismatch.")
    return json_dict_list

# Usage example:
entries = read_tekgen("tekgen.tsv")
print("Entries processed:", len(entries))
