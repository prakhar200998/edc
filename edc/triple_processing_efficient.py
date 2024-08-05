import openai
from openai import OpenAI 
import ast

# Set up OpenAI API Key
client = OpenAI(api_key='sk-HX-vp8VFHI2TgNV2JnvxewFiwjL_a6Lf3MeLZN8vCtT3BlbkFJUAoyckTqHcD8n0ECYoRogxarw6QNE0mrPrt9mow3EA')

def format_triple(subject, predicate, object):
    # Apply formatting rules
    formatted_subject = ''.join(word.capitalize() for word in subject.split('_'))
    formatted_predicate = '_'.join(word.upper() for word in predicate.split('_'))
    formatted_object = ''.join(word.capitalize() for word in object.split('_'))
    return formatted_subject, formatted_predicate, formatted_object

def consult_llm(triples):
    # Provide detailed context for LLM to optimize formatting
    prompt = """
    We need to format triples for a Neo4j graph database, ensuring optimal querying capabilities. Each element of the triples—subject, predicate, and object—must be adapted to meet specific standards suitable for graph querying.

    Requirements:
    - Normalize spaces, hyphens, numbers, percentages, and decimals.
    - Subject and object should be in CamelCase without special characters except necessary numbers.
    - Predicates should be uppercase with underscores between words if there are multiple words in the predicate, instead of spaces.
    - Do not alter specific entities such as numbers.

    Here are a few examples:

    Current triples:
    - Subject: "Nuclear_Science_Department"
      Predicate: "location"
      Object: "EPN"
    
    Formatted triple:
    - Subject: "NuclearScienceDepartment"
      Predicate: "LOCATION"
      Object: "EPN"

    Current triples:
    - Subject: "geothermal_power_development"
      Predicate: "processingStatus"
      Object: "under_way"

    Formatted triple:
    - Subject: "GeothermalPowerDevelopment"
      Predicate: "PROCESSING_STATUS"
      Object: "UnderWay"

    Current triples:
    - Subject: "geothermal_power_development"
      Predicate: "processing_status"
      Object: "under_way"

    Formatted triple:
    - Subject: "GeothermalPowerDevelopment"
      Predicate: "PROCESSING_STATUS"
      Object: "UnderWay"
    
    Current triples:
    - Subject: "Nuclear_Science_Department"
      Predicate: "hasInfrastructure"
      Object: "large_infrastructure"
    
    Formatted triple:
    - Subject: "NuclearScienceDepartment"
      Predicate: "HAS_INFRASTRUCTURE"
      Object: "LargeInfrastructure"

    Current triples:
    """

    for triple in triples:
        subject, predicate, object = triple
        prompt += f"- Subject: \"{subject}\"\n  Predicate: \"{predicate}\"\n  Object: \"{object}\"\n"
    
    prompt += """
    Respond strictly with the formatted triples, each on a new line, with no additional text.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo-16k" if you prefer the turbo models
            messages=[
                {"role": "system", "content": "You are assisting in formatting data for a Neo4j graph."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500  # Adjust max tokens as needed
        )
        formatted_response = response.choices[0].message.content.strip().split('\n')
        print(f"Formatted response: {formatted_response}")  # Add this line to debug the formatted response
        
        formatted_triples = []
        for i in range(0, len(formatted_response), 3):
            subject_line = formatted_response[i]
            predicate_line = formatted_response[i + 1]
            object_line = formatted_response[i + 2]
            formatted_subject = subject_line.split(": ")[1].strip('"')
            formatted_predicate = predicate_line.split(": ")[1].strip('"')
            formatted_object = object_line.split(": ")[1].strip('"')
            formatted_triples.append((formatted_subject, formatted_predicate, formatted_object))
        
        return formatted_triples
    except Exception as e:
        print(f"Error during API call: {e}")
        return triples  # Fallback to the original triples

def process_file(input_filename, output_filename):
    formatted_triples = []
    
    # Calculate the total number of lines in the input file
    with open(input_filename, 'r') as file:
        total_lines = sum(1 for _ in file)
    print(f"Total lines to process: {total_lines}")
    
    line_count = 0
    triple_count = 0
    
    with open(input_filename, 'r') as file:
        for line in file:
            line_count += 1
            # Safe evaluation of the string to a Python literal list
            try:
                triple_list = ast.literal_eval(line.strip())
                formatted_triple_list = []
                for triple in triple_list:
                    triple_count += 1
                    s, p, o = triple
                    fs, fp, fo = format_triple(s.strip(), p.strip(), o.strip())
                    formatted_triple_list.append((fs, fp, fo))
                
                formatted_triples_from_llm = consult_llm(formatted_triple_list)
                formatted_triples.append(formatted_triples_from_llm)
                    
                print(f"Processed {triple_count} triples")
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing line {line_count}: {e}")  # Add this line to debug parsing errors
                continue  # Skip lines that cannot be parsed into triples
            
            print(f"Processed {line_count} lines")
    
    # Writing output to a new file
    with open(output_filename, 'w') as outfile:
        for triple_list in formatted_triples:
            for triple in triple_list:
                outfile.write(str(triple) + '\n')

# Replace 'path_to_your_file.txt' with your file's path
# Specify the output file path
file_to_process = '../output/squad_schema_canonicalization/edc_output.txt'
processed_file = '../output/squad_schema_canonicalization/edc_output_processed_eff.txt'
process_file(file_to_process, processed_file)
