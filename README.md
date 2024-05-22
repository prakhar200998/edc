
# EDC Repository

This repository contains scripts and data for entity extraction, schema definition, and canonicalization. Below, you will find instructions on how to set up and use the code provided.

## Folder Structure

- `datasets/`: Contains .txt files for different datasets.
- `few_shot_examples/`: Contains subfolders for each dataset with few-shot example .txt files.
- `schemas/`: Contains schema files for target alignment for each dataset.
- `output/`: Contains subfolders for processed triplets and scripts for further processing.
- `scripts/`: Shell scripts to run the processes.
- `prompt_templates/`: Contains templates for different types of prompts.
- Other files: Various scripts and configuration files.

## Setup

1. **Clone the Repository:**
   \`\`\`bash
   git clone <repository_url>
   cd edc
   \`\`\`

2. **Install Dependencies:**
   Ensure you have \`conda\` installed. Then, create and activate the environment:
   \`\`\`bash
   conda env create -f environment.yml
   conda activate edc
   \`\`\`

3. **Set Up OpenAI API Key:**
   Add your OpenAI API key in your terminal session:
   \`\`\`bash
   export OPENAI_API_KEY='your_openai_api_key'
   \`\`\`
   Also, add this key to the \`triple_processing.py\` file.

## Usage

1. **Running the Target Alignment and Schema Canonicalization:**
   Use the provided shell script to run the code:
   \`\`\`bash
   bash run_edc.sh
   \`\`\`

2. **Processing Triplets:**
   After running the script, process the generated triplets using:
   \`\`\`bash
   python output/chemotherapy_schema_canonicalization/triple_processing.py
   \`\`\`

3. **Creating the Graph:**
   To create the graph in Neo4j, run:
   \`\`\`bash
   python output/chemotherapy_schema_canonicalization/neo4j_creation.py
   \`\`\`

4. **Custom Datasets:**
   - Create your dataset \`.txt\` file and place it in the \`datasets/\` folder.
   - Create corresponding few-shot example files and place them in a new subfolder within \`few_shot_examples/\`.
   - Create a schema file for your dataset and place it in the \`schemas/\` folder.
   - Update paths in the scripts as necessary.

## Notes

- Ensure that the paths in the scripts are correctly set to your data and schema files.
- You can copy the scripts in the \`output/\` folder to your own directory and modify the paths to fit your setup.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
