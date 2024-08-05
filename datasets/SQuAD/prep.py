from datasets import load_dataset

# Load the SQuAD dataset
ds = load_dataset("squad")

# List of selected titles
selected_titles = [
     "Capital_punishment_in_the_United_States", "Hard_rock", "Internet_service_provider", 
    "Matter", "United_States_Air_Force", 
    "Pharmaceutical_industry", "Lighting", 
    "Police",  "Renewable_energy_commercialization", "Unicode", "Institute_of_technology", 
    "Macintosh", "Light-emitting_diode", "Brain", "Pesticide", "Windows_8", "Videoconferencing", 
    "Genetics", "Food_security", "Education", "Prime_minister"
]

# Function to extract unique context strings based on selected titles
def extract_selected_data(dataset, titles):
    contexts = set()
    questions = []
    answers = []
    for item in dataset:
        title = item.get('title', '').replace(" ", "_")
        if title in titles:
            context = item['context']
            contexts.add(context)
            question = item['question']
            for answer in item['answers']['text']:
                questions.append(question)
                answers.append(answer)
    return list(contexts), questions, answers

# Extract context strings, questions, and answers from train and validation datasets
train_contexts, train_questions, train_answers = extract_selected_data(ds['train'], selected_titles)
validation_contexts, validation_questions, validation_answers = extract_selected_data(ds['validation'], selected_titles)

# Combine data
all_contexts = list(set(train_contexts + validation_contexts))
all_questions = train_questions + validation_questions
all_answers = train_answers + validation_answers

# Save data to .txt files
def save_to_file(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write(item + '\n')

# Specify the output file paths
output_contexts_file = 'selected_squad_contexts_new.txt'
output_questions_file = 'selected_squad_questions_new.txt'
output_answers_file = 'selected_squad_answers_new.txt'

# Save contexts, questions, and answers to their respective files
save_to_file(all_contexts, output_contexts_file)
save_to_file(all_questions, output_questions_file)
save_to_file(all_answers, output_answers_file)

print(f"Contexts have been saved to {output_contexts_file}")
print(f"Questions have been saved to {output_questions_file}")
print(f"Answers have been saved to {output_answers_file}")
