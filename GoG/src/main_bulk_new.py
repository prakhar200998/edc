import logging
import os
from sentence_transformers import SentenceTransformer, util
# from query_processor import handle_query  # Assuming this is still needed
from utils import extract_response_new
from entropy import fetch_response, identify_high_entropy_tokens, extract_triple_and_generate_question

# Configure logging
logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the sentence transformer model for calculating semantic similarity
# model = SentenceTransformer('all-MiniLM-L6-v2')

def process_question(question, instruction, example, context):
    # Combine context with the question for each prompt
    prompt_with_context = context + "\n\n" + question

    # Approach 1: Direct answer with context
    direct_response_mid = fetch_response(prompt_with_context)
    direct_response = direct_response_mid.choices[0].message.content

    # Approach 2: High entropy token identification and further processing
    response_mid = fetch_response(prompt_with_context)
    highest_entropy_token = identify_high_entropy_tokens(response_mid)
    triple, generated_question = extract_triple_and_generate_question(response_mid, highest_entropy_token)
    response_final = extract_response_new(generated_question, instruction, example)

    # Approach 3: Direct to extract_response_new
    response_direct = extract_response_new(question, instruction, example)

    # Function to process the response (used for both response_final and response_direct)
    def process_response(response):
        if "Thought" in response and "Action" in response:
            # Extract the thought text
            thought_text = response.split("Thought")[1].split("Action")[0].strip()
            
            # Remove any leading numbers and colons from the thought text
            thought_text = thought_text.split(":", 1)[-1].strip()
            
            # Extract the content inside the brackets from the Action part
            action_content = response.split("Action")[1].strip()
            bracket_content = ""
            if "[" in action_content and "]" in action_content:
                bracket_content = action_content.split("[")[1].split("]")[0].strip()

            # Append the content inside brackets to the processed response
            if bracket_content:
                if not thought_text.endswith('.'):
                    thought_text += '.'
                thought_text += f" {bracket_content}."

            return thought_text.strip()
        else:
            return response.strip()



    # Process responses before returning
    processed_direct_response = process_response(direct_response)
    processed_response_final = process_response(response_final)
    processed_response_direct = process_response(response_direct)

    return processed_direct_response, processed_response_final, processed_response_direct


def main():
    try:
        instruction = """
        Solve a question answering task with interleaving Thought, Action, Observation steps. Only output the Thought and the Action based on the question provided initially. In subsequent iterations, you will be provided with the previous round of thought, action, and observation. Use these to generate the next thought and action. 

        Your search should prioritize identifying specific entities and their relationships rather than broad phrases or concepts. If the search results yield no significant new information after two attempts, you should conclude the task.

        Thoughts should reason about the current situation, and Actions can be three types:
        (1) Search[entity], which searches the 3 most similar entities on the graph and returns their one-hop subgraphs.
        (2) Generate[thought], which generates some new triples related to your last thought.
        (3) Finish[answer1 | answer2 | ...], which returns the answer and finishes the task. Choose this once you feel there is enough information to answer the question and/or there are no new observations in successive steps.
        """

        example = """
        Example 1: What types of treatments are available for breast cancer?
        Topic Entity: [Breast Cancer]
        Thought 1: I need to search for 'Breast Cancer' to find the treatments associated with it.
        Action 1: Search[Breast Cancer]
        Observation 1: Breast Cancer, treated_with, Chemotherapy; Breast Cancer, treated_with, Radiation Therapy
        Thought 2: Both chemotherapy and radiation therapy are noted. I should check what specific drugs are used in chemotherapy for breast cancer.
        Action 2: Search[Chemotherapy for Breast Cancer]
        Observation 2: Chemotherapy for Breast Cancer, uses_drug, Taxol; Chemotherapy for Breast Cancer, uses_drug, Adriamycin
        Thought 3: Taxol and Adriamycin are commonly used. I should provide this information.
        Action 3: Finish[Taxol | Adriamycin]
        
        Example 2: What are the common side effects of chemotherapy for lung cancer?
        Topic Entity: [Chemotherapy for Lung Cancer]
        Thought 1: First, identify the common drugs used in chemotherapy for lung cancer.
        Action 1: Search[Chemotherapy for Lung Cancer]
        Observation 1: Chemotherapy for Lung Cancer, uses_drug, Cisplatin; Chemotherapy for Lung Cancer, uses_drug, Etoposide
        Thought 2: Now, find the side effects of these drugs.
        Action 2: Search[Side Effects of Cisplatin | Side Effects of Etoposide]
        Observation 2: Cisplatin, causes_side_effect, Nausea; Etoposide, causes_side_effect, Hair Loss
        Thought 3: Nausea and hair loss are major concerns. Let's summarize this.
        Action 3: Finish[Nausea | Hair Loss]

        Example 3: How does chemotherapy target cancer cells?
        Topic Entity: [Chemotherapy Mechanism]
        Thought 1: I need to explore how chemotherapy drugs work at the cellular level.
        Action 1: Search[Chemotherapy Mechanism]
        Observation 1: Chemotherapy, targets, Rapidly Dividing Cancer Cells; Chemotherapy, inhibits, Cell Division
        Thought 2: The mechanism focuses on rapidly dividing cells. What specific process is targeted by these drugs?
        Action 2: Generate[Targeting process of chemotherapy drugs]
        Observation 2: Chemotherapy drugs, inhibit, DNA Replication; Chemotherapy drugs, induce, Apoptosis
        Thought 3: DNA replication inhibition and apoptosis induction are key. This should be noted.
        Action 3: Finish[Inhibits DNA Replication | Induces Apoptosis]
        """

        # Simplified context
        simplified_context = """
        The questions pertain to the SQuAD dataset with the following titles: 
        "Capital_punishment_in_the_United_States", "Hard_rock", "Internet_service_provider", 
        "Matter", "United_States_Air_Force", "Pharmaceutical_industry", "Lighting", 
        "Police", "Renewable_energy_commercialization", "Unicode", "Institute_of_technology", 
        "Macintosh", "Light-emitting_diode", "Brain", "Pesticide", "Windows_8", 
        "Videoconferencing", "Genetics", "Food_security", "Education", "Prime_minister".
        Answer the questions in a word, a few words, or a sentence or two at maximum.
        """

        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # File paths
        questions_file = os.path.join(script_dir, "../../datasets/SQuAD/selected_squad_questions_new.txt")
        ground_truth_file = os.path.join(script_dir, "../../datasets/SQuAD/selected_squad_answers_new.txt")
        combined_output_file = os.path.join(script_dir, "combined_results.txt")
        answer_file_1 = os.path.join(script_dir, "results_direct_context.txt")
        answer_file_2 = os.path.join(script_dir, "results_high_entropy.txt")
        answer_file_3 = os.path.join(script_dir, "results_direct_extract.txt")

        # Read questions and ground truth answers
        with open(questions_file, "r") as q_file, \
             open(ground_truth_file, "r") as gt_file, \
             open(combined_output_file, "w") as combined_file, \
             open(answer_file_1, "w") as res_file_1, \
             open(answer_file_2, "w") as res_file_2, \
             open(answer_file_3, "w") as res_file_3:

            questions = q_file.readlines()
            ground_truths = gt_file.readlines()
            total_questions = min(2, len(questions))  # Limit to the first 10 questions

            for i in range(total_questions):
                question = questions[i].strip()
                ground_truth = ground_truths[i].strip()
                logging.info(f"Processing Question {i+1}: {question}")
                
                direct_response, response_final, response_direct = process_question(
                    question, instruction, example, simplified_context)

                # Write the processed responses to their respective files
                res_file_1.write(f"{direct_response}\n")
                res_file_2.write(f"{response_final}\n")
                res_file_3.write(f"{response_direct}\n")

                # Write to the combined output file in the desired format
                combined_file.write(f"Q: {question}\n")
                combined_file.write(f"Ground truth: {ground_truth}\n")
                combined_file.write(f"Direct_Context: {direct_response}\n")
                combined_file.write(f"Direct_Extract: {response_direct}\n")
                combined_file.write(f"High_Entropy: {response_final}\n\n")

                logging.info(f"Question {i+1} processed. Combined output and individual files written.")

        print("Processing completed for the first 10 questions.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()