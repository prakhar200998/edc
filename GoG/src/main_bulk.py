import logging
from sentence_transformers import SentenceTransformer, util
from query_processor import handle_query
from utils import extract_response_new
from entropy import fetch_response, identify_high_entropy_tokens, extract_triple_and_generate_question

# Configure logging
logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the sentence transformer model for calculating semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

def compare_answers(generated, ground_truth):
    gen_emb = model.encode(generated, convert_to_tensor=True)
    truth_emb = model.encode(ground_truth, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(gen_emb, truth_emb).item()
    return similarity, similarity > 0.7  # Returning both similarity score and boolean based on threshold

def process_question(question, instruction, example):
    prompt = question + " Answer in a word, a few words or 1-2 sentences maximum."
    response_mid = fetch_response(prompt)
    highest_entropy_token = identify_high_entropy_tokens(response_mid)
    triple, generated_question = extract_triple_and_generate_question(response_mid, highest_entropy_token)
    response_final = extract_response_new(generated_question, instruction, example)
    return response_final

def main():
    try:
        instruction = """
        Solve a question answering task with interleaving Thought, Action, Observation steps. Only output the Thought and the Action on the basis of the question fed to you initially. In subsequent iterations, you will be fed with the previous round of thought, action and observation. Use these to generate the next thought and action. Thought can reason about the current situation, and Action can be three types: 
        (1) Search[entity], which searches the 3 most similar entities on the graph and returns their one-hop subgraphs.
        (2) Generate[thought], which generate some new triples related to your last thought.
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

        with open("selected_squad_questions_new.txt", "r") as questions_file, \
             open("selected_squad_answers_new.txt", "r") as answers_file, \
             open("results.txt", "w") as results_file:

            questions = questions_file.readlines()
            answers = answers_file.readlines()
            correct_count = 0
            total = 0

            for question, ground_truth in zip(questions, answers):
                response = process_question(question.strip(), instruction, example).strip()
                similarity, is_correct = compare_answers(response, ground_truth.strip())
                results_file.write(f"{response}\n")
                logging.info(f"Question: {question.strip()} | Generated: {response} | Ground Truth: {ground_truth.strip()} | Similarity: {similarity:.2f}")
                
                if is_correct:
                    correct_count += 1
                total += 1

            # logging this to ensure that the accuracy is recorded for future reference
            accuracy = correct_count / total if total else 0
            logging.info(f"Summary Metrics: Total Questions = {total}, Correct Answers = {correct_count}, Accuracy = {accuracy:.2f}")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Total Questions Processed: {total}")
            print(f"Correct Answers: {correct_count}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
