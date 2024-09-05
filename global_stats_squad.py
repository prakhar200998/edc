# Accuracy for each system
accuracy_squad = {
    'baseline_LLM': 0.48,
    'EDC_GoG_pipeline': 0.62,
    'integrated_EDC_GoG_refinement': 0.68,
}

# Query Answering (QA) Accuracy for Incomplete graph (100 examples)
qa_accuracy_squad = {
    'EDC_pipeline': 0.12,
    'GoG_pipeline': 0.53,
    'integrated_EDC_GoG_refinement': 0.49
}

# Execution Time (in minutes) QA - 500 examples
execution_time_squad = {
    'EDC_Graph_Creation': 185,
    'QA_GoG': 153,
    'QA_integrated': 246,
    'Accuracy_Calculation': 24,
}

# Scalability (Processing Time in minutes for varying data sizes) (10% QA))
scalability_squad = {
    'data_size_5': {
        'EDC_Graph_Creation': 49,
        'QA_GoG': 42,
        'QA_integrated': 68,
        'Accuracy_Calculation': 7,
    },
    'data_size_20': {
        'EDC_Graph_Creation': 185,
        'QA_GoG': 153,
        'QA_integrated': 246,
        'Accuracy_Calculation': 24,
    },
    'data_size_50': {
        'EDC_Graph_Creation': 453,
        'QA_GoG': 378,
        'QA_integrated': 612,
        'Accuracy_Calculation': 57,
    }
}

#Dataset Size (20 titles from SQuAD dataset)  (Schema Canonicalization)

dataset_size_squad = {
    'nodes': 9616,
    'relationships': 8610
} 


