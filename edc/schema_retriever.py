from typing import List
import os
from pathlib import Path
import edc.utils.llm_utils as llm_utils
import re
from edc.utils.e5_mistral_utils import MistralForSequenceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import copy


class SchemaRetriever:
    # The class to handle the last stage: Schema Canonicalization
    def __init__(self, target_schema_dict: dict, embedding_model, embedding_tokenizer) -> None:
        # The canonicalizer uses an embedding model to first fetch candidates from the target schema, then uses a verifier schema to decide which one to canonicalize to or not
        # canonoicalize at all.

        self.target_schema_dict = target_schema_dict
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer

        # Embed the target schema

        self.target_schema_embedding_dict = {}

        for relation, relation_definition in target_schema_dict.items():
            embedding = llm_utils.get_embedding_e5mistral(
                self.embedding_model,
                self.embedding_tokenizer,
                relation_definition,
            )
            self.target_schema_embedding_dict[relation] = embedding

    def update_schema_embedding_dict(self):
        for relation, relation_definition in self.target_schema_dict.items():
            if relation in self.target_schema_embedding_dict:
                continue
            embedding = llm_utils.get_embedding_e5mistral(
                self.embedding_model,
                self.embedding_tokenizer,
                relation_definition,
            )
            self.target_schema_embedding_dict[relation] = embedding

    def retrieve_relevant_relations(self, query_input_text: str, top_k=10):
        target_relation_list = list(self.target_schema_embedding_dict.keys())
        target_relation_embedding_list = list(self.target_schema_embedding_dict.values())

        query_embedding = llm_utils.get_embedding_e5mistral(
            self.embedding_model,
            self.embedding_tokenizer,
            query_input_text,
            "Retrieve descriptions of relations that are present in the given text.",
        )
        scores = np.array([query_embedding]) @ np.array(target_relation_embedding_list).T

        scores = scores[0]
        highest_score_indices = np.argsort(-scores)

        return [target_relation_list[idx] for idx in highest_score_indices[:top_k]]
