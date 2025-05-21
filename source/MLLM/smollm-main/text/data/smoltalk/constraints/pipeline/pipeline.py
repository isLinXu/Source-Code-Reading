from distilabel.llms import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import MagpieGenerator
from ifeval_tasks import (
    IFEvalInstructionIdListAssignator,
    IFEvalKwargsAssignator,
)
from json_schemas import (
    IFEVAL_INSTRUCTION_ID_LIST_JSON_SCHEMA,
    IFEVAL_RESPONSE_VERIFICATION_FUNCTION_ARGUMENTS_JSON_SCHEMA,
)
from system_prompts import IFEVAL_SYSTEM_PROMPT

IFEVAL_INSTRUCTION_CONFLICTS = {
    "keywords:existence": {"keywords:existence"},
    "keywords:frequency": {"keywords:frequency"},
    "keywords:forbidden_words": {"keywords:forbidden_words"},
    "keywords:letter_frequency": {"keywords:letter_frequency"},
    "language:response_language": {
        "change_case:english_capital",
        "change_case:english_lowercase",
        "startend:end_checker",
        "keywords:frequency",
        "keywords:forbidden_words",
        "detectable_format:multiple_sections",
        "keywords:existence",
        "language:response_language",
    },
    "length_constraints:number_sentences": {"length_constraints:number_sentences"},
    "length_constraints:number_paragraphs": {
        "length_constraints:number_sentences",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_paragraphs",
    },
    "length_constraints:number_words": {"length_constraints:number_words"},
    "length_constraints:nth_paragraph_first_word": {
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_paragraphs",
    },
    "detectable_content:number_placeholders": {
        "detectable_content:number_placeholders"
    },
    "detectable_content:postscript": {"detectable_content:postscript"},
    "detectable_format:number_bullet_lists": {"detectable_format:number_bullet_lists"},
    "detectable_format:constrained_response": {
        "startend:quotation",
        "length_constraints:number_words",
        "detectable_format:constrained_response",
        "change_case:english_capital",
        "startend:end_checker",
        "keywords:forbidden_words",
        "length_constraints:number_sentences",
        "combination:repeat_prompt",
        "combination:two_responses",
        "punctuation:no_comma",
        "detectable_format:number_highlighted_sections",
        "change_case:english_lowercase",
        "detectable_format:number_bullet_lists",
        "detectable_content:number_placeholders",
        "keywords:letter_frequency",
        "keywords:frequency",
        "length_constraints:number_paragraphs",
        "keywords:existence",
        "length_constraints:nth_paragraph_first_word",
        "detectable_format:title",
        "change_case:capital_word_frequency",
        "detectable_format:json_format",
        "detectable_format:multiple_sections",
        "detectable_content:postscript",
        "language:response_language",
    },
    "detectable_format:number_highlighted_sections": {
        "detectable_format:number_highlighted_sections"
    },
    "detectable_format:multiple_sections": {
        "detectable_format:multiple_sections",
        "detectable_format:number_highlighted_sections",
        "language:response_language",
    },
    "detectable_format:json_format": {
        "startend:quotation",
        "length_constraints:number_words",
        "detectable_format:constrained_response",
        "change_case:english_capital",
        "detectable_format:number_bullet_lists",
        "detectable_content:number_placeholders",
        "startend:end_checker",
        "keywords:letter_frequency",
        "keywords:frequency",
        "length_constraints:number_paragraphs",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_sentences",
        "language:response_language",
        "combination:repeat_prompt",
        "detectable_format:title",
        "change_case:capital_word_frequency",
        "combination:two_responses",
        "detectable_format:json_format",
        "punctuation:no_comma",
        "detectable_format:number_highlighted_sections",
        "detectable_format:multiple_sections",
        "detectable_content:postscript",
        "change_case:english_lowercase",
    },
    "detectable_format:title": {"detectable_format:title"},
    "combination:two_responses": {
        "startend:quotation",
        "length_constraints:number_words",
        "detectable_format:constrained_response",
        "change_case:english_capital",
        "detectable_format:number_bullet_lists",
        "detectable_content:number_placeholders",
        "startend:end_checker",
        "keywords:letter_frequency",
        "keywords:frequency",
        "length_constraints:number_paragraphs",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_sentences",
        "combination:repeat_prompt",
        "change_case:capital_word_frequency",
        "combination:two_responses",
        "detectable_format:json_format",
        "detectable_format:number_highlighted_sections",
        "detectable_format:multiple_sections",
        "detectable_content:postscript",
        "change_case:english_lowercase",
    },
    "combination:repeat_prompt": {
        "startend:quotation",
        "length_constraints:number_words",
        "detectable_format:constrained_response",
        "change_case:english_capital",
        "detectable_format:number_bullet_lists",
        "detectable_content:number_placeholders",
        "startend:end_checker",
        "keywords:letter_frequency",
        "keywords:forbidden_words",
        "keywords:frequency",
        "length_constraints:number_paragraphs",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_sentences",
        "language:response_language",
        "combination:repeat_prompt",
        "change_case:capital_word_frequency",
        "combination:two_responses",
        "detectable_format:json_format",
        "detectable_format:number_highlighted_sections",
        "detectable_format:multiple_sections",
        "detectable_content:postscript",
        "change_case:english_lowercase",
    },
    "startend:end_checker": {"startend:end_checker"},
    "change_case:capital_word_frequency": {
        "change_case:english_capital",
        "change_case:capital_word_frequency",
        "change_case:english_lowercase",
    },
    "change_case:english_capital": {"change_case:english_capital"},
    "change_case:english_lowercase": {
        "change_case:english_capital",
        "change_case:english_lowercase",
    },
    "punctuation:no_comma": {"punctuation:no_comma"},
    "startend:quotation": {"startend:quotation", "detectable_format:title"},
}


with Pipeline(name="ifeval-like-dataset").ray() as pipeline:
    instruction_generator = MagpieGenerator(
        llm=vLLM(
            model="Qwen/Qwen2.5-72B-Instruct",
            tokenizer="Qwen/Qwen2.5-72B-Instruct",
            magpie_pre_query_template="qwen2",
            extra_kwargs={
                "tensor_parallel_size": 8,
                "max_model_len": 8192,
                "enable_prefix_caching": True,
            },
            generation_kwargs={
                "temperature": 0.8,
                "top_p": 1.0,
                "max_new_tokens": 1024,
                "stop": [
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|endoftext|>",
                    "<tool_call>",
                ],
                "stop_token_ids": [151643, 151644, 151645, 151657],
            },
        ),
        system_prompt=IFEVAL_SYSTEM_PROMPT,
        batch_size=1000,
        num_rows=500000,
    )

    instruction_id_list_assignator = IFEvalInstructionIdListAssignator(
        llm=vLLM(
            model="Qwen/Qwen2.5-72B-Instruct",
            tokenizer="Qwen/Qwen2.5-72B-Instruct",
            magpie_pre_query_template="qwen2",
            extra_kwargs={
                "tensor_parallel_size": 8,
                "max_model_len": 2048,
                "enable_prefix_caching": True,
            },
            generation_kwargs={
                "temperature": 0.2,
                "max_new_tokens": 256,
            },
            structured_output={
                "format": "json",
                "schema": IFEVAL_INSTRUCTION_ID_LIST_JSON_SCHEMA,
            },
        ),
        input_batch_size=2000,
    )

    instruction_kwargs_assignator = IFEvalKwargsAssignator(
        llm=vLLM(
            model="Qwen/Qwen2.5-72B-Instruct",
            tokenizer="Qwen/Qwen2.5-72B-Instruct",
            magpie_pre_query_template="qwen2",
            extra_kwargs={
                "tensor_parallel_size": 8,
                "max_model_len": 2048,
                "enable_prefix_caching": True,
            },
            generation_kwargs={
                "temperature": 0.2,
                "max_new_tokens": 512,
            },
            structured_output={
                "format": "json",
                "schema": IFEVAL_RESPONSE_VERIFICATION_FUNCTION_ARGUMENTS_JSON_SCHEMA,
            },
        ),
        input_batch_size=2000,
    )

    (
        instruction_generator
        >> instruction_id_list_assignator
        >> instruction_kwargs_assignator
    )

if __name__ == "__main__":
    distiset = pipeline.run(use_cache=True)
    distiset.push_to_hub(
        "argilla-warehouse/tons-of-ifeval-like-data", include_script=True, private=True
    )
