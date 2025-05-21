IFEVAL_RESPONSE_VERIFICATION_FUNCTION_ARGUMENTS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {
            "type": ["string", "null"],
        },
        "num_sentences": {
            "type": ["integer", "null"],
            "minimum": 1,
        },
        "relation": {
            "type": ["string", "null"],
            "enum": ["less than", "at least"],
        },
        "num_placeholders": {
            "type": ["integer", "null"],
            "minimum": 1,
        },
        "num_bullets": {
            "type": ["integer", "null"],
            "minimum": 1,
        },
        "starter": {
            "type": ["string", "null"],
        },
        "num_highlights": {
            "type": ["integer", "null"],
            "minimum": 1,
        },
        "section_spliter": {
            "type": ["string", "null"],
            "enum": ["Section", "SECTION"],
        },
        "num_sections": {
            "type": ["integer", "null"],
            "minimum": 1,
        },
        "num_paragraphs": {
            "type": ["integer", "null"],
            "minimum": 1,
        },
        "postscript_marker": {
            "type": ["string", "null"],
            "enum": ["P.S.", "P.P.S"],
        },
        "original_message": {
            "type": ["string", "null"],
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
        },
        "keyword": {
            "type": ["string", "null"],
        },
        "frequency": {
            "type": ["integer", "null"],
            "minimum": 1,
        },
        "num_words": {
            "type": ["integer", "null"],
            "minimum": 1,
        },
        "nth_paragraph": {
            "type": ["integer", "null"],
            "minimum": 1,
        },
        "first_word": {
            "type": ["string", "null"],
        },
        "key_sentences": {
            "type": ["array", "null"],
            "items": {"type": "string"},
        },
        "forbidden_words": {
            "type": ["array", "null"],
            "items": {"type": "string"},
        },
        "original_paragraph": {
            "type": ["string", "null"],
        },
        "low": {
            "type": ["integer", "null"],
            "minimum": 0,
        },
        "high": {
            "type": ["integer", "null"],
            "minimum": 0,
        },
        "prompt_to_repeat": {
            "type": ["string", "null"],
        },
        "end_phrase": {
            "type": ["string", "null"],
        },
        "letter": {
            "type": ["string", "null"],
            "minLength": 1,
            "maxLength": 1,
            "pattern": "[a-zA-Z]",
        },
        "let_frequency": {
            "type": ["integer", "null"],
            "minimum": 1,
        },
        "let_relation": {
            "type": ["string", "null"],
            "enum": ["less than", "at least"],
        },
        "capital_frequency": {
            "type": ["integer", "null"],
            "minimum": 1,
        },
        "capital_relation": {
            "type": ["string", "null"],
            "enum": ["less than", "at least"],
        },
    },
    "required": [
        "language",
        "num_sentences",
        "relation",
        "num_placeholders",
        "num_bullets",
        "starter",
        "num_highlights",
        "section_spliter",
        "num_sections",
        "num_paragraphs",
        "postscript_marker",
        "original_message",
        "keywords",
        "keyword",
        "frequency",
        "num_words",
        "nth_paragraph",
        "first_word",
        "key_sentences",
        "forbidden_words",
        "original_paragraph",
        "low",
        "high",
        "prompt_to_repeat",
        "end_phrase",
        "letter",
        "let_frequency",
        "let_relation",
        "capital_frequency",
        "capital_relation",
    ],
    "additionalProperties": False,
}

IFEVAL_INSTRUCTION_ID_LIST_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "instruction_id_list": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "keywords:existence",
                    "keywords:frequency",
                    "keywords:forbidden_words",
                    "keywords:letter_frequency",
                    "language:response_language",
                    "length_constraints:number_sentences",
                    "length_constraints:number_paragraphs",
                    "length_constraints:number_words",
                    "length_constraints:nth_paragraph_first_word",
                    "detectable_content:number_placeholders",
                    "detectable_content:postscript",
                    "detectable_format:number_bullet_lists",
                    "detectable_format:constrained_response",
                    "detectable_format:number_highlighted_sections",
                    "detectable_format:multiple_sections",
                    "detectable_format:json_format",
                    "detectable_format:title",
                    "combination:two_responses",
                    "combination:repeat_prompt",
                    "startend:end_checker",
                    "change_case:capital_word_frequency",
                    "change_case:english_capital",
                    "change_case:english_lowercase",
                    "punctuation:no_comma",
                    "startend:quotation",
                ],
            },
            "uniqueItems": True,
        }
    },
    "required": ["instruction_id_list"],
}
