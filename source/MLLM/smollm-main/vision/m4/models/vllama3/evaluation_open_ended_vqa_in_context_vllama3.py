from typing import List

from m4.evaluation.custom_metrics.doc_vqa_metrics import DVQAMetrics
from m4.evaluation.custom_metrics.open_ended_vqa_metrics import OEVQAMetrics
from m4.models.vgpt2.evaluation_open_ended_vqa_in_context_vgpt2 import Vgpt2OpenEndedVQAInContext


class VLlama3OpenEndedVQAInContext(Vgpt2OpenEndedVQAInContext):
    model_class: str = "VLlama3ForCausalLM"
    tokenizer_max_seq_len = 10240


class VQAv2VLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    id_column_name: str = "question_id"


class VQAv2Modif10kVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_10k"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation_10k"
    validation_query_split_name: str = "validation_10k"
    validation_support_split_name: str = "train"


class VQAv2NewSplitsVLlama3OpenEndedVQAInContextAcc(VQAv2VLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class VQAv2Part0NewSplitsVLlama3OpenEndedVQAInContextAcc(VQAv2NewSplitsVLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_0"


class VQAv2Part1NewSplitsVLlama3OpenEndedVQAInContextAcc(VQAv2NewSplitsVLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_1"


class VQAv2Part2NewSplitsVLlama3OpenEndedVQAInContextAcc(VQAv2NewSplitsVLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_2"


class VQAv2Part3NewSplitsVLlama3OpenEndedVQAInContextAcc(VQAv2NewSplitsVLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_3"


class VQAv2Part4NewSplitsVLlama3OpenEndedVQAInContextAcc(VQAv2NewSplitsVLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_4"


class VQAv2Part5NewSplitsVLlama3OpenEndedVQAInContextAcc(VQAv2NewSplitsVLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_5"


class VQAv2SampleVLlama3OpenEndedVQAInContextAcc(VQAv2VLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Sample"


class VQAv2DummyVLlama3OpenEndedVQAInContextAcc(VQAv2VLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Dummy"


class VQAv2ChatbotVLlama3OpenEndedVQAInContextAcc(VQAv2VLlama3OpenEndedVQAInContextAcc):
    dataset_config: str = "validation"
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "train"
    test_query_split_name: str = "test"
    test_support_split_name: str = "train"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{question}\n"
                "Give a very brief answer.<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class OKVQAVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    # We are considering the raw answers. In the original paper,
    # they are doing a step of stemming (standardize pluralization and conjugation).
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"


class OKVQANewSplitsVLlama3OpenEndedVQAInContextAcc(OKVQAVLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class OKVQASampleVLlama3OpenEndedVQAInContextAcc(OKVQAVLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif-Sample"


class OKVQAChatbotVLlama3OpenEndedVQAInContextAcc(OKVQAVLlama3OpenEndedVQAInContextAcc):
    # We are considering the raw answers. In the original paper,
    # they are doing a step of stemming (standardize pluralization and conjugation).
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{question}\n"
                "Give a very brief answer.<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class TextVQAVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "textvqa"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"  # List of strings


class TextVQANewSplitsVLlama3OpenEndedVQAInContextAcc(TextVQAVLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/textvqa_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class TextVQASampleVLlama3OpenEndedVQAInContextAcc(TextVQAVLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/textvqa-Sample"


class TextVQAChatbotVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "textvqa"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "train"
    test_query_split_name: str = "test"
    test_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"  # List of strings
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{question}\n"
                "Give a very brief answer.<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
        1: {
            "prefix": "{bos_token}",
            "example": """<image>Answer the following question about the image using as few words as possible. Follow these additional instructions:
-Always answer a binary question with Yes or No.
-When asked what time it is, reply with the time seen in the image.
-Do not put any full stops at the end of the answer.
-Do not put quotation marks around the answer.
-An answer with one or two words is favorable.
-Do not apply common sense knowledge. The answer can be found in the image.
Question: {question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n""",
        },
    }
    selected_prompt_template_id = 1
    bool_instruct_templates = True


class AdVQAVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/AdVQA_modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"


class AdVQASampleVLlama3OpenEndedVQAInContextAcc(AdVQAVLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/AdVQA_modif-Sample"


class VizWizVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/VizWiz"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    id_column_name: str = "filename"


class VizWizSampleVLlama3OpenEndedVQAInContextAcc(VizWizVLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VizWiz-Sample"


class VizWizNewSplitsVLlama3OpenEndedVQAInContextAcc(VizWizVLlama3OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VizWiz_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    prompt_templates_dict = {
        0: {
            "prefix": (
                "{bos_token}Instruction: provide an answer to the question. Use the image to answer. Answer"
                " 'unanswerable' if the question is ambiguous or impossible to answer.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        1: {
            "prefix": (
                "{bos_token}Instruction: provide an answer to the question. Use the image to answer. Answer"
                " 'unanswerable' if the question is impossible to answer.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        2: {
            "prefix": (
                "{bos_token}Instruction: provide an answer to the question when possible, otherwise say unanswerable."
                " Use the image to answer.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        3: {
            "prefix": (
                "{bos_token}Task: Answer the questions based on the image when possible, otherwise say unanswerable.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        4: {
            "prefix": (
                "{bos_token}Task: Answer the questions based on the image. When it's ambiguous, answer unanswerable.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        5: {
            "prefix": (
                "{bos_token}Homework:\nExercise:Answer the questions based on the image. When it's ambiguous, answer"
                " unanswerable.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        6: {
            "prefix": (
                "{bos_token}Exercise:Answer the questions based on the image. When it's ambiguous, answer"
                " unanswerable.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        7: {
            "prefix": (
                "{bos_token}Homework:\nExercise:When possible, answer the questions based on the image, otherwise say"
                " unanswerable.\n"
            ),
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
    }
    selected_prompt_template_id = 3


class DocVQAVLlama3OpenEndedVQAInContextAnls(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/DocumentVQA"
    metric_name: str = "DocVQAMetrics"
    metric_kwargs = {
        "metrics": [
            DVQAMetrics.ANLS,
            DVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "train"
    test_query_split_name: str = "test"
    test_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    id_column_name: str = "questionId"
    selected_prompt_template_id = 9


class DocVQASampleVLlama3OpenEndedVQAInContextAnls(DocVQAVLlama3OpenEndedVQAInContextAnls):
    dataset_name: str = "HuggingFaceM4/DocumentVQA-Sample"
    metric_kwargs = {
        "metrics": [
            DVQAMetrics.ANLS,
        ]
    }


class DocVQAChatbotVLlama3OpenEndedVQAInContextAnls(DocVQAVLlama3OpenEndedVQAInContextAnls):
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{question}\nGive a very brief"
                " answer.<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
        1: {
            "prefix": "{bos_token}",
            "example": """User:<image>Give a short and terse answer to the following question. Do not paraphrase or reformat the text you see in the image. Do not include any full stops. Just give the answer without additional explanation.
Question: {question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n""",
        },
    }
    selected_prompt_template_id = 1
    bool_instruct_templates = True


class MMMUVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/MMMU-modif-with-categories"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.OE_MMMU_STYLE_PER_BUCKET_ACCURACY,
            OEVQAMetrics.OE_MMMU_STYLE_VQA_ACCURACY,
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "dev"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "dev"
    test_query_split_name: str = "test"
    test_support_split_name: str = "dev"
    image_column_name: str = "images"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    buckets_keys: List[str] = ["broad_category", "narrow_category"]
    id_column_name: str = "id"
    selected_prompt_template_id = 14


class MathVistaVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/MathVista-modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.OE_MMMU_STYLE_VQA_ACCURACY,
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "validation"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "validation"
    test_query_split_name: str = "test"
    test_support_split_name: str = "validation"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    id_column_name: str = "pid"
    selected_prompt_template_id = 15


class MMVETChatbotVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/MM_VET_modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS_MMVET,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "test"
    test_query_split_name: str = "test"
    test_support_split_name: str = "test"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answer"
    id_column_name: str = "question_id"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": "User:<image>{question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n",
        },
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class LLaVAWildChatbotVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/LLaVA_Wild_Modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS_LLAVA_WILD,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "test"
    test_query_split_name: str = "test"
    test_support_split_name: str = "test"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answer"
    id_column_name: str = "question_id"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": "User:<image>{question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n",
        },
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class ChartQAChatbotVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/ChartQA"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.OE_RELAXED_VQA_ACCURACY,
        ]
    }
    default_query_split_name: str = "val"
    default_support_split_name: str = "val"
    validation_query_split_name: str = "val"
    validation_support_split_name: str = "val"
    test_query_split_name: str = "test"
    test_support_split_name: str = "test"
    image_column_name: str = "image"
    question_column_name: str = "query"
    answers_column_name: str = "label"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{question}\nGive a very brief"
                " answer.<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
        1: {
            "prefix": "{bos_token}",
            "example": """User:<image>For the question below, follow the following instructions:
-The answer should contain as few words as possible.
-Don’t paraphrase or reformat the text you see in the image.
-Answer a binary question with Yes or No.
-When asked to give a numerical value, provide a number like 2 instead of Two.
-If the final answer has two or more items, provide it in the list format like [1, 2].
-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.
-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.
-Don’t include any units in the answer.
-Do not include any full stops at the end of the answer.
-Try to include the full label from the graph when asked about an entity.
Question: {question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n""",
        },
    }
    selected_prompt_template_id = 1
    bool_instruct_templates = True


class GQAChatbotVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/GQA"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "train"
    test_query_split_name: str = "test"
    test_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    id_column_name: str = "question_id"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": "User:<image>{question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n",
        },
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class MathVistaOpenEndedChatbotVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/MathVista_OpenEnded"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.OE_ONLY_MMMU_STYLE_VQA_ACCURACY,
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "testmini"
    default_support_split_name: str = "testmini"
    validation_query_split_name: str = "testmini"
    validation_support_split_name: str = "testmini"
    test_query_split_name: str = "test"
    test_support_split_name: str = "testmini"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answer"
    context_column_names: List[str] = ["instruction"]
    id_column_name: str = "pid"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": "User:<image>{question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n",
        },
        1: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{question}\n{instruction}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
        2: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{instruction}\n{question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
    }
    selected_prompt_template_id = 2
    bool_instruct_templates = True


class MMMUOpenEndedChatbotVLlama3OpenEndedVQAInContextAcc(VLlama3OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/MMMU_OpenEnded"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.OE_ONLY_MMMU_STYLE_VQA_ACCURACY,
            OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "dev"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "dev"
    test_query_split_name: str = "test"
    test_support_split_name: str = "dev"
    image_column_name: str = "images"
    question_column_name: str = "question"
    answers_column_name: str = "answer"
    id_column_name: str = "id"
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": "User: {question}<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n",
        },
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class MMMUOpenEndedSingleImageChatbotVLlama3OpenEndedVQAInContextAcc(
    MMMUOpenEndedChatbotVLlama3OpenEndedVQAInContextAcc
):
    dataset_name: str = "HuggingFaceM4/MMMU_OpenEnded_single_image"
    default_query_split_name: str = "validation"
    default_support_split_name: str = "validation"
    validation_query_split_name: str = "validation"
    validation_support_split_name: str = "validation"
    test_query_split_name: str = "validation"
    test_support_split_name: str = "validation"
