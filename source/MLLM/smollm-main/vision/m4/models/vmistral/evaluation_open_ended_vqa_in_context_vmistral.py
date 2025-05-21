from typing import List

from m4.evaluation.custom_metrics.doc_vqa_metrics import DVQAMetrics
from m4.evaluation.custom_metrics.open_ended_vqa_metrics import OEVQAMetrics
from m4.models.vgpt2.evaluation_open_ended_vqa_in_context_vgpt2 import Vgpt2OpenEndedVQAInContext


class VMistralOpenEndedVQAInContext(Vgpt2OpenEndedVQAInContext):
    model_class: str = "VMistralForCausalLM"
    tokenizer_max_seq_len = 4096


class VQAv2VMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
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


class VQAv2Modif10kVMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
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


class VQAv2NewSplitsVMistralOpenEndedVQAInContextAcc(VQAv2VMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class VQAv2Part0NewSplitsVMistralOpenEndedVQAInContextAcc(VQAv2NewSplitsVMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_0"


class VQAv2Part1NewSplitsVMistralOpenEndedVQAInContextAcc(VQAv2NewSplitsVMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_1"


class VQAv2Part2NewSplitsVMistralOpenEndedVQAInContextAcc(VQAv2NewSplitsVMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_2"


class VQAv2Part3NewSplitsVMistralOpenEndedVQAInContextAcc(VQAv2NewSplitsVMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_3"


class VQAv2Part4NewSplitsVMistralOpenEndedVQAInContextAcc(VQAv2NewSplitsVMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_4"


class VQAv2Part5NewSplitsVMistralOpenEndedVQAInContextAcc(VQAv2NewSplitsVMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_5"


class VQAv2SampleVMistralOpenEndedVQAInContextAcc(VQAv2VMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Sample"


class VQAv2DummyVMistralOpenEndedVQAInContextAcc(VQAv2VMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Dummy"


class VQAv2ChatbotVMistralOpenEndedVQAInContextAcc(VQAv2VMistralOpenEndedVQAInContextAcc):
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


class OKVQAVMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
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


class OKVQANewSplitsVMistralOpenEndedVQAInContextAcc(OKVQAVMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class OKVQASampleVMistralOpenEndedVQAInContextAcc(OKVQAVMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif-Sample"


class OKVQAChatbotVMistralOpenEndedVQAInContextAcc(OKVQAVMistralOpenEndedVQAInContextAcc):
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
    test_query_split_name: str = "validation"
    test_support_split_name: str = "train"
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


class TextVQAVMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
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


class TextVQANewSplitsVMistralOpenEndedVQAInContextAcc(TextVQAVMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/textvqa_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class TextVQASampleVMistralOpenEndedVQAInContextAcc(TextVQAVMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/textvqa-Sample"


class TextVQAChatbotVMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
    dataset_name: str = "textvqa"
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
    test_query_split_name: str = "validation"
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
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class AdVQAVMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
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


class AdVQASampleVMistralOpenEndedVQAInContextAcc(AdVQAVMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/AdVQA_modif-Sample"


class VizWizVMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
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


class VizWizSampleVMistralOpenEndedVQAInContextAcc(VizWizVMistralOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VizWiz-Sample"


class VizWizNewSplitsVMistralOpenEndedVQAInContextAcc(VizWizVMistralOpenEndedVQAInContextAcc):
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


class DocVQAVMistralOpenEndedVQAInContextAnls(VMistralOpenEndedVQAInContext):
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


class DocVQASampleVMistralOpenEndedVQAInContextAnls(DocVQAVMistralOpenEndedVQAInContextAnls):
    dataset_name: str = "HuggingFaceM4/DocumentVQA-Sample"
    metric_kwargs = {
        "metrics": [
            DVQAMetrics.ANLS,
        ]
    }


class DocVQAChatbotVMistralOpenEndedVQAInContextAnls(DocVQAVMistralOpenEndedVQAInContextAnls):
    prompt_templates_dict_instruct = {
        0: {
            "prefix": "{bos_token}",
            "example": (
                "User:<image>{question}\nGive a very brief"
                " answer.<end_of_utterance>\nAssistant: {answer}<end_of_utterance>\n"
            ),
        },
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class MMMUVMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
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


class MathVistaVMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
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


class MMVETChatbotVMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
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


class LLaVAWildChatbotVMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
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


class ChartQAChatbotVMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
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
    }
    selected_prompt_template_id = 0
    bool_instruct_templates = True


class GQAChatbotVMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
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


class MathVistaOpenEndedChatbotVMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
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


class MMMUOpenEndedChatbotVMistralOpenEndedVQAInContextAcc(VMistralOpenEndedVQAInContext):
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
