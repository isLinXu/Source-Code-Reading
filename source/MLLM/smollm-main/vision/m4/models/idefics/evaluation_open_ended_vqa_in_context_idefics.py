from m4.evaluation.custom_metrics.open_ended_vqa_metrics import OEVQAMetrics
from m4.models.vgpt2.evaluation_open_ended_vqa_in_context_vgpt2 import Vgpt2OpenEndedVQAInContext


class IdeficsOpenEndedVQAInContext(Vgpt2OpenEndedVQAInContext):
    model_class: str = "IdeficsForCausalLM"
    tokenizer_max_seq_len = 2048


class VQAv2IdeficsOpenEndedVQAInContextAcc(IdeficsOpenEndedVQAInContext):
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


class VQAv2NewSplitsIdeficsOpenEndedVQAInContextAcc(VQAv2IdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class VQAv2Part0NewSplitsIdeficsOpenEndedVQAInContextAcc(VQAv2NewSplitsIdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_0"


class VQAv2Part1NewSplitsIdeficsOpenEndedVQAInContextAcc(VQAv2NewSplitsIdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_1"


class VQAv2Part2NewSplitsIdeficsOpenEndedVQAInContextAcc(VQAv2NewSplitsIdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_2"


class VQAv2Part3NewSplitsIdeficsOpenEndedVQAInContextAcc(VQAv2NewSplitsIdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_3"


class VQAv2Part4NewSplitsIdeficsOpenEndedVQAInContextAcc(VQAv2NewSplitsIdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_4"


class VQAv2Part5NewSplitsIdeficsOpenEndedVQAInContextAcc(VQAv2NewSplitsIdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif_support_query_sets_part_5"


class VQAv2SampleIdeficsOpenEndedVQAInContextAcc(VQAv2IdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Sample"


class VQAv2DummyIdeficsOpenEndedVQAInContextAcc(VQAv2IdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Dummy"


class OKVQAIdeficsOpenEndedVQAInContextAcc(IdeficsOpenEndedVQAInContext):
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


class OKVQANewSplitsIdeficsOpenEndedVQAInContextAcc(OKVQAIdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class OKVQASampleIdeficsOpenEndedVQAInContextAcc(OKVQAIdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif-Sample"


class TextVQAIdeficsOpenEndedVQAInContextAcc(IdeficsOpenEndedVQAInContext):
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


class TextVQANewSplitsIdeficsOpenEndedVQAInContextAcc(TextVQAIdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/textvqa_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 9


class TextVQASampleIdeficsOpenEndedVQAInContextAcc(TextVQAIdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/textvqa-Sample"


class AdVQAIdeficsOpenEndedVQAInContextAcc(IdeficsOpenEndedVQAInContext):
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


class AdVQASampleIdeficsOpenEndedVQAInContextAcc(AdVQAIdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/AdVQA_modif-Sample"


class VizWizIdeficsOpenEndedVQAInContextAcc(IdeficsOpenEndedVQAInContext):
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


class VizWizSampleIdeficsOpenEndedVQAInContextAcc(VizWizIdeficsOpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VizWiz-Sample"


class VizWizNewSplitsIdeficsOpenEndedVQAInContextAcc(VizWizIdeficsOpenEndedVQAInContextAcc):
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
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
        1: {
            "prefix": (
                "{bos_token}Instruction: provide an answer to the question. Use the image to answer. Answer"
                " 'unanswerable' if the question is impossible to answer.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
        2: {
            "prefix": (
                "{bos_token}Instruction: provide an answer to the question when possible, otherwise say unanswerable."
                " Use the image to answer.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
        3: {
            "prefix": (
                "{bos_token}Task: Answer the questions based on the image when possible, otherwise say unanswerable.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
        4: {
            "prefix": (
                "{bos_token}Task: Answer the questions based on the image. When it's ambiguous, answer unanswerable.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
        5: {
            "prefix": (
                "{bos_token}Homework:\nExercise:Answer the questions based on the image. When it's ambiguous, answer"
                " unanswerable.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
        6: {
            "prefix": (
                "{bos_token}Exercise:Answer the questions based on the image. When it's ambiguous, answer"
                " unanswerable.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
        7: {
            "prefix": (
                "{bos_token}Homework:\nExercise:When possible, answer the questions based on the image, otherwise say"
                " unanswerable.\n"
            ),
            "example": (
                "Image:{token_around_image}{image_token}{token_around_image}Question: {question} Answer: {answer}\n"
            ),
        },
    }
    selected_prompt_template_id = 3
