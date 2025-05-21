import base64
import json
from io import BytesIO
from multiprocessing import cpu_count

import fasttext
import kenlm
import pandas as pd
import sentencepiece
from datasets import load_from_disk
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

from m4.sourcing.data_collection.processors.web_document_filtering import FilteringFunctions
from m4.sourcing.data_collection.utils.filtering_utils import (
    DIGITS_RE,
    FLAGGED_WORDS,
    NON_PRINTING_CHARACTERS_RE,
    SPECIAL_CHARACTERS,
    STOPWORDS,
    UNICODE_PUNCTUATION,
)


NUM_MAX_EXAMPLES = 100_000

PATH_IMAGE_TEXT_PAIRS_DATASET = "./large_files/laion_2b_en_100k"
SPLIT_IMAGE_TEXT_PAIRS = "train"
PATH_LANG_ID_MODEL = "./large_files/lid.176.bin"
PATH_SENTENCEPIECE_MODEL = "./large_files/en.sp.model"
PATH_KENLM_MODEL = "./large_files/en.arpa.bin"
PATH_COMMON_WORDS = "./large_files/words_oscar.json"  # Find at s3://m4-datasets/trash/common_words.json

NUM_DECIMALS_ROUNDING = 3

PATH_SAVE_DATAFRAME = "./large_files/stats_vis_choose_filtering_params.pkl"


image_text_pairs_dataset = load_from_disk(PATH_IMAGE_TEXT_PAIRS_DATASET)[SPLIT_IMAGE_TEXT_PAIRS]
image_text_pairs_dataset = image_text_pairs_dataset.select(range(min(len(image_text_pairs_dataset), NUM_MAX_EXAMPLES)))

with open(PATH_COMMON_WORDS) as f:
    COMMON_WORDS = json.load(f)
lang_id_model = fasttext.load_model(PATH_LANG_ID_MODEL)
sentencepiece_model = sentencepiece.SentencePieceProcessor()
sentencepiece_model.load(PATH_SENTENCEPIECE_MODEL)
kenlm_model = kenlm.Model(PATH_KENLM_MODEL)


def transform_img(img):
    img = img.convert("RGB")
    img.thumbnail((50, 50))
    with BytesIO() as buffer:
        img.save(buffer, "png")
        base_64_encoding = base64.b64encode(buffer.getvalue()).decode()
    return f'<img src="data:image/png;base64,{base_64_encoding}">'


def get_stats_map_func(example):
    img = example["image"]
    txt = example["text"]
    example["img"] = transform_img(img)
    example["caption"] = txt
    example["original\nwidth"] = img.size[0]
    example["original\nheight"] = img.size[1]
    example["aspect\nratio"] = round(img.size[0] / img.size[1], NUM_DECIMALS_ROUNDING)
    example["number\nwords"] = len(FilteringFunctions.split_on_whitespace(text=txt, new_line=True, tab=True))
    example["character\nrepetition\nratio"] = round(
        FilteringFunctions.compute_character_repetition_ratio(text=txt, character_repetition_length=10),
        NUM_DECIMALS_ROUNDING,
    )
    example["word\nrepetition\nratio"] = round(
        FilteringFunctions.compute_word_repetition_ratio(
            text=txt, strip_characters=SPECIAL_CHARACTERS, word_repetition_length=1
        ),
        NUM_DECIMALS_ROUNDING,
    )
    example["special\ncharacter\nratio"] = round(
        FilteringFunctions.compute_special_character_ratio(text=txt, special_characters=SPECIAL_CHARACTERS),
        NUM_DECIMALS_ROUNDING,
    )
    example["stop\nword\nratio"] = round(
        FilteringFunctions.compute_stopword_ratio(text=txt, strip_characters=SPECIAL_CHARACTERS, stopwords=STOPWORDS),
        NUM_DECIMALS_ROUNDING,
    )
    example["flagged\nword\nratio"] = round(
        FilteringFunctions.compute_flagged_word_ratio(
            text=txt, strip_characters=SPECIAL_CHARACTERS, flagged_words=FLAGGED_WORDS
        ),
        NUM_DECIMALS_ROUNDING,
    )
    example["common\nword\nratio"] = round(
        FilteringFunctions.compute_common_word_ratio(
            text=txt, strip_characters=SPECIAL_CHARACTERS, common_words=COMMON_WORDS
        ),
        NUM_DECIMALS_ROUNDING,
    )
    example["language\nid\nscore"] = round(
        FilteringFunctions.compute_lang_id_pred_score(text=txt, lang_id_model=lang_id_model)[1], NUM_DECIMALS_ROUNDING
    )
    example["perplexity\nscore"] = FilteringFunctions.compute_perplexity_score(
        text=txt,
        non_printing_characters_re=NON_PRINTING_CHARACTERS_RE,
        digits_re=DIGITS_RE,
        unicode_punctuation=UNICODE_PUNCTUATION,
        sentencepiece_model=sentencepiece_model,
        kenlm_model=kenlm_model,
    )
    example["clip\nscore"] = round(json.loads(example["meta"])["similarity"], NUM_DECIMALS_ROUNDING)
    return example


image_text_pairs_dataset = image_text_pairs_dataset.map(
    get_stats_map_func, remove_columns=image_text_pairs_dataset.column_names, num_proc=cpu_count()
)
df = pd.DataFrame(image_text_pairs_dataset)
df.to_pickle(PATH_SAVE_DATAFRAME)
