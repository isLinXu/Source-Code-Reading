import re

from m4.sourcing.data_collection.utils import NON_PRINTING_CHARACTERS_RE, SPECIAL_CHARACTERS, STOPWORDS


class PairFiltering:
    def __init__(
        self,
        cond_check_image_in_simplified_dom_tree,
        cond_check_format,
        valid_formats,
        cond_check_size_image,
        original_width_min_cutoff,
        original_width_max_cutoff,
        original_height_min_cutoff,
        original_height_max_cutoff,
        rendered_width_min_cutoff,
        rendered_width_max_cutoff,
        rendered_height_min_cutoff,
        rendered_height_max_cutoff,
        aspect_ratio_max_cutoff,
        cond_remove_non_printing_characters,
        cond_standardize_whitespace,
        cond_check_number_words,
        number_words_min_cutoff,
        number_words_max_cutoff,
        cond_check_special_character_ratio,
        special_character_ratio_max_cutoff,
        cond_check_stopword_ratio,
        stopword_ratio_min_cutoff,
        cond_check_repetition_ratio,
        repetition_ratio_max_cutoff,
        cond_check_clip_score,
        clip_score_min_cutoff,
    ):
        self.cond_check_image_in_simplified_dom_tree = cond_check_image_in_simplified_dom_tree
        self.cond_check_format = cond_check_format
        self.valid_formats = valid_formats
        self.cond_check_size_image = cond_check_size_image
        self.original_width_min_cutoff = original_width_min_cutoff
        self.original_width_max_cutoff = original_width_max_cutoff
        self.original_height_min_cutoff = original_height_min_cutoff
        self.original_height_max_cutoff = original_height_max_cutoff
        self.rendered_width_min_cutoff = rendered_width_min_cutoff
        self.rendered_width_max_cutoff = rendered_width_max_cutoff
        self.rendered_height_min_cutoff = rendered_height_min_cutoff
        self.rendered_height_max_cutoff = rendered_height_max_cutoff
        self.aspect_ratio_max_cutoff = aspect_ratio_max_cutoff
        self.cond_remove_non_printing_characters = cond_remove_non_printing_characters
        self.cond_standardize_whitespace = cond_standardize_whitespace
        self.cond_check_number_words = cond_check_number_words
        self.number_words_min_cutoff = number_words_min_cutoff
        self.number_words_max_cutoff = number_words_max_cutoff
        self.cond_check_special_character_ratio = cond_check_special_character_ratio
        self.special_character_ratio_max_cutoff = special_character_ratio_max_cutoff
        self.cond_check_stopword_ratio = cond_check_stopword_ratio
        self.stopword_ratio_min_cutoff = stopword_ratio_min_cutoff
        self.cond_check_repetition_ratio = cond_check_repetition_ratio
        self.repetition_ratio_max_cutoff = repetition_ratio_max_cutoff
        self.cond_check_clip_score = cond_check_clip_score
        self.clip_score_min_cutoff = clip_score_min_cutoff

    def __call__(self, media_info):
        # Make sure that we have access to at least one text
        available_text_keys = set(
            text_key for text_key in ["formatted_filename", "alt_text", "extracted_text"] if text_key in media_info
        )
        if not available_text_keys:
            return None

        # Image filtering
        if self.cond_check_image_in_simplified_dom_tree:
            if not PairFiltering.check_image_in_simplified_dom_tree(media_info):
                return None

        if self.cond_check_format:
            if not PairFiltering.check_format(media_info, self.valid_formats):
                return None

        if self.cond_check_size_image:
            if not PairFiltering.check_size_image(
                media_info,
                self.original_width_min_cutoff,
                self.original_width_max_cutoff,
                self.original_height_min_cutoff,
                self.original_height_max_cutoff,
                self.rendered_width_min_cutoff,
                self.rendered_width_max_cutoff,
                self.rendered_height_min_cutoff,
                self.rendered_height_max_cutoff,
                self.aspect_ratio_max_cutoff,
            ):
                return None

        # Text normalization
        if self.cond_remove_non_printing_characters:
            for text_key in available_text_keys:
                media_info[text_key] = PairFiltering.remove_non_printing_characters(
                    text=media_info[text_key], non_printing_characters_re=NON_PRINTING_CHARACTERS_RE
                )

        if self.cond_standardize_whitespace:
            for text_key in available_text_keys:
                media_info[text_key] = PairFiltering.standardize_whitespace(text=media_info[text_key])

        # Text filtering
        if self.cond_check_number_words:
            for text_key in available_text_keys:
                if not PairFiltering.check_number_words(
                    media_info, text_key, self.number_words_min_cutoff, self.number_words_max_cutoff
                ):
                    available_text_keys.remove(text_key)
            if not available_text_keys:
                return None

        if self.cond_check_special_character_ratio:
            for text_key in available_text_keys:
                if not PairFiltering.check_special_character_ratio(
                    media_info, text_key, self.special_character_ratio_max_cutoff
                ):
                    available_text_keys.remove(text_key)
            if not available_text_keys:
                return None

        if self.cond_check_stopword_ratio:
            for text_key in available_text_keys:
                if not PairFiltering.check_stopword_ratio(media_info, text_key, self.stopword_ratio_min_cutoff):
                    available_text_keys.remove(text_key)
            if not available_text_keys:
                return None

        if self.cond_check_repetition_ratio:
            for text_key in available_text_keys:
                if not PairFiltering.check_repetition_ratio(media_info, text_key, self.repetition_ratio_max_cutoff):
                    available_text_keys.remove(text_key)
            if not available_text_keys:
                return None

        # Text-image filtering
        if self.cond_check_clip_score:
            for text_key in available_text_keys:
                if not PairFiltering.check_clip_score(media_info, text_key, self.clip_score_min_cutoff):
                    available_text_keys.remove(text_key)
            if not available_text_keys:
                return None

        media_info["retained_text_keys_after_filtering"] = list(available_text_keys)
        return media_info

    @staticmethod
    def check_image_in_simplified_dom_tree(media_info):
        if not media_info["image_in_simplified_dom_tree"]:
            return False
        return True

    @staticmethod
    def check_format(media_info, valid_formats):
        if "format" in media_info:
            if media_info["format"] not in valid_formats:
                return False
        return True

    @staticmethod
    def check_size_image(
        media_info,
        original_width_min_cutoff,
        original_width_max_cutoff,
        original_height_min_cutoff,
        original_height_max_cutoff,
        rendered_width_min_cutoff,
        rendered_width_max_cutoff,
        rendered_height_min_cutoff,
        rendered_height_max_cutoff,
        aspect_ratio_max_cutoff,
    ):
        if not (original_width_min_cutoff <= media_info["original_width"] <= original_width_max_cutoff):
            return False
        if not (original_height_min_cutoff <= media_info["original_height"] <= original_height_max_cutoff):
            return False
        if "rendered_width" in media_info:
            if not (rendered_width_min_cutoff <= media_info["rendered_width"] <= rendered_width_max_cutoff):
                return False
        if "rendered_height" in media_info:
            if not (rendered_height_min_cutoff <= media_info["rendered_height"] <= rendered_height_max_cutoff):
                return False
        if not (
            1 / aspect_ratio_max_cutoff
            <= media_info["original_width"] / media_info["original_height"]
            <= aspect_ratio_max_cutoff
        ):
            return False
        return True

    @staticmethod
    def remove_empty_el_from_list(list_):
        return [el for el in list_ if el]

    @staticmethod
    def remove_non_printing_characters(text, non_printing_characters_re):
        return non_printing_characters_re.sub("", text)

    @staticmethod
    def standardize_whitespace(
        text,
        whitespace=[
            " ",
            " ",
            " ",
            " ",
            " ",
            "　",
            " ",
            " ",
            " ",
            " ",
            "￼",
            "",
        ],
    ):
        """There are different whitespace characters."""
        whitespace = set(whitespace)
        text = "".join([char if char not in whitespace else " " for char in text])
        return text

    @staticmethod
    def split_on_whitespace(
        text,
        new_line=False,
        tab=False,
    ):
        """This method also removes concatenated spaces."""
        sep = [" "] + new_line * ["\n"] + tab * ["\t"]
        sep = "|".join(sep)
        split_text = re.split(sep, text)
        split_text = PairFiltering.remove_empty_el_from_list(split_text)
        return split_text

    @staticmethod
    def strip(text, strip_characters):
        """Way faster than text.strip(strip_characters)
        since strip_characters is a set instead of a str,
        and it contains a lot of elements (all the emojis)."""
        if not text:
            return text
        beg_ind = 0
        end_ind = len(text)
        for i in range(len(text)):
            if text[i] in strip_characters:
                beg_ind += 1
            else:
                break
        for i in range(1, len(text) + 1):
            if text[-i] in strip_characters:
                end_ind -= 1
            else:
                break
        text_stripped = text[beg_ind:end_ind]
        return text_stripped

    @staticmethod
    def get_words_from_text(text, lower_case, strip_words, strip_characters):
        """Get words from a text. Non reversible since the text
        is split on multiple characters, words are stripped of
        special characters and characters are converted to lower case.
        Useful to compute ratios, like the stopword ratio."""
        words = PairFiltering.split_on_whitespace(text, new_line=True, tab=True)
        if lower_case:
            words = [word.lower() for word in words]
        if strip_words:
            words = [PairFiltering.strip(word, strip_characters) for word in words]
            words = PairFiltering.remove_empty_el_from_list(words)
        return words

    @staticmethod
    def check_number_words(media_info, text_key, number_words_min_cutoff, number_words_max_cutoff):
        if text_key not in media_info:
            return True
        words = PairFiltering.get_words_from_text(
            media_info[text_key], lower_case=True, strip_words=True, strip_characters=SPECIAL_CHARACTERS
        )
        number_words = len(words)
        if (number_words < number_words_min_cutoff) or (number_words > number_words_max_cutoff):
            return False
        return True

    @staticmethod
    def compute_special_character_ratio(text, special_characters):
        if len(text) == 0:
            return 0
        special_character_ratio = len([char for char in text if char in special_characters]) / len(text)
        return special_character_ratio

    @staticmethod
    def check_special_character_ratio(media_info, text_key, special_character_ratio_max_cutoff):
        if text_key not in media_info:
            return True
        special_character_ratio = PairFiltering.compute_special_character_ratio(
            text=media_info[text_key], special_characters=SPECIAL_CHARACTERS
        )
        if special_character_ratio > special_character_ratio_max_cutoff:
            return False
        return True

    @staticmethod
    def compute_stopword_ratio(text, stopwords):
        words = PairFiltering.get_words_from_text(
            text, lower_case=True, strip_words=True, strip_characters=SPECIAL_CHARACTERS
        )
        if not words:
            return 0
        stopword_ratio = len([word for word in words if word in stopwords]) / len(words)
        return stopword_ratio

    @staticmethod
    def check_stopword_ratio(media_info, text_key, stopword_ratio_min_cutoff):
        if text_key not in media_info:
            return True
        stopword_ratio = PairFiltering.compute_stopword_ratio(text=media_info[text_key], stopwords=STOPWORDS)
        if stopword_ratio < stopword_ratio_min_cutoff:
            return False
        return True

    @staticmethod
    def compute_repetition_ratio(text):
        words = PairFiltering.get_words_from_text(
            text, lower_case=True, strip_words=True, strip_characters=SPECIAL_CHARACTERS
        )
        if not words:
            return 0
        freq_words = {}
        for word in words:
            freq_words[word] = freq_words.get(word, 0) + 1
        repetition_ratio = sum([freq_words[word] - 1 for word in freq_words]) / len(words)
        return repetition_ratio

    @staticmethod
    def check_repetition_ratio(media_info, text_key, repetition_ratio_max_cutoff):
        if text_key not in media_info:
            return True
        repetition_ratio = PairFiltering.compute_repetition_ratio(media_info[text_key])
        if repetition_ratio > repetition_ratio_max_cutoff:
            return False
        return True

    @staticmethod
    def check_clip_score(media_info, text_key, clip_score_min_cutoff):
        if text_key not in media_info:
            return True
        if f"clip_score_image_{text_key}" in media_info:
            clip_score = media_info[f"clip_score_image_{text_key}"]
            if clip_score < clip_score_min_cutoff:
                return False
        return True
