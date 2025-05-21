import json

from m4.sourcing.data_collection.processors import FilteringFunctions


class LaionPairFiltering:
    def __init__(
        self,
        cond_check_size_image,
        original_width_min_cutoff,
        original_width_max_cutoff,
        original_height_min_cutoff,
        original_height_max_cutoff,
        aspect_ratio_max_cutoff,
        cond_check_number_words,
        strip_characters,
        number_words_min_cutoff,
        number_words_max_cutoff,
        cond_check_word_repetition_ratio,
        word_repetition_length,
        word_repetition_max_cutoff,
        cond_check_special_character_ratio,
        special_character_ratio_max_cutoff,
        cond_check_common_word_ratio,
        path_common_words,
        common_word_ratio_min_cutoff,
    ):
        self.cond_check_size_image = cond_check_size_image
        self.original_width_min_cutoff = original_width_min_cutoff
        self.original_width_max_cutoff = original_width_max_cutoff
        self.original_height_min_cutoff = original_height_min_cutoff
        self.original_height_max_cutoff = original_height_max_cutoff
        self.aspect_ratio_max_cutoff = aspect_ratio_max_cutoff
        self.cond_check_number_words = cond_check_number_words
        self.strip_characters = strip_characters
        self.number_words_min_cutoff = number_words_min_cutoff
        self.number_words_max_cutoff = number_words_max_cutoff
        self.cond_check_word_repetition_ratio = cond_check_word_repetition_ratio
        self.word_repetition_length = word_repetition_length
        self.word_repetition_max_cutoff = word_repetition_max_cutoff
        self.cond_check_special_character_ratio = cond_check_special_character_ratio
        self.special_character_ratio_max_cutoff = special_character_ratio_max_cutoff
        self.cond_check_common_word_ratio = cond_check_common_word_ratio
        self.path_common_words = path_common_words
        with open(path_common_words) as f:
            self.common_words = json.load(f)
        self.common_word_ratio_min_cutoff = common_word_ratio_min_cutoff

    def __call__(self, pair):
        text = pair["text"]
        image = pair["image"]
        image_metadata = {"original_width": image.size[0], "original_height": image.size[1]}

        if self.cond_check_size_image:
            if not FilteringFunctions.check_size_image(
                image_metadata=image_metadata,
                original_width_min_cutoff=self.original_width_min_cutoff,
                original_width_max_cutoff=self.original_width_max_cutoff,
                original_height_min_cutoff=self.original_height_min_cutoff,
                original_height_max_cutoff=self.original_height_max_cutoff,
                rendered_width_min_cutoff=None,
                rendered_width_max_cutoff=None,
                rendered_height_min_cutoff=None,
                rendered_height_max_cutoff=None,
                aspect_ratio_max_cutoff=self.aspect_ratio_max_cutoff,
            ):
                return False

        if self.cond_check_number_words:
            if not FilteringFunctions.check_number_words(
                text=text,
                strip_characters=self.strip_characters,
                number_words_min_cutoff=self.number_words_min_cutoff,
                number_words_max_cutoff=self.number_words_max_cutoff,
            ):
                return False

        if self.cond_check_word_repetition_ratio:
            if not FilteringFunctions.check_word_repetition_ratio(
                text=text,
                strip_characters=self.strip_characters,
                word_repetition_length=self.word_repetition_length,
                word_repetition_max_cutoff=self.word_repetition_max_cutoff,
            ):
                return False

        if self.cond_check_special_character_ratio:
            if not FilteringFunctions.check_special_character_ratio(
                text=text,
                special_characters=self.strip_characters,
                special_character_ratio_max_cutoff=self.special_character_ratio_max_cutoff,
            ):
                return False

        if self.cond_check_common_word_ratio:
            if not FilteringFunctions.check_common_word_ratio(
                text=text,
                strip_characters=self.strip_characters,
                common_words=self.common_words,
                common_word_ratio_min_cutoff=self.common_word_ratio_min_cutoff,
            ):
                return False

        return True

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.cond_check_size_image,
                self.original_width_min_cutoff,
                self.original_width_max_cutoff,
                self.original_height_min_cutoff,
                self.original_height_max_cutoff,
                self.aspect_ratio_max_cutoff,
                self.cond_check_number_words,
                self.strip_characters,
                self.number_words_min_cutoff,
                self.number_words_max_cutoff,
                self.cond_check_word_repetition_ratio,
                self.word_repetition_length,
                self.word_repetition_max_cutoff,
                self.cond_check_special_character_ratio,
                self.special_character_ratio_max_cutoff,
                self.cond_check_common_word_ratio,
                self.path_common_words,
                self.common_word_ratio_min_cutoff,
            ),
        )
