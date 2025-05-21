import time

import plotly.figure_factory as ff
import streamlit as st
import yaml
from datasets import load_dataset
from humanfriendly import format_timespan

from m4.sourcing.data_collection.processors import (
    DOMTreeSimplificator,
    PairFiltering,
    PreExtractionSimplificator,
    TextMediaPairsExtractor,
)


class Visualization:
    def __init__(self, path_config_filter_text_image_pairs):
        with open(path_config_filter_text_image_pairs) as f:
            self.filtering_params = yaml.load(f, Loader=yaml.FullLoader)

    def visualization(self):
        self.set_title()
        self.choose_extraction_options()
        self.extraction()
        self.statistics_without_filtering()
        self.choose_filtering_options()
        self.filtering()
        self.get_statistics_on_extracted_pairs()
        self.analysis_discarded_pairs()

    def set_title(self):
        st.title("Visualization to help balance precision and recall of the extraction strategy")

    def choose_extraction_options(self):
        st.header("Extraction options")
        self.num_docs = st.number_input(
            "Number of documents to perform the extraction on", min_value=1, max_value=1000, value=100
        )
        self.should_compute_clip_scores = st.checkbox(
            "Compute CLIP scores during the extraction (Warning: way slower when checked)", value=False
        )

    def extraction(self):
        self.extracted_pairs = Visualization.get_extracted_pairs(
            num_docs=self.num_docs,
            should_compute_clip_scores=self.should_compute_clip_scores,
        )

    @staticmethod
    @st.cache(suppress_st_warning=True)
    def get_extracted_pairs(num_docs, should_compute_clip_scores):
        st.header("Extraction")

        def load_examples(num_docs):
            dataset = load_dataset(
                "bs-modeling-metadata/c4-en-html-with-metadata",
                streaming=True,
                split="train",
                use_auth_token=True,
            )
            return list(dataset.take(num_docs))

        docs = load_examples(num_docs)

        dom_tree_simplificator = DOMTreeSimplificator(
            strip_multiple_linebreaks=True,
            strip_multiple_spaces=True,
            remove_html_comments=True,
            replace_line_break_tags=True,
            unwrap_tags=True,
            strip_tags=True,
            strip_special_divs=True,
            remove_dates=True,
            remove_empty_leaves=True,
            unnest_nodes=True,
            remake_tree=True,
        )
        pre_extraction_simplificator = PreExtractionSimplificator(
            only_text_image_nodes=True,
            format_texts=True,
            merge_consecutive_text_nodes=True,
        )
        extractor = TextMediaPairsExtractor(
            dom_tree_simplificator=dom_tree_simplificator,
            pre_extraction_simplificator=pre_extraction_simplificator,
            also_extract_images_not_in_simplified_dom_tree=True,
            extract_clip_scores=should_compute_clip_scores,
        )

        st.markdown("Extraction progress bar")
        extraction_progress_bar = st.progress(0.0)

        def extract_pairs(doc, ind):
            extracted_pairs = extractor(html_str=doc["html"], page_url=doc["url"])
            extraction_progress_bar.progress((ind + 1) / num_docs)
            return extracted_pairs

        start_extraction_time = time.time()

        extracted_pairs = [extract_pairs(doc, ind) for ind, doc in enumerate(docs)]
        extracted_pairs = [sub_el for el in extracted_pairs for sub_el in el]

        end_extraction_time = time.time()
        extraction_time = format_timespan(round(end_extraction_time - start_extraction_time))
        st.markdown(f"Extraction done in {extraction_time}")
        st.balloons()

        return extracted_pairs

    @staticmethod
    def plot_distributions(hist_data, group_labels, bin_size, title):
        """hist_data is a list of statistics lists (works up to 3, if more is needed, add some colors
        and annotation colors)"""

        def check_same_number_list(list_):
            # Useful to check if a matrix is singular, otherwise Plotly raises an error
            # Lists without any element or with only one are automatically discarded
            if not list_:
                return True
            num = list_[0]
            for el in list_:
                if el != num:
                    return False
            return True

        colors = ["#200CCF", "#DB2A2A", "#0CCF6D"][: len(hist_data)]
        annotation_color = ["#11066B", "#B30707", "#0B7540"][: len(hist_data)]

        count_del = 0
        for i in range(len(hist_data)):
            if check_same_number_list(hist_data[i - count_del]):
                del hist_data[i - count_del]
                del group_labels[i - count_del]
                del colors[i - count_del]
                del annotation_color[i - count_del]
                count_del += 1

        fig = ff.create_distplot(hist_data, group_labels, colors=colors, bin_size=bin_size, show_rug=False)
        fig["layout"].update(title=title)
        for data, color in zip(hist_data, annotation_color):
            fig.add_vline(
                x=sum(data) / len(data), line_width=2, line_dash="dash", line_color=color, annotation_text="x̄"
            )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def truncate(list_, max_val):
        return [el if el < max_val else max_val for el in list_]

    def statistics_without_filtering(self):
        st.header("Statistics (without filtering)")

        def image_sizes(
            max_width=1_000,
            bin_size_width=10,
            max_height=1_000,
            bin_size_height=10,
            max_num_pixels=1_000_000,
            bin_size_num_pixels=1_000,
        ):
            st.subheader("Image sizes")
            col1, col2, col3 = st.columns(3)

            with col1:
                original_widths = Visualization.truncate(
                    [pair["original_width"] for pair in self.extracted_pairs], max_val=max_width
                )
                Visualization.plot_distributions(
                    [original_widths],
                    ["All images"],
                    bin_size=bin_size_width,
                    title="Distribution of original widths of images",
                )

            with col2:
                original_heights = Visualization.truncate(
                    [pair["original_height"] for pair in self.extracted_pairs], max_val=max_height
                )
                Visualization.plot_distributions(
                    [original_heights],
                    ["All pairs"],
                    bin_size=bin_size_height,
                    title="Distribution of original heights of images",
                )

            with col3:
                original_num_pixels = Visualization.truncate(
                    [pair["original_width"] * pair["original_height"] for pair in self.extracted_pairs],
                    max_val=max_num_pixels,
                )
                Visualization.plot_distributions(
                    [original_num_pixels],
                    ["All images"],
                    bin_size=bin_size_num_pixels,
                    title="Distribution of numbers of pixels of images",
                )

        def text_lengths():
            st.subheader("Text lengths")
            Visualization.plot_distributions(
                [
                    [len(pair[text_key].split(" ")) for pair in self.extracted_pairs if text_key in pair]
                    for text_key in ["formatted_filename", "alt_text", "extracted_text"]
                ],
                ["Formatted filename", "Alt text", "Extracted text"],
                bin_size=1,
                title="Distribution of numbers of words",
            )

        def clip_scores():
            if self.should_compute_clip_scores:
                st.subheader("CLIP scores")
                Visualization.plot_distributions(
                    [
                        [
                            pair[f"clip_score_image_{text_key}"]
                            for pair in self.extracted_pairs
                            if f"clip_score_image_{text_key}" in pair
                        ]
                        for text_key in ["formatted_filename", "alt_text", "extracted_text"]
                    ],
                    ["Formatted filename", "Alt text", "Extracted text"],
                    bin_size=0.02,
                    title="Distribution of CLIP scores",
                )

        image_sizes()
        text_lengths()
        clip_scores()

    def choose_filtering_options(self):
        st.header("Filtering options")

        text_keys = ["Formatted filename", "Alt text", "Extracted text"]
        text_key = st.selectbox("Choose the type of text to pair with images", text_keys, index=2)
        self.text_key = text_key.lower().replace(" ", "_")

        st.write("-----")

        self.should_remove_images_not_in_simplified_dom_trees = st.checkbox(
            "Remove images not in simplified DOM trees", value=False
        )

        st.write("-----")

        self.should_remove_images_not_in_valid_formats = st.checkbox("Remove images not in valid formats", value=False)
        if self.should_remove_images_not_in_valid_formats:
            self.valid_formats = st.multiselect(
                "Valid formats",
                options=list(self.filtering_params["valid_formats"]),
                default=self.filtering_params["valid_formats"],
            )

        st.write("-----")

        self.should_remove_images_not_in_valid_sizes = st.checkbox("Remove images not in valid sizes", value=False)
        if self.should_remove_images_not_in_valid_sizes:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                self.original_width_min_cutoff = st.number_input(
                    "Minimum original width",
                    min_value=1,
                    max_value=None,
                    value=self.filtering_params["original_width_min_cutoff"],
                    step=1,
                )
                self.rendered_width_min_cutoff = st.number_input(
                    "Minimum rendered width",
                    min_value=1,
                    max_value=None,
                    value=self.filtering_params["rendered_width_min_cutoff"],
                    step=1,
                )
            with col2:
                self.original_width_max_cutoff = st.number_input(
                    "Maximum original width",
                    min_value=1,
                    max_value=None,
                    value=self.filtering_params["original_width_max_cutoff"],
                    step=1,
                )
                self.rendered_width_max_cutoff = st.number_input(
                    "Maximum rendered width",
                    min_value=1,
                    max_value=None,
                    value=self.filtering_params["rendered_width_max_cutoff"],
                    step=1,
                )
            with col3:
                self.original_height_min_cutoff = st.number_input(
                    "Minimum original height",
                    min_value=1,
                    max_value=None,
                    value=self.filtering_params["original_height_min_cutoff"],
                    step=1,
                )
                self.rendered_height_min_cutoff = st.number_input(
                    "Minimum rendered height",
                    min_value=1,
                    max_value=None,
                    value=self.filtering_params["rendered_height_min_cutoff"],
                    step=1,
                )
            with col4:
                self.original_height_max_cutoff = st.number_input(
                    "Maximum original height",
                    min_value=1,
                    max_value=None,
                    value=self.filtering_params["original_height_max_cutoff"],
                    step=1,
                )
                self.rendered_height_max_cutoff = st.number_input(
                    "Maximum rendered height",
                    min_value=1,
                    max_value=None,
                    value=self.filtering_params["rendered_height_max_cutoff"],
                    step=1,
                )
            self.aspect_ratio_max_cutoff = st.number_input(
                "Maximum aspect ratio",
                min_value=1.0,
                max_value=None,
                value=float(self.filtering_params["aspect_ratio_max_cutoff"]),
                step=0.5,
            )

        st.write("-----")

        self.should_remove_texts_not_in_valid_number_words = st.checkbox(
            "Remove texts not having a valid number of words", value=False
        )
        if self.should_remove_texts_not_in_valid_number_words:
            col1, col2 = st.columns(2)
            with col1:
                self.number_words_min_cutoff = st.number_input(
                    "Minimum number of words",
                    min_value=0,
                    max_value=None,
                    value=self.filtering_params["number_words_min_cutoff"],
                    step=1,
                )
            with col2:
                self.number_words_max_cutoff = st.number_input(
                    "Maximum number of words",
                    min_value=0,
                    max_value=None,
                    value=self.filtering_params["number_words_max_cutoff"],
                    step=1,
                )

        st.write("-----")

        self.should_remove_texts_with_too_high_special_character_ratio = st.checkbox(
            "Remove texts with a too high special character ratio", value=False
        )
        if self.should_remove_texts_with_too_high_special_character_ratio:
            self.special_character_ratio_max_cutoff = st.number_input(
                "Maximum special character ratio",
                min_value=0.0,
                max_value=1.0,
                value=self.filtering_params["special_character_ratio_max_cutoff"],
                step=0.01,
            )

        st.write("-----")

        self.should_remove_texts_with_too_high_repetition_ratio = st.checkbox(
            "Remove texts with a too high repetition ratio", value=False
        )
        if self.should_remove_texts_with_too_high_repetition_ratio:
            self.repetition_ratio_max_cutoff = st.number_input(
                "Maximum repetition ratio",
                min_value=0.0,
                max_value=1.0,
                value=self.filtering_params["repetition_ratio_max_cutoff"],
                step=0.01,
            )

        st.write("-----")

        self.should_remove_pairs_with_too_low_clip_score = st.checkbox(
            "Remove pairs with a too low CLIP score", value=False
        )
        if self.should_remove_pairs_with_too_low_clip_score:
            self.clip_score_min_cutoff = st.number_input(
                "Minimum CLIP score",
                min_value=0.0,
                max_value=1.0,
                value=self.filtering_params["clip_score_min_cutoff"],
                step=0.01,
            )

    def filtering(self):
        def should_keep_pair(pair):
            # pair = media_info
            if self.text_key not in pair:
                return False

            if self.should_remove_images_not_in_simplified_dom_trees:
                if not PairFiltering.check_image_in_simplified_dom_tree(pair):
                    return False

            if self.should_remove_images_not_in_valid_formats:
                if not PairFiltering.check_format(pair, self.valid_formats):
                    return False

            if self.should_remove_images_not_in_valid_sizes:
                if not PairFiltering.check_size_image(
                    pair,
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
                    return False

            if self.should_remove_texts_not_in_valid_number_words:
                if not PairFiltering.check_number_words(
                    pair, self.text_key, self.number_words_min_cutoff, self.number_words_max_cutoff
                ):
                    return False

            if self.should_remove_texts_with_too_high_special_character_ratio:
                if not PairFiltering.check_special_character_ratio(
                    pair, self.text_key, self.special_character_ratio_max_cutoff
                ):
                    return False

            if self.should_remove_texts_with_too_high_repetition_ratio:
                if not PairFiltering.check_repetition_ratio(pair, self.text_key, self.repetition_ratio_max_cutoff):
                    return False

            if self.should_remove_pairs_with_too_low_clip_score:
                if not PairFiltering.check_clip_score(pair, self.text_key, self.clip_score_min_cutoff):
                    return False

            return True

        all_pairs = [[pair, should_keep_pair(pair)] for pair in self.extracted_pairs]
        self.retained_pairs = [pair for pair, keep_pair in all_pairs if keep_pair]
        self.discarded_pairs = [pair for pair, keep_pair in all_pairs if not keep_pair]

    def get_statistics_on_extracted_pairs(self):
        st.header("Statistics for retained and discarded pairs")

        def number_pairs():
            st.subheader("Number of pairs")
            st.markdown(
                "*Retained pairs*: "
                f"**{len(self.retained_pairs)}/{len(self.extracted_pairs)} "
                f"({round(len(self.retained_pairs)/len(self.extracted_pairs)*100, 1)}%)**"
            )
            st.markdown(
                "*Discarded pairs*: "
                f"**{len(self.discarded_pairs)}/{len(self.extracted_pairs)} "
                f"({round(len(self.discarded_pairs)/len(self.extracted_pairs)*100, 1)}%)**"
            )

        def image_sizes(
            max_width=1_000,
            bin_size_width=10,
            max_height=1_000,
            bin_size_height=10,
            max_num_pixels=1_000_000,
            bin_size_num_pixels=1_000,
        ):
            st.subheader("Image sizes")
            col1, col2, col3 = st.columns(3)

            with col1:
                original_widths_retained_pairs = Visualization.truncate(
                    [pair["original_width"] for pair in self.retained_pairs], max_val=max_width
                )
                original_widths_discarded_pairs = Visualization.truncate(
                    [pair["original_width"] for pair in self.discarded_pairs], max_val=max_width
                )
                Visualization.plot_distributions(
                    [original_widths_retained_pairs, original_widths_discarded_pairs],
                    ["Retained pairs", "Discarded pairs"],
                    bin_size=bin_size_width,
                    title="Distribution of original widths of images",
                )

            with col2:
                original_heights_retained_pairs = Visualization.truncate(
                    [pair["original_height"] for pair in self.retained_pairs], max_val=max_height
                )
                original_heights_discarded_pairs = Visualization.truncate(
                    [pair["original_height"] for pair in self.discarded_pairs], max_val=max_height
                )
                Visualization.plot_distributions(
                    [original_heights_retained_pairs, original_heights_discarded_pairs],
                    ["Retained pairs", "Discarded pairs"],
                    bin_size=bin_size_height,
                    title="Distribution of original heights of images",
                )

            with col3:
                original_num_pixels_retained_pairs = Visualization.truncate(
                    [pair["original_width"] * pair["original_height"] for pair in self.retained_pairs],
                    max_val=max_num_pixels,
                )
                original_num_pixels_discarded_pairs = Visualization.truncate(
                    [pair["original_width"] * pair["original_height"] for pair in self.discarded_pairs],
                    max_val=max_num_pixels,
                )
                Visualization.plot_distributions(
                    [original_num_pixels_retained_pairs, original_num_pixels_discarded_pairs],
                    ["Retained pairs", "Discarded pairs"],
                    bin_size=bin_size_num_pixels,
                    title="Distribution of numbers of pixels of images",
                )

        def text_lengths():
            st.subheader("Text lengths")
            num_words_retained_pairs = [
                len(pair[self.text_key].split(" ")) for pair in self.retained_pairs if self.text_key in pair
            ]
            num_words_discarded_pairs = [
                len(pair[self.text_key].split(" ")) for pair in self.discarded_pairs if self.text_key in pair
            ]
            Visualization.plot_distributions(
                [num_words_retained_pairs, num_words_discarded_pairs],
                ["Retained pairs", "Discarded pairs"],
                bin_size=1,
                title=f"Distribution of numbers of words in the {self.text_key.replace('_', ' ')}",
            )

        def clip_scores():
            if self.should_compute_clip_scores:
                st.subheader("CLIP scores")
                clip_scores_retained_pairs = [
                    pair[f"clip_score_image_{self.text_key}"]
                    for pair in self.retained_pairs
                    if f"clip_score_image_{self.text_key}" in pair
                ]
                clip_scores_discarded_pairs = [
                    pair[f"clip_score_image_{self.text_key}"]
                    for pair in self.discarded_pairs
                    if f"clip_score_image_{self.text_key}" in pair
                ]
                Visualization.plot_distributions(
                    [clip_scores_retained_pairs, clip_scores_discarded_pairs],
                    ["Retained pairs", "Discarded pairs"],
                    bin_size=0.02,
                    title=f"Distribution of CLIP scores for the {self.text_key.replace('_', ' ')}",
                )

        number_pairs()
        image_sizes()
        text_lengths()
        clip_scores()

    def analysis_discarded_pairs(self):
        num_discarded_tot = len(self.discarded_pairs)
        perc_discarded_tot = round(num_discarded_tot / len(self.extracted_pairs) * 100, 1)
        st.header(
            f"Analysis of discarded pairs: {num_discarded_tot}/{len(self.extracted_pairs)} ({perc_discarded_tot}%)"
        )

        if not self.discarded_pairs:
            st.markdown("No pair discarded")

        else:
            num_discarded_filter = len([1 for pair in self.discarded_pairs if self.text_key not in pair])
            perc_discarded_filter = round(num_discarded_filter / num_discarded_tot * 100, 1)
            st.markdown(
                "Discarded because of the *chosen type of text not being in pairs*:"
                f" **{num_discarded_filter}/{num_discarded_tot} ({perc_discarded_filter}%)**"
            )

            def display_discarded_by_filter(should_use_filter, func_filter, msg_filter):
                if should_use_filter:
                    num_discarded_filter = len([1 for pair in self.discarded_pairs if not func_filter(pair)])
                    perc_discarded_filter = round(num_discarded_filter / num_discarded_tot * 100, 1)
                    st.markdown(
                        f"Discarded by the filter on *{msg_filter}*:"
                        f" **{num_discarded_filter}/{num_discarded_tot} ({perc_discarded_filter}%)**"
                    )

            display_discarded_by_filter(
                self.should_remove_images_not_in_simplified_dom_trees,
                lambda pair: PairFiltering.check_image_in_simplified_dom_tree(pair),
                "not being in simplified DOM trees",
            )

            display_discarded_by_filter(
                self.should_remove_images_not_in_valid_formats,
                lambda pair: PairFiltering.check_format(pair, self.valid_formats),
                "not being in valid formats",
            )

            display_discarded_by_filter(
                self.should_remove_images_not_in_valid_sizes,
                lambda pair: PairFiltering.check_size_image(
                    pair,
                    self.original_width_min_cutoff,
                    self.original_width_max_cutoff,
                    self.original_height_min_cutoff,
                    self.original_height_max_cutoff,
                    self.rendered_width_min_cutoff,
                    self.rendered_width_max_cutoff,
                    self.rendered_height_min_cutoff,
                    self.rendered_height_max_cutoff,
                    self.aspect_ratio_max_cutoff,
                ),
                "not being in valid image sizes",
            )

            display_discarded_by_filter(
                self.should_remove_texts_not_in_valid_number_words,
                lambda pair: PairFiltering.check_number_words(
                    pair, self.text_key, self.number_words_min_cutoff, self.number_words_max_cutoff
                ),
                "not having a valid number of words",
            )

            display_discarded_by_filter(
                self.should_remove_texts_with_too_high_special_character_ratio,
                lambda pair: PairFiltering.check_special_character_ratio(
                    pair, self.text_key, self.special_character_ratio_max_cutoff
                ),
                "having a too high special character ratio",
            )

            display_discarded_by_filter(
                self.should_remove_texts_with_too_high_repetition_ratio,
                lambda pair: PairFiltering.check_repetition_ratio(
                    pair, self.text_key, self.repetition_ratio_max_cutoff
                ),
                "having a too high repetition ratio",
            )

            display_discarded_by_filter(
                self.should_remove_pairs_with_too_low_clip_score,
                lambda pair: PairFiltering.check_clip_score(pair, self.text_key, self.clip_score_min_cutoff),
                "having a too low CLIP score",
            )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    path_config_filter_text_image_pairs = "./m4/sourcing/data_collection/configs/config_filter_text_image_pairs.yaml"
    visualization = Visualization(path_config_filter_text_image_pairs=path_config_filter_text_image_pairs)
    visualization.visualization()
