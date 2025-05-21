import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import streamlit as st
from datasets import load_dataset
from tqdm import tqdm


# Useful to add the path to the m4 module to Streamlit
sys.path.append(str(Path(sys.path[0]).parent.absolute().parent.absolute().parent.absolute().parent.absolute()))

from m4.sourcing.data_collection.processors import (
    DOMTreeSimplificator,
    PreExtractionSimplificator,
    TextMediaPairsExtractor,
)


def check_image_quality(media_info):
    """
    Args_ : Media Node
    Returns :
            img_has_good_quality: Boolean indictating there is an image with good quality (defined by its height, width, and aspect ratio)
            w: image width
            h: image height
    """
    w, h = media_info["original_width"], media_info["original_height"]
    img_width_plus_height = w + h
    img_has_good_quality = False
    aspect_ratio = h / (w + 1e-8)
    if w > 64 and h > 64 and 1 / 5 < aspect_ratio < 5:
        img_has_good_quality = True
    return img_has_good_quality, img_width_plus_height


def check_text(media_info):
    """
    Args_ : Media Node
    Returns :
            has_text: Boolean indictating if there is a text that corresponds to the media
            txt_dict: Dictionary mapping each text_length to its text type (filename, alt-text, extracted_text)
    Note:
            All variables are set to 0 if they don't exist in the media node
    """
    has_text = False
    txt_dict = {"formatted_filename": 0, "alt_text": 0, "extracted_text": 0}
    for text_type in ["formatted_filename", "alt_text", "extracted_text"]:
        try:
            curr_txt_len = len(media_info[text_type])
            txt_dict[text_type] = curr_txt_len
            if curr_txt_len > 0:
                has_text = True
        except Exception:
            pass

    return has_text, txt_dict


def check_CLIP(media_info):
    """
    Args_ : Media Node
    Returns :
            clip_score_max_per_img: Max CLIP score per Image
            clip_nbr_per_img: Number of CLIP scores for a given image
            clip_dict: Dictionary mapping each CLIP score to its text type (filename, alt-text, extracted_text).
    Note:
            All variables are set to 0 if they don't exist in the media node
    """
    clip_score_max_per_img = 0
    clip_nbr_per_img = 0
    clip_dict = {"formatted_filename": 0, "alt_text": 0, "extracted_text": 0}
    clip_var_list = ["formatted_filename", "alt_text", "extracted_text"]
    for i, clip_type in enumerate(
        ["clip_score_image_formatted_filename", "clip_score_image_alt_text", "clip_score_image_extracted_text"]
    ):
        try:
            curr_clip = media_info[clip_type]
            clip_dict[clip_var_list[i]] = curr_clip
            if curr_clip > clip_score_max_per_img:
                clip_score_max_per_img = curr_clip
            if curr_clip > 0:
                clip_nbr_per_img += 1
        except Exception:
            pass

    return clip_score_max_per_img, clip_nbr_per_img, clip_dict


def update_df_metrics_and_lists_for_extraction_method(
    media_info, aggregate_metrics_df, image_centric_df, text_centric_df, extraction_method_name
):
    """_summary_
    Given a Media_Node and the Extraction_Method_Name used to get this Media_Node,
    this function uses the Media_Node's values to update the 2D Dataframes' numbers
    and append values to the 3D Dataframes' lists.
    """

    img_has_good_quality, img_width_plus_height = check_image_quality(media_info)
    has_text, txt_dict = check_text(media_info)

    aggregate_metrics_df[extraction_method_name]["images_nbr"] += 1
    aggregate_metrics_df[extraction_method_name]["images_of_quality_nbr"] += 1 if img_has_good_quality else 0
    aggregate_metrics_df[extraction_method_name]["images_with_txt_pair_nbr"] += 1 if has_text else 0

    image_centric_df[extraction_method_name]["images_width_plus_height"].append(img_width_plus_height)

    if use_clip_scores:
        clip_score_max_per_img, clip_nbr_per_img, clip_dict = check_CLIP(media_info)
        image_centric_df[extraction_method_name]["clip_max_per_img"].append(clip_score_max_per_img)
        image_centric_df[extraction_method_name]["clip_nbr_per_img"].append(clip_nbr_per_img)

        for key in ["formatted_filename", "alt_text", "extracted_text"]:
            text_centric_df[extraction_method_name]["len_" + key].append(txt_dict[key])
            text_centric_df[extraction_method_name]["clip_" + key].append(clip_dict[key])
    else:
        for key in ["formatted_filename", "alt_text", "extracted_text"]:
            text_centric_df[extraction_method_name]["len_" + key].append(txt_dict[key])
            text_centric_df[extraction_method_name]["clip_" + key].append(0)

    return aggregate_metrics_df, image_centric_df, text_centric_df


def get_extraction_evaluation_metrics(
    num_docs_to_consider=100,
    use_clip_scores=True,
):
    """_summary_

    Args:
        num_docs_to_consider (int, optional): _description_. Defaults to 100.
        use_clip_scores (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    dataset = load_dataset(
        "bs-modeling-metadata/c4-en-html-with-metadata",
        streaming=True,
        split="train",
        use_auth_token=True,
    )
    dataset = list(dataset.take(num_docs_to_consider))

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
        extract_clip_scores=use_clip_scores,
    )

    # List of all extraction methods considered
    extraction_methods_names = ["DOM", "Residual"]
    # Time variable
    avg_extraction_time = 0
    # Dataframe containing metrics aggregated on the fly
    aggregate_metrics_df = pd.DataFrame(
        0,
        columns=extraction_methods_names,
        index=["images_nbr", "images_of_quality_nbr", "images_with_txt_pair_nbr"],
    )
    # Dataframe containing lists of datapoints collected for each image
    image_centric_df = pd.DataFrame(
        columns=extraction_methods_names, index=["clip_nbr_per_img", "clip_max_per_img", "images_width_plus_height"]
    )
    # Dataframe containing lists of datapoints collected for each text
    text_centric_df = pd.DataFrame(
        columns=extraction_methods_names,
        index=[
            "len_formatted_filename",
            "len_alt_text",
            "len_extracted_text",
            "clip_formatted_filename",
            "clip_alt_text",
            "clip_extracted_text",
        ],
    )
    # Best way I found so far to set-up 3D dataframes
    for col in image_centric_df:
        for row in list(image_centric_df.index.values):
            image_centric_df[col][row] = []
    for col in text_centric_df:
        for row in list(text_centric_df.index.values):
            text_centric_df[col][row] = []

    # For each page in the dataset, extract media content and update metrics
    for i, example in enumerate(tqdm(dataset)):
        html_str = example["html"]
        url = example["url"]
        start_time_extraction = time.time()
        DOM_and_residual_content = extractor(html_str, url)
        end_time_extraction = time.time()
        avg_extraction_time += end_time_extraction - start_time_extraction

        # Set-up all the different extractions methods and list them
        # TODO: Add Filtered extraction_method
        simple_DOM_tree_imgs = [
            media_info for media_info in DOM_and_residual_content if media_info["image_in_simplified_dom_tree"]
        ]
        residual_images = [
            media_info for media_info in DOM_and_residual_content if not media_info["image_in_simplified_dom_tree"]
        ]

        extraction_methods_dict = {
            extraction_methods_names[0]: simple_DOM_tree_imgs,
            extraction_methods_names[1]: residual_images,
        }

        # Each of those lists will lead to an update of the column
        # corresponding to their extraction method in each dataframe
        for extraction_method_name in extraction_methods_dict:
            media_list = extraction_methods_dict[extraction_method_name]
            for media_info in media_list:
                (
                    aggregate_metrics_df,
                    image_centric_df,
                    text_centric_df,
                ) = update_df_metrics_and_lists_for_extraction_method(
                    media_info, aggregate_metrics_df, image_centric_df, text_centric_df, extraction_method_name
                )
    avg_extraction_time = avg_extraction_time / len(dataset)

    return extraction_methods_names, aggregate_metrics_df, image_centric_df, text_centric_df, avg_extraction_time


class Visualization:
    def __init__(self, num_docs, use_clip_scores=True):
        self.num_docs = num_docs
        self.use_clip_scores = use_clip_scores

        (
            self.extraction_methods_names,
            self.aggregate_metrics_df,
            self.image_centric_df,
            self.text_centric_df,
            self.avg_extraction_time,
        ) = get_extraction_evaluation_metrics(
            num_docs_to_consider=self.num_docs,
            use_clip_scores=self.use_clip_scores,
        )
        self.df_aggregate_metric_names = list(self.aggregate_metrics_df.index.values)
        self.df_image_centric_metric_names = list(self.image_centric_df.index.values)
        self.df_text_centric_metric_names = list(self.text_centric_df.index.values)

    def visualize(self):
        self.image_text_pair_recall()
        self.images_quality_recall()
        self.display_bar_charts(
            header="Image quality and quantity comparisons",
            list_metric_to_compare=["images_nbr", "images_of_quality_nbr", "images_width_plus_height"],
        )
        self.display_distribution_plot(
            list_extraction_methods=self.extraction_methods_names,
            list_metric_to_compare=["images_width_plus_height"],
            title="Distribution of Images' Dimensions (width + height)",
            bin_size=[50, 50],
            max_value=2500,
        )
        if self.use_clip_scores:
            self.display_distribution_plot(
                list_extraction_methods=self.extraction_methods_names,
                list_metric_to_compare=["clip_max_per_img"],
                title="Distribution of Max CLIP Scores",
                bin_size=[0.02, 0.02],
                max_value=None,
            )
            self.display_distribution_plot(
                list_extraction_methods=["DOM"],
                list_metric_to_compare=["clip_formatted_filename", "clip_alt_text", "clip_extracted_text"],
                title="Distribution of CLIP Score per Text Type in DOM extraction",
                bin_size=[0.02, 0.02, 0.02],
                max_value=None,
            )

        self.display_distribution_plot(
            list_extraction_methods=["DOM"],
            list_metric_to_compare=["len_formatted_filename", "len_alt_text", "len_extracted_text"],
            title="Distribution of Text Types Lengths in DOM extraction",
            bin_size=[5, 5, 5],
            max_value=200,
        )

    def image_text_pair_recall(self):
        st.header("Most important stats:")
        avg_extraction_time = self.avg_extraction_time
        DOM_Recall = self.get_percentage_ratio(
            ratio_numerator=self.aggregate_metrics_df["DOM"]["images_with_txt_pair_nbr"],
            denominator=self.aggregate_metrics_df["DOM"]["images_nbr"],
        )
        res_Recall = self.get_percentage_ratio(
            ratio_numerator=self.aggregate_metrics_df["Residual"]["images_with_txt_pair_nbr"],
            denominator=self.aggregate_metrics_df["Residual"]["images_nbr"],
        )
        DOM_General_Recall = self.get_percentage_ratio(
            ratio_numerator=self.aggregate_metrics_df["DOM"]["images_with_txt_pair_nbr"],
            denominator=self.aggregate_metrics_df["DOM"]["images_nbr"]
            + self.aggregate_metrics_df["Residual"]["images_nbr"],
        )
        st.write(f"Avg Extraction Time per page: {avg_extraction_time:.2f}sec")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="DOM Recall over DOM Images", value=f"{DOM_Recall:.2f}%")
        col2.metric(label="Residual Recall over Residual Images", value=f"{res_Recall:.2f}%")
        col3.metric(label="DOM Recall over All Images", value=f"{DOM_General_Recall:.2f}%")

    def images_quality_recall(self):
        col1, col2, col3 = st.columns(3)
        DOM_quality_images__DOM_images = self.get_percentage_ratio(
            ratio_numerator=self.aggregate_metrics_df["DOM"]["images_of_quality_nbr"],
            denominator=self.aggregate_metrics_df["DOM"]["images_nbr"],
        )
        res_quality_images__res_images = self.get_percentage_ratio(
            ratio_numerator=self.aggregate_metrics_df["Residual"]["images_of_quality_nbr"],
            denominator=self.aggregate_metrics_df["Residual"]["images_nbr"],
        )
        DOM_quality_images__All_images = self.get_percentage_ratio(
            ratio_numerator=self.aggregate_metrics_df["DOM"]["images_of_quality_nbr"],
            denominator=self.aggregate_metrics_df["DOM"]["images_of_quality_nbr"]
            + self.aggregate_metrics_df["Residual"]["images_of_quality_nbr"],
        )

        col1.metric(label="DOM Quality Images / DOM images", value=f"{DOM_quality_images__DOM_images:.2f}%")
        col2.metric(label="Residual Quality Images / Residual images", value=f"{res_quality_images__res_images:.2f}%")
        col3.metric(label="DOM Quality Images / All Quality images", value=f"{DOM_quality_images__All_images:.2f}%")

    # Helper Methods
    def get_percentage_ratio(self, ratio_numerator=0, denominator=1):
        ratio_in_percentage = ratio_numerator / denominator * 100
        return ratio_in_percentage

    def get_bar_chart_from_aggregate_metrics(
        self, metrics_values, metrics_names, x_label_categories, x_label, y_label, bar_size=30
    ):
        df = pd.DataFrame(
            {
                y_label: metrics_values,
                x_label: x_label_categories,
                "Caption": metrics_names,
            }
        )
        chart = alt.Chart(df).mark_bar(size=bar_size).encode(x=f"{x_label}:N", y=f"{y_label}:Q", color="Caption:N")

        return chart

    def get_chart_infos_from_3D_df(self, df, col_list, rows_list, reduction="mean"):
        lists = []
        names_list = []
        for col in col_list:
            for row in rows_list:
                if reduction == "mean":
                    lists.append(np.mean(df[col][row]))
                    list_name = col + "_avg_" + row

                names_list.append(list_name)
        return np.array(lists), np.array(names_list)

    def get_dist_infos_from_3D_df(self, df, col_list, rows_list, max_value=None):
        lists = []
        names_list = []
        for col in col_list:
            for row in rows_list:
                if max_value:
                    list_to_append = [el if el < max_value else max_value for el in df[col][row]]
                    lists.append(list_to_append)
                else:
                    lists.append(df[col][row])
                list_name = col + "_" + row
                names_list.append(list_name)
        return lists, names_list

    def display_distribution_plot(
        self, list_extraction_methods, list_metric_to_compare, title=None, bin_size=[1, 1], max_value=None
    ):
        if list_metric_to_compare[0] in self.df_image_centric_metric_names:
            lists_of_metrics_to_plot, list_of_metric_names_to_plot = self.get_dist_infos_from_3D_df(
                self.image_centric_df,
                col_list=list_extraction_methods,
                rows_list=list_metric_to_compare,
                max_value=max_value,
            )
        elif list_metric_to_compare[0] in self.df_text_centric_metric_names:
            lists_of_metrics_to_plot, list_of_metric_names_to_plot = self.get_dist_infos_from_3D_df(
                self.text_centric_df,
                col_list=list_extraction_methods,
                rows_list=list_metric_to_compare,
                max_value=max_value,
            )
        fig = ff.create_distplot(lists_of_metrics_to_plot, list_of_metric_names_to_plot, bin_size=bin_size)
        if title:
            fig.update_layout(title_text=title)
        st.plotly_chart(fig, use_container_width=True)

    def display_bar_charts(self, header, list_metric_to_compare):
        """
        Given a list of metrics to compare, makes one bar chart per metric and compares over
        all extraction methods.
        Each bar chart has its own column, so it is better to put no more than 3 metrics.
        """
        charts = []
        for metric_to_compare in list_metric_to_compare:
            if metric_to_compare in self.df_aggregate_metric_names:
                chart = self.get_bar_chart_from_aggregate_metrics(
                    [self.aggregate_metrics_df[col][metric_to_compare] for col in self.aggregate_metrics_df],
                    [col + "_" + metric_to_compare for col in self.aggregate_metrics_df],
                    x_label_categories=self.extraction_methods_names,
                    x_label="Extraction methods",
                    y_label=metric_to_compare,
                    bar_size=30,
                )
            else:
                if metric_to_compare in self.df_image_centric_metric_names:
                    metrics_list, metrics_name_list = self.get_chart_infos_from_3D_df(
                        self.image_centric_df, col_list=self.extraction_methods_names, rows_list=[metric_to_compare]
                    )
                elif metric_to_compare in self.df_text_centric_metric_names:
                    metrics_list, metrics_name_list = self.get_chart_infos_from_3D_df(
                        self.text_centric_df, col_list=self.extraction_methods_names, rows_list=[metric_to_compare]
                    )

                chart = self.get_bar_chart_from_aggregate_metrics(
                    metrics_list,
                    metrics_name_list,
                    x_label_categories=self.extraction_methods_names,
                    x_label="Extraction methods",
                    y_label=metric_to_compare,
                    bar_size=30,
                )

            charts.append(chart)

        if header is not None:
            st.header(header)
        columns = st.columns(len(list_metric_to_compare))
        for chart_idx, column in enumerate(columns):
            with column:
                st.altair_chart(charts[chart_idx], use_container_width=True)


if __name__ == "__main__":
    num_docs = 10
    use_clip_scores = True
    visualization = Visualization(num_docs=num_docs, use_clip_scores=use_clip_scores)
    st.set_page_config(layout="wide")
    visualization.visualize()
