import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from m4.sourcing.data_collection.processors import DOMTreeSimplificator
from m4.sourcing.data_collection.utils import (
    INTERESTING_TAGS_SET,
    STRIP_TAGS,
    load_dataset_html,
    make_selectolax_tree,
    simplify_media_node,
)


def get_example(save_file=True):
    # Keys in an example. Only "html" is useful.
    # keys = [
    #     "c4_shard",
    #     "c4_timestamp",
    #     "html",
    #     "url",
    #     "metadata_html",
    #     "text",
    #     "html_footer",
    #     "html_head",
    #     "html_title",
    #     "HtmlPreprocessor_error",
    #     "HtmlPreprocessor_error_comment",
    #     "metadata_url",
    #     "metadata_timestamp",
    #     "metadata_generation_length_text",
    #     "metadata_generation_length_sentence",
    #     "metadata_generation_datasource",
    #     "metadata_website_desc",
    # ]

    dataset = load_dataset_html()
    example = next(dataset)
    html_str = example["html"]

    if save_file:
        f = open("./m4/sourcing/data_collection/outputs/example_html.txt", "w")
        f.write(html_str)
        f.close()


def check_performance_simplification_methods(
    num_docs_to_consider=1000,
):
    dataset = load_dataset_html()
    list_nb_nodes = []
    tot_time = 0
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
    for _ in tqdm(range(num_docs_to_consider)):
        example = next(dataset)
        html_str = example["html"]
        start_time_tree_simplification = time.time()
        selectolax_tree = dom_tree_simplificator(html_str, type_return="selectolax_tree")
        end_time_tree_simplification = time.time()
        tot_time += end_time_tree_simplification - start_time_tree_simplification
        list_nb_nodes.append(sum([1 for node in selectolax_tree.root.traverse()]))
    print("With the current tree simplification strategy:")
    if list_nb_nodes:
        print(
            "Average number of nodes in the html tree:",
            np.mean(list_nb_nodes),
        )
    print("Tree simplification done in %s seconds" % (round(tot_time, 3)))


def check_interesting_tags_removed(
    num_docs_to_consider=1000,
):
    for i in range(len(STRIP_TAGS)):
        print("Iteration:", i)
        print(
            "Additional tag stripped from previous iteration:",
            STRIP_TAGS[i],
        )
        dataset = load_dataset_html()
        num_interesting_nodes = 0
        for _ in range(num_docs_to_consider):
            example = next(dataset)
            html_str = example["html"]
            selectolax_tree = make_selectolax_tree(html_str)
            selectolax_tree.strip_tags(STRIP_TAGS[: i + 1])
            for node in selectolax_tree.root.traverse():
                if node.tag in INTERESTING_TAGS_SET:
                    num_interesting_nodes += 1
        print(
            "Number of interesting nodes found after stripping:",
            num_interesting_nodes,
        )
        print("--------------------")


def count_tags(
    list_tags_to_count,
    simplify_html_tree=True,
    num_docs_to_consider=10000,
    save_file=False,
):
    if save_file:
        dic_count_tags = {el: [] for el in list_tags_to_count}
    else:
        dic_count_tags = {el: 0 for el in list_tags_to_count}

    dataset = load_dataset_html()

    for i in tqdm(range(num_docs_to_consider)):
        example = next(dataset)
        html_str = example["html"]
        if simplify_html_tree:
            tree_simplificator = DOMTreeSimplificator()
            selectolax_tree = tree_simplificator(html_str, type_return="selectolax_tree")
        else:
            selectolax_tree = make_selectolax_tree(html_str)

        for node in selectolax_tree.root.traverse():
            if node.tag in dic_count_tags:
                if save_file:
                    dic_count_tags[node.tag].append(node.html)
                else:
                    dic_count_tags[node.tag] += 1

    if save_file:
        with open("./m4/sourcing/data_collection/outputs/count_tags.json", "w") as f:
            json.dump(dic_count_tags, f)
        dic_count_tags = {key: len(val) for key, val in dic_count_tags.items()}
    print(dic_count_tags)


def get_media(
    num_docs_to_consider=1000,
    simplify_html_tree=True,
    save_file=True,
):
    dataset = load_dataset_html()
    media = {}

    for i in tqdm(range(num_docs_to_consider)):
        example = next(dataset)
        html_str = example["html"]
        url = example["url"]

        if simplify_html_tree:
            tree_simplificator = DOMTreeSimplificator()
            selectolax_tree = tree_simplificator(html_str, type_return="selectolax_tree")
        else:
            selectolax_tree = make_selectolax_tree(html_str)

        for node in selectolax_tree.root.traverse():
            simplified_media_node = simplify_media_node(
                node,
                page_url=url,
            )
            if simplified_media_node:
                media[node.tag] = media.get(node.tag, []) + [simplified_media_node]

    print(
        "Number of documents considered:",
        num_docs_to_consider,
    )
    for key in media:
        print(f"{key}: {len(media[key])}")

    if save_file:
        with open("./m4/sourcing/data_collection/outputs/media.json", "w") as f:
            json.dump(media, f)


def get_distribution_number_words_images():
    number_words_images = []
    with open("outputs/media.json") as f:
        media = json.load(f)
    images = media["img"]
    for image in images:
        if "alt_text" not in image:
            number_words_images.append(0)
        else:
            number_words_images.append(len(image["alt_text"].split(" ")))
    number_words_images = np.array(number_words_images)
    d = np.diff(np.unique(number_words_images)).min()
    left_of_first_bin = number_words_images.min() - float(d) / 2
    right_of_last_bin = number_words_images.max() + float(d) / 2
    plt.hist(
        number_words_images,
        np.arange(left_of_first_bin, right_of_last_bin + d),
        d,
    )
    plt.title("Histogram of the number of words in the alt text of an image")
    plt.show()


def get_url_images(save_file=True):
    url_images = []
    internal_path = 0
    with open("outputs/media.json") as f:
        media = json.load(f)
    images = media["img"]
    for image in images:
        url = image["src"]
        if url.startswith("http"):
            url_images.append(url)
        else:
            internal_path += 1
    print(f"Images with internal path: {internal_path} ({internal_path / (internal_path + len(url_images)) * 100}%)")
    if save_file:
        f = open("./m4/sourcing/data_collection/outputs/url_images.txt", "w")
        f.write("\n".join(url_images))
        f.close()


def get_ratio_success_download_images():
    num_successes = 0
    num_tot = 0
    for file in os.listdir("downloaded_images/"):
        if file.endswith("_stats.json"):
            path_stats = os.path.join("downloaded_images/", file)
            with open(path_stats) as f:
                stats = json.load(f)
                num_successes += stats["successes"]
                num_tot += stats["count"]
    print(f"Num successes when downloading images: {num_successes}/{num_tot} ({num_successes / num_tot * 100}%)")


def get_distribution_size_images():
    original_widths = []
    original_heights = []
    original_sizes = []
    for file in os.listdir("downloaded_images/"):
        path_dir = os.path.join("downloaded_images/", file)
        if os.path.isdir(path_dir):
            for json_file in os.listdir(path_dir):
                if json_file.endswith(".json"):
                    path_json = os.path.join(path_dir, json_file)
                    with open(path_json) as f:
                        json_stats = json.load(f)
                    original_widths.append(json_stats["original_width"])
                    original_heights.append(json_stats["original_height"])
                    original_sizes.append(json_stats["original_width"] * json_stats["original_height"])
    plt.hist(original_widths, bins=100)
    plt.title("Histogram of the original width of images")
    plt.show()
    plt.hist(original_heights, bins=100)
    plt.title("Histogram of the original height of images")
    plt.show()
    plt.hist(original_sizes, bins=1000)
    plt.title("Histogram of the original size (total number of pixels) of images")
    plt.show()

    display_widths = []
    display_heights = []
    display_sizes = []
    with open("./m4/sourcing/data_collection/outputs/media.json") as f:
        media = json.load(f)
    images = media["img"]
    for image in images:
        if (
            ("rendered_width" in image)
            and image["rendered_width"]
            and ("rendered_height" in image)
            and image["rendered_height"]
        ):
            try:
                width = int(image["rendered_width"])
                height = int(image["rendered_height"])
                display_widths.append(width)
                display_heights.append(height)
                display_sizes.append(width * height)
            except Exception:
                pass
    print(len(display_widths))
    print(len(display_heights))
    plt.hist(display_widths, bins=100)
    plt.title("Histogram of the display width of images")
    plt.show()
    plt.hist(display_heights, bins=100)
    plt.title("Histogram of the display height of images")
    plt.show()
    plt.hist(display_sizes, bins=1000)
    plt.title("Histogram of the display size (total number of pixels) of images")
    plt.show()


if __name__ == "__main__":
    pass
