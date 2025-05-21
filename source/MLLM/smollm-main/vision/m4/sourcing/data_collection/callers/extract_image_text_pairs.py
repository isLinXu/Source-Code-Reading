import argparse

import jsonlines

from m4.sourcing.data_collection.processors import (
    DOMTreeSimplificator,
    PreExtractionSimplificator,
    TextMediaPairsExtractor,
)
from m4.sourcing.data_collection.utils import load_dataset_html


def extract_image_text_pairs(
    num_min_docs_to_consider=1_000,
    num_min_images_to_consider=1_000,
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
    only_text_image_nodes=True,
    format_texts=True,
    merge_consecutive_text_nodes=True,
    also_extract_images_not_in_simplified_dom_tree=True,
    extract_clip_scores=True,
    print_results=True,
    save_file=True,
    save_txt_format=False,
    return_pairs=False,
):
    dom_tree_simplificator = DOMTreeSimplificator(
        strip_multiple_linebreaks=strip_multiple_linebreaks,
        strip_multiple_spaces=strip_multiple_spaces,
        remove_html_comments=remove_html_comments,
        replace_line_break_tags=replace_line_break_tags,
        unwrap_tags=unwrap_tags,
        strip_tags=strip_tags,
        strip_special_divs=strip_special_divs,
        remove_dates=remove_dates,
        remove_empty_leaves=remove_empty_leaves,
        unnest_nodes=unnest_nodes,
        remake_tree=remake_tree,
    )
    pre_extraction_simplificator = PreExtractionSimplificator(
        only_text_image_nodes=only_text_image_nodes,
        format_texts=format_texts,
        merge_consecutive_text_nodes=merge_consecutive_text_nodes,
    )
    extractor = TextMediaPairsExtractor(
        dom_tree_simplificator=dom_tree_simplificator,
        pre_extraction_simplificator=pre_extraction_simplificator,
        also_extract_images_not_in_simplified_dom_tree=also_extract_images_not_in_simplified_dom_tree,
        extract_clip_scores=extract_clip_scores,
    )

    def extraction_from_one_example(example):
        html_str = example["html"]
        url = example["url"]
        extraction = extractor(html_str, url)
        return extraction

    dataset = load_dataset_html()
    images = []
    current_doc = 0

    while not ((len(images) >= num_min_images_to_consider) and (current_doc >= num_min_docs_to_consider)):
        # As soon as we reach both of the minimum counts, we exit
        example = next(dataset)
        extraction = extraction_from_one_example(example)
        images += extraction
        current_doc += 1
        if current_doc % 100 == 0:
            print(f"Extraction done for {current_doc} documents. Extracted {len(images)} images.")

    if print_results:
        print(f"{len(images)} images extracted")
        print(
            "Number of images with alt text of more than 3 words:",
            sum(1 for image in images if ("alt_text" in image) and len(image["alt_text"].split(" ")) >= 3),
        )
        print(
            "Number of images with text: ",
            sum(1 for image in images if "extracted_text" in image),
        )

    if save_file:
        with jsonlines.open("outputs/image_text_pairs.jsonl", "w") as writer:
            writer.write_all(images)

    if save_txt_format:
        valid_keys = ["document_url", "src", "alt_text", "extracted_text"]
        with open("outputs/image_text_pairs.txt", "w") as f:
            for image in images:
                for key in valid_keys:
                    if key in image:
                        f.write(f"{key}: {image[key]}\n")
                f.write("\n\n\n\n")

    if return_pairs:
        return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracting (text, media) pairs from potentialy simplified HTML DOMs."
    )
    parser.add_argument(
        "--num_min_docs_to_consider",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "--num_min_images_to_consider",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "--strip_multiple_linebreaks",
        action="store_true",
    )
    parser.add_argument(
        "--strip_multiple_spaces",
        action="store_true",
    )
    parser.add_argument(
        "--remove_html_comments",
        action="store_true",
    )
    parser.add_argument(
        "--replace_line_break_tags",
        action="store_true",
    )
    parser.add_argument(
        "--unwrap_tags",
        action="store_true",
    )
    parser.add_argument(
        "--strip_tags",
        action="store_true",
    )
    parser.add_argument(
        "--strip_special_divs",
        action="store_true",
    )
    parser.add_argument(
        "--remove_dates",
        action="store_true",
    )
    parser.add_argument(
        "--remove_empty_leaves",
        action="store_true",
    )
    parser.add_argument(
        "--unnest_nodes",
        action="store_true",
    )
    parser.add_argument(
        "--remake_tree",
        action="store_true",
    )
    parser.add_argument(
        "--only_text_image_nodes",
        action="store_true",
    )
    parser.add_argument(
        "--format_texts",
        action="store_true",
    )
    parser.add_argument(
        "--merge_consecutive_text_nodes",
        action="store_true",
    )
    parser.add_argument(
        "--also_extract_images_not_in_simplified_dom_tree",
        action="store_true",
    )
    parser.add_argument(
        "--extract_clip_scores",
        action="store_true",
    )
    parser.add_argument(
        "--print_results",
        action="store_true",
    )
    parser.add_argument(
        "--save_file",
        action="store_true",
    )
    parser.add_argument(
        "--save_txt_format",
        action="store_true",
    )
    parser.add_argument(
        "--return_pairs",
        action="store_true",
    )
    args = parser.parse_args()

    extract_image_text_pairs(
        num_min_docs_to_consider=args.num_min_docs_to_consider,
        num_min_images_to_consider=args.num_min_images_to_consider,
        strip_multiple_linebreaks=args.strip_multiple_linebreaks,
        strip_multiple_spaces=args.strip_multiple_spaces,
        remove_html_comments=args.remove_html_comments,
        replace_line_break_tags=args.replace_line_break_tags,
        unwrap_tags=args.unwrap_tags,
        strip_tags=args.strip_tags,
        strip_special_divs=args.strip_special_divs,
        remove_dates=args.remove_dates,
        remove_empty_leaves=args.remove_empty_leaves,
        unnest_nodes=args.unnest_nodes,
        remake_tree=args.remake_tree,
        only_text_image_nodes=args.only_text_image_nodes,
        format_texts=args.format_texts,
        merge_consecutive_text_nodes=args.merge_consecutive_text_nodes,
        print_results=args.print_results,
        also_extract_images_not_in_simplified_dom_tree=args.also_extract_images_not_in_simplified_dom_tree,
        extract_clip_scores=args.extract_clip_scores,
        save_file=args.save_file,
        save_txt_format=args.save_txt_format,
        return_pairs=args.return_pairs,
    )
