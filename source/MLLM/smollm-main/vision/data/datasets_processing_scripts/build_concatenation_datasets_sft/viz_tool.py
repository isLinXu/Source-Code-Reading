"""
To deploy: streamlit run viz_tool.py
If on HFC, probably will need to do port forwarding to get it in your local browser.
"""

import glob
import os

import streamlit as st
from datasets import load_from_disk


def display_examples(dataset, num_examples=300):
    st.header(f"Sample Examples - First {num_examples}")

    for i in range(num_examples):
        st.subheader(f"Example {i}")

        example = dataset[i]

        images = [im for im in example["images"] if im is not None]
        for im in images:
            st.image(im, width=250, caption=f"Image dimension: {im.size}")

        texts = example["texts"]
        for turn in texts:
            display_text = f"<strong>User:</strong> {turn['user']}\n".replace("\n", "<br>")
            st.markdown(f"<pre>{display_text}</pre>", unsafe_allow_html=True)
            display_text = f"<strong>Assistant:</strong> {turn['assistant']}\n".replace("\n", "<br>")
            st.markdown(f"<pre>{display_text}</pre>", unsafe_allow_html=True)

        st.divider()


def main():
    st.set_page_config(page_title="Prompted set viewer", layout="wide")

    merged_or_not = st.sidebar.radio(
        label="Multi-turn merged or not?",
        options=["Original", "Multi-turn merged"],
        index=1,
    )
    if merged_or_not == "Original":
        DATASET_FOLDER = "/fsx/hugo/fine_tuning_datasets"
    elif merged_or_not == "Multi-turn merged":
        DATASET_FOLDER = "/fsx/hugo/fine_tuning_datasets_merge_image_individual"
    else:
        DATASET_FOLDER = None

    def load_hf_dataset(dataset_path):
        if dataset_path is None or dataset_path == "":
            return None

        try:
            dataset = load_from_disk(f"{DATASET_FOLDER}/{dataset_path}")
            return dataset
        except Exception:
            return None

    all_datasets = [os.path.basename(p) for p in glob.glob(DATASET_FOLDER + "/*")]
    datasets_list = [""] + all_datasets

    st.sidebar.header("Prompted set viewer")
    selected_dataset = st.sidebar.selectbox(
        label="Select a dataset",
        options=datasets_list,
        index=0,
    )

    dataset = load_hf_dataset(selected_dataset)

    if dataset is not None:
        num_examples = st.sidebar.slider(
            label="Number of examples to display", value=200, min_value=100, max_value=2000
        )
        st.sidebar.write(f"Dataset length: `{len(dataset)}`")
        display_examples(dataset, num_examples=num_examples)


if __name__ == "__main__":
    main()
