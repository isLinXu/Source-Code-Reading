import random
from pathlib import Path

import streamlit as st
from datasets import load_from_disk


st.set_page_config(layout="wide")

processed_data_dir = Path("/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION")
original_data_dir = Path("/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML")
shard_id = 30
exclude_shards = [34]

processed_ds_name_2 = (
    processed_data_dir / f"shard_{shard_id}" / "wikipedia_html_enterprise-with-images-and-html-full-v1-v2"
)
shard_ds = load_from_disk(processed_ds_name_2)

shard_ds = shard_ds.filter(lambda x: x["html"] is not None)
num_docs = len(shard_ds)

st.header("Document")
if st.button("Select a random document"):
    dct_idx = random.randint(a=0, b=num_docs - 1)
else:
    dct_idx = 0
idx = st.number_input(
    f"Select a document among the first {num_docs} ones",
    min_value=0,
    max_value=num_docs - 1,
    value=dct_idx,
    step=1,
    help=f"Index between 0 and {num_docs-1}",
)
current_example = shard_ds[idx]
current_html = current_example["html"]


col1, col2 = st.columns(2)
with col1:
    st.subheader("Raw html rendering")
    st.components.v1.html(current_html, height=700, scrolling=True)
with col2:
    st.subheader("Texts and images extracted from the html")
    for text, img in zip(current_example["texts"], current_example["images"]):
        if img is not None:
            st.image(img, caption=text)
        else:
            st.write(text)
