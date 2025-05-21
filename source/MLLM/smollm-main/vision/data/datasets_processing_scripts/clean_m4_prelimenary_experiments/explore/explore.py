import random
from functools import partial

import streamlit as st
from datasets import load_from_disk
from selectolax.parser import HTMLParser


st.set_page_config(layout="wide")


classes_to_remove = [
    "box-More_footnotes plainlinks metadata ambox ambox-style ambox-More_footnotes",
    "box-Unreferenced plainlinks metadata ambox ambox-content ambox-Unreferenced",
    "multiple-issues-text mw-collapsible",
    "box-Expand_language plainlinks metadata ambox ambox-notice",
]

css_rule = st.text_input("CSS rule", value="[class~='footer']")


def filter_doc_with_matching_class(example, css_rule):
    current_html = example["html"]
    tree = HTMLParser(current_html)
    matche = tree.css_first(css_rule)
    if matche:
        return True
    return False


@st.cache_data  # it is caching but is incredibly slow when N is big.
def load_examples(num_docs, css_rule):
    dataset = load_from_disk(
        "/home/lucile/data/web_document_dataset_45M_sharded_ftered_2_line_deduplicated_with_html/train/shard_215"
    )
    # dataset = dataset.select(range(num_docs))
    dataset = dataset.filter(partial(filter_doc_with_matching_class, css_rule=css_rule), num_proc=32)
    return dataset


num_docs = 1000

dataset = load_examples(num_docs, css_rule=css_rule)
num_docs = len(dataset)

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


current_example = dataset[idx]
current_html = current_example["html"]

st.write(current_example["document_url"])

col_text, col_img = st.columns(2)

with col_text:
    st.header("Text")
    st.write("\n".join([text for text in current_example["texts"] if text is not None]))

with col_img:
    for img in current_example["images"]:
        if img is not None:
            st.image(img)


st.header("HTML")
st.components.v1.html(current_html, height=1000, scrolling=True)

parser = HTMLParser(current_html)

nodes = parser.css(css_rule)
st.header(css_rule)
if nodes:
    for node in nodes:
        st.components.v1.html(node.html)
        st.write(node.html)

# for class_ in classes_to_remove:
#     nodes = parser.css(f"[class='{class_}']")
#     st.header(f"Class {class_}")
#     if nodes:
#         for node in nodes:
#             st.components.v1.html(node.html)

# nodes = parser.css('[class="box-More_footnotes plainlinks metadata ambox ambox-style ambox-More_footnotes"]')
# "****"
# nodes
# # print out the text content of each node
# for node in nodes:
#     st.components.v1.html(node.html)
