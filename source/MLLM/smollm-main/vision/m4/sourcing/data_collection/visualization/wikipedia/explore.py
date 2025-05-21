import random
from functools import partial

import streamlit as st
from datasets import load_dataset
from selectolax.parser import HTMLParser


st.set_page_config(layout="wide")


classes_to_remove = [
    "box-More_footnotes plainlinks metadata ambox ambox-style ambox-More_footnotes",
    "box-Unreferenced plainlinks metadata ambox ambox-content ambox-Unreferenced",
    "multiple-issues-text mw-collapsible",
    "box-Expand_language plainlinks metadata ambox ambox-notice",
]


def filter_doc_with_matching_class(example, classes_to_remove):
    current_html = example["html"]
    tree = HTMLParser(current_html)
    for class_ in classes_to_remove:
        # matche = tree.css_first(f"[class='{class_}']")
        matche = tree.css_first("[role~='note']")
        if matche:
            return True
    return False


@st.experimental_memo  # it is caching but is incredibly slow when N is big.
def load_examples(num_docs):
    dataset = load_dataset("/home/lucile/data/wikipedia/html_enterprise/script/wikipedia.py")[
        "train"
    ]  # load_from_disk("/home/lucile/data/wikipedia/test/arrow")
    # dataset = dataset.select(range(num_docs))
    dataset = dataset.filter(partial(filter_doc_with_matching_class, classes_to_remove=classes_to_remove))
    return [dataset[i] for i in range(min(num_docs, len(dataset)))]


num_docs = 1000

examples = load_examples(num_docs)

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
current_example = examples[idx]
current_html = current_example["html"]

st.components.v1.html(current_html, height=450, scrolling=True)

parser = HTMLParser(current_html)

nodes = parser.css("[role~='note']")
st.header("role~='note'")
if nodes:
    for node in nodes:
        st.components.v1.html(node.html)
        st.write(node.html)

for class_ in classes_to_remove:
    nodes = parser.css(f"[class='{class_}']")
    st.header(f"Class {class_}")
    if nodes:
        for node in nodes:
            st.components.v1.html(node.html)

# nodes = parser.css('[class="box-More_footnotes plainlinks metadata ambox ambox-style ambox-More_footnotes"]')
# "****"
# nodes
# # print out the text content of each node
# for node in nodes:
#     st.components.v1.html(node.html)
