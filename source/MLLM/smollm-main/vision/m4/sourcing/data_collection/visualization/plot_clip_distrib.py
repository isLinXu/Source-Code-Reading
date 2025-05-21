import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from m4.sourcing.data_collection.utils import NB_BINS, kl_div


# Reference
red_caps = np.load("./m4/sourcing.data_collection/outputs/clip_scores_red_caps_10000.npy")
sbu_captions = np.load("./m4/sourcing.data_collection/outputs/clip_scores_sbu_captions_10000.npy")
laion_captions = np.load("./m4/sourcing.data_collection/outputs/clip_scores_laion400m_10000.npy")
all = np.concatenate((red_caps, sbu_captions))

fig = go.Figure()
fig.add_trace(go.Histogram(x=red_caps, nbinsx=NB_BINS, histnorm="percent", name="red caps"))
fig.add_trace(go.Histogram(x=sbu_captions, nbinsx=NB_BINS, histnorm="percent", name="sbu captions"))
fig.add_trace(go.Histogram(x=laion_captions, nbinsx=NB_BINS, histnorm="percent", name="laion"))
# fig.add_trace(go.Histogram(x=all, nbinsx=NB_BINS, histnorm="percent", name="Red caps + SBU captions"))

fig.update_layout(barmode="overlay", xaxis_title_text="CLIP score", yaxis_title_text="Frequency %")
fig.update_traces(opacity=0.5)

fig.write_image("./m4/sourcing.data_collection/outputs/distributions_reference.png")

# Extracted
scores_alt = []
scores_text = []
scores_filename = []
with open("./m4/sourcing.data_collection/outputs/image_text_pairs.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        if "clip_score_image_alt_text" in data:
            scores_alt.append(data["clip_score_image_alt_text"])
        if "clip_score_image_extracted_text" in data:
            scores_text.append(data["clip_score_image_extracted_text"])
        if "clip_score_image_formatted_filename" in data:
            scores_filename.append(data["clip_score_image_formatted_filename"])

fig = make_subplots(rows=3, cols=1)
fig.add_trace(go.Histogram(x=all, nbinsx=NB_BINS, histnorm="percent", name="Red caps + SBU captions"), 1, 1)
kl_alt = kl_div(scores_alt, all)
fig.add_trace(go.Histogram(x=scores_alt, nbinsx=NB_BINS, histnorm="percent", name=f"Alt text (KL={kl_alt:.2f})"), 1, 1)

fig.add_trace(go.Histogram(x=all, nbinsx=NB_BINS, histnorm="percent", name="Red caps + SBU captions"), 2, 1)
kl_text = kl_div(scores_text, all)
fig.add_trace(
    go.Histogram(x=scores_text, nbinsx=NB_BINS, histnorm="percent", name=f"Extracted text (KL={kl_text:.2f})"), 2, 1
)

fig.add_trace(go.Histogram(x=all, nbinsx=NB_BINS, histnorm="percent", name="Red caps + SBU captions"), 3, 1)
kl_filename = kl_div(scores_filename, all)
fig.add_trace(
    go.Histogram(
        x=scores_filename, nbinsx=NB_BINS, histnorm="percent", name=f"Extracted filename (KL={kl_filename:.2f})"
    ),
    3,
    1,
)

fig.update_layout(barmode="overlay", xaxis_title_text="CLIP score", yaxis_title_text="Frequency %")
fig.update_traces(opacity=0.5)

fig.write_image("./m4/sourcing.data_collection/outputs/distributions_extracted.png")
