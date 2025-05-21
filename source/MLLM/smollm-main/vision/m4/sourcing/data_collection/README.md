# Data Collection


## Goal of `data_collection`

This folder is aimed to:
- Simplify HTML DOM trees;
- Convert the simplified DOM trees to another structure adapted for an extraction;
- Perform an extraction (either of image-text pairs, or web documents);
- Perform a filtering on the extraction (either on image-text pairs, or web documents);
- Visualize the results.


## Organization of `data_collection`

The sub-folder `processors` contains the files defining the functions to do the big operations:
- The simplification of DOM trees in `dom_tree_simplificator.py`;
- The convertion of DOM trees to a more adapted structure in `pre_extraction_simplificator`;
- The extraction of web documents in `web_document_extractor.py`;
- The filtering of web documents in `web_document_filtering.py`;
- The extraction of pairs in `pair_extractor.py`;
- The filtering of pairs in `pair_filtering.py`.

These files require other functions or external parameters to work, which are defined in the sub-folder `utils`.

The call of these operations from `processors` to obtain outputs (in `outputs`, which is now a bit outdated, or rather `large_files` at the root of the repo) are done in the sub-folder `callers`.

The sub-folder `visualization` contains the streamlit apps to visualize these outputs:
- In `global_visualization.py`, one can view how the simplification of DOM trees affect the HTML codes, the trees, and the rendered webpages. We also visualize web documents and extract image-text pairs with additional information.
- In `web_document_visualization.py`, one can visualize web documents and see the impact of filtering on them.
- In `pair_visualization.py`, one can obtain statistics on the extracted image-text pairs, and see the impact of filtering on these statistics.
- `plot_clib_distrib.py` is used to obtain the distributions of CLIP scores of reference datasets, to compare with our distribution.
- We used `pair_stat_dashboard.py` to obtain a lot of statistics on the pairs at the beginning. This file might not be maintained anymore.

The sub-folder `debug` is used for debugging and could be ignored.


## Explanation of the data collection pipeline


### Starting point for the dataset

We start with the dataset [`c4-en-html-with-metadata`](https://huggingface.co/datasets/bs-modeling-metadata/c4-en-html-with-metadata), which contains 45M of English HTML documents whose url corresponds to C4 examples gathered by the modeling metadata group.

Each example of the dataset contains much information about the metadata, but we are currently only interested in the columns `html` and `url`.

The full dataset is downloaded on Jean Zay at the path `/gpfsscratch/rech/cnw/commun/local_datasets/c4-en-html-with-metadata-arrow/` (5.9T), and the full dataset containing only the columns `html` and `url` is at the path `/gpfsscratch/rech/cnw/urd43gx/c4_en_html` (3.5T).


### From an HTML string to a tree structure

We use [Selectolax](https://github.com/rushter/selectolax) to efficiently parse the HTML strings and create trees. Each node in the tree corresponds to a tag or a text, and we can have information about its attributes.


### Simplifying the DOM trees

With the `processor` `DOMTreeSimplificator`, we make several simplifications to have simplified DOM trees and remove the unnecessary information:
- We remove comments in the HTML strings;
- We replace the tags `<br>`, `<br/>` and `<br />` by a break of line;
- We unwrap a list of tags (see below), meaning that we remove the tags, but we keep the content inside;
- We strip every tag not in a list (see below), meaning that we completely remove both the tags and what’s inside;
- In a complementary way to the previous step, we additionally remove some tags which contains wrong attribute values, for example a `<div>` tag containing an attribute `class` with the value `date`;
- We remove empty nodes;
- We un-nest nodes, meaning that if a parent node only has one child and no text associated, the child becomes the parent.

Tags that we unwrap: `a`, `abbr`, `acronym`, `b`, `bdi`, `bdo`, `big`, `cite`, `code`, `data`, `dfn`, `em`, `font`, `i`, `ins`, `kbd`, `mark`, `q`, `s`, `samp`, `shadow`, `small`, `span`, `strike`, `strong`, `sub`, `sup`, `time`, `tt`, `u`, `var`, `wbr`.

After unwrapping the tags, tags **not** in this list are removed:
- Tags defining a structure : `address`, `article`, `aside`, `blink`, `blockquote`, `body`, `br`, `caption`, `center`, `dd`, `dl`, `dt`, `div`, `figcaption`, `h`, `h1`, `h2`, `h3`, `h4`, `h5`, `h6`, `hgroup`, `html`, `legend`, `main`, `marquee`, `ol`, `p`, `section`, `summary`, `title`, `ul`;
- Tags defining a media: `audio`, `embed`, `figure`, `iframe`, `img`, `object`, `picture`, `video`;
- Tags that could contain an interesting attribute: `source`.

Tags that we could consider if we really want to have a high recall (but we cannot do anything with most of them): `bgsound`, `button`, `canvas`, `col`, `colgroup`, `datalist`, `details`, `dialog`, `dir`, `fieldset`, `form`, `frame`, `frameset`, `head`, `header`, `input`, `li`, `label`, `map`, `nav`, `optgroup`, `option`, `pre`, `select`, `slot`, `svg`, `table`, `tbody`, `td`, `template`, `tfoot`, `th`, `thead`, `tr`, `track`, `xmp`.

For the 3rd point, they are tags that we currently chose to strip, but that we might reconsider later: `table` (and its associated tags), `form`, `li`, `head`, `header`, `nav`. We chose to remove these tags, either because it is hard to transform to a text (how do we transform a `table` to something clear with a linear text?), or because it can contain useful information, but in most cases this is noise of information related to the navigation in the website (`li` for example).


### Having a good structure

With the `processor` `PreExtractionSimplification`, we traverse the simplified DOM tree and append the nodes to a list to have a better structure.

If the node is a media node, we extract the interesting attributes. We check the validity of the source URL, if it’s not valid we discard the media node.

If the node is a text node, then we format it with the following strategy:
- In the HTML string, replace every +2 `\n` with only one `\n`;
- In the HTML string, replace every +2 spaces with only one space;
- In the HTML string, replace `<br> tags` (and their various forms) with `#BR_TAG#`;
- Within a text node, replace every `\n` or `\t` with a space;
- Within a text node, replace every +2 spaces with only one space;
- Within a text node, split on `#BR_TAG#`, strip each element, and merge on `\n`. If the very first and/or last characters of the text are spaces, make sure to keep them.

Then, we have the possibility to merge two consecutive text nodes (and repeat this operation) with the following strategy:
- Append the separation at the end of the first text (if there is any, it will be one space) to a set, and remove it from this text;
- Append the separation at the beginning of the second text (if there is any, it will be one space) to the set, and remove it from this text;
- Consider all the tags that differ from the path of node 1 to the path of node 2. Append to the set the separations induced by each of these tags;
- The biggest separation in the set wins, in this order: `\n\n`, `\n`, space, nothing. Merge the two texts with this separation;
- When a text node cannot be merged anymore (the previous and following nodes are not text nodes), strip the text.


### Intuition behind Selectolax trees and merging text nodes

Selectolax builds trees by creating a new node for each tag present in the HTML document, and a new node for each non-empty text.

For exemple, consider the HTML document:
```
html_str = """
<html>
<body>
<div>
    this is a test
    <h1>Heading</h1>
    <p>
        p-inner
    </p>
    p-trailing
</div>
</body>
</html>
"""
```

When traversing the tree (depth-first) and printing the path of the nodes (which include their tag as the last component of the path), we obtain:
- `.html`
- `.html.head`
- `.html.body`
- `.html.body.-text` (the text is "\n", because there is a line break between `body` and `div`)
- `.html.body.div`
- - `.html.body.div.-text` (the text is "\n\tthis is a test\n\t")
`.html.body.div.h1`
- `.html.body.div.h1.-text` (the text is "Heading")
- `.html.body.div.-text` (the text is "\n\t")
- `.html.body.div.p`
- `.html.body.div.p.-text` (the text is "\n\t\tp-inner\n\t")
- `.html.body.div.-text` (the text is "\n\tp-trailing\n")
- `.html.body.-text` (the text is "\n\n")

Now we want to merge these text nodes.

We first merge the first text node, "\n" at `.html.body.-text`, and the second text node, "\n\tthis is a test\n\t" at `.html.body.div.-text`.

To do that, we follow the strategy by first formatting the two texts, which results in “ “ and " this is a test ".

We notice that in the first text “ “, we have a space separation at the beginning of the text (it is a particular case since it is also at the end here, but considering it at the end would not change the result in the end). So we'll need to keep this space at the beginning of the merged text nodes, and we can remove it from the text 1, which becomes the empty text “”.

We notice that there isn’t any separation at the end of the text 1, but there is a separation at the beginning of the text 2, which is a space. So we add this space to our set of separations, and we remove it from the text 2, which becomes “this is a test “.

Now, we simply have to check the differences between the paths of the two text nodes. We only have `div` which is different, and `div` induces a line break `\n`, so we add “\n” to our set of separation.

Our set of separations includes “ “ and “\n”. The strongest is “\n”, so this will be our separation between the two text nodes, which becomes “ “ (the first leading space that we should not forget about) + “” (text 1) + “\n” (separation) + “this is a test “ (text 2) = “ this is a test “.

This merged text nodes takes the path of the second text nodes, which is `.html.body.div.-text`.

We can now merge this new merged text node “ this is a test “ at .`html.body.div.-text` with “Heading” at `.html.body.div.h1.-text`. This results after the operation in “ this is a test\n\nHeading” at `.html.body.div.h1.-text`.

And so on until merging everything, which results in “ this is a test\n\nHeading\n\np-inner\n\np-trailing\n\n”.

Since we cannot merge this node anymore, we can strip it to obtain “this is a test\n\nHeading\n\np-inner\n\np-trailing”. This is what is rendering by testing online on an [HTML editor](https://htmledit.squarefree.com/).


### Web documents extraction

At this stage, web documents are simply the structure detailed previously, where each node is either a text or an image (or nothing if we couldn’t download the image).

We use the the `processor` `CommonCrawlWebDocumentExtractor` to extract web documents. Before doing this, make sure to [set up a DNS resolver](https://github.com/rom1504/img2dataset#setting-up-a-bind9-resolver).

Performances for 10K documents, on a Mac M1 Pro (all steps are done with multiprocessing with 10 cores) **without** resizing images:
- Step 1 - Extracting and processing the HTML files: 15 secs
- Step 2 - Getting the URLs of all images: < 1 sec
- Step 3 - Downloading the images with `img2dataset`: 4 min
- Step 4 - Create the dataset containing all images (**2 Go**): 7 secs
- Step 5 - Replace the URLs in the dataset obtained after Step 1 with images bytes (image retrieval): 2 secs

On 1M documents on a GCP machine (60 cores):
- Processing the HTML documents, simplifying them, and creating a dataset with the desired structure: 7min
- Download images: 1h24min
- Create the dataset containing all images: 30min
- Retrieve images and add them to the initial dataset: 16min

1964081 out of 4034154 images were successfully downloaded (48.7%).
The dataset containing all images weighs 185G.
The final dataset with texts and images weighs 243G.

On 10M documents on a GCP machine (60 cores):
- Processing the HTML documents, simplifying them, and creating a dataset with the desired structure: 1h03min
- Download images: 11h15min
- Create the dataset containing all images: 6h29min
- Retrieve images and add them to the initial dataset: 2h25min
- Saving the dataset : 1h36min
- Sharding the dataset : 7h29min


### Web document filtering

The filtering of web documents is done at different levels. First, we modify the document with a filtering at node level. Then, we decide if we keep the document with a filtering at document level.

**Node level:**

For each image, filters on:
- The format;
- The size (original and rendered widths and heights, side ratio).

For each paragraph in a text, filters on:
- The number of words;
- The special character ratio;
- The stop word ratio.

**Doc level:**

Filters on:
- The number of images;
- The number of words;
- The character repetition ratio;
- The word repetition ratio;
- The special character ratio;
- The stop word ratio;
- The flagged word ratio;
- The language identification prediction score;
- The perplexity score.


### Image-text pairs extraction

With the `processor` `TextMediaPairsExtractor`, we extract image-text pairs first by looking at images in the list of nodes from the previously described structure. We only keep images that we are able to download. Then, to form pairs with an image, we consider the alt-text (if present), we consider the formatted filename (essentially applying some regexes to prettify the original name of the file), and we consider the extracted text if the node just after the image is a text node. We then split this text on `\n\n` and consider the first element (essentially the first paragraph). We also have the possibility to extract images not present in the simplified DOM tree, but in this case the extracted text is never present.


### Image-text pairs filtering

With the `processor` `PairFiltering`, we essentially filter image-text pairs based on:
- The format of the images;
- The size of the images (original and displayed width, original and displayed height, side ratio);
- The number of words in the texts;
- The special character ratio in the texts;
- The repetition ratio in the texts;
- The CLIP scores.
