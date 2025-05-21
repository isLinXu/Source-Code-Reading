# Web document filtering documentation

The filtering is done at node and document levels.

At node level, we consider paragraphs or images. This filtering is done to clean the document before doing the filtering at document level, which is deciding if we want to keep the document or not.

Some filters are defined at both node and document levels. If the thresholds were the same for these two levels, it wouldn't be useful to call these filters at document level again, since it would automatically pass the filtering, given the fact that we removed the problematic nodes at node level.

However, the thresholds shouldn't be the same at node and document levels. In a text, at node level, you can have short sentences while at document level, you see the bigger picture. Then, you can be much stricter on the threshold at document level while keeping a high recall than at node level.

https://github.com/huggingface/m4/blob/4e95b234c1206355848faf0e77492717e3e70154/m4/sourcing/data_collection/processors/utils/filtering_utils.py#L37


## Filtering at node level

This operation is done in a `.map`. So we have web documents as inputs, and modified web documents as outputs.

### Filtering at node levels for texts

We start by **modifying** the texts by doing:
- Non printing characters removal (just removing weird characters that are not rendered when displaying a text, but still visible for a computer);
- White space standardization (there are many different white space characters, and we need to standardize them to be able to split on white space characters to obtain the words in a text, useful later for the filtering).

https://github.com/huggingface/m4/blob/4e95b234c1206355848faf0e77492717e3e70154/m4/sourcing/data_collection/processors/web_document_filtering.py#L61

https://github.com/huggingface/m4/blob/4e95b234c1206355848faf0e77492717e3e70154/m4/sourcing/data_collection/processors/web_document_filtering.py#L65

Then, for each text, we split on `\n\n` to obtain paragraphs, and for each paragraph, we apply a filtering on:
- The number of words;
- The character repetition ratio;
- The word repetition ratio;
- The special character ratio;
- The stop word ratio;
- The flagged word ratio;
- The ratio of punctuation characters vs number of words;
- The common word ratio;
- The language identification prediction score;
- The perplexity score.

See details below for the calculations of these quantities.

We remove paragraphs that do not pass the filtering, and join the remaining paragraphs on `\n\n`.


### Filtering at node levels for images

For each image, we filter on:
- The format;
- The size (original and rendered widths and heights, aspect ratio).

https://github.com/huggingface/m4/blob/4e95b234c1206355848faf0e77492717e3e70154/m4/sourcing/data_collection/processors/web_document_filtering.py#L13

https://github.com/huggingface/m4/blob/4e95b234c1206355848faf0e77492717e3e70154/m4/sourcing/data_collection/processors/web_document_filtering.py#L20


## Filtering at document level

This operation is done in a `.filter`. We have the modified web documents as inputs (the outputs of the filtering at node level), and the booleans indicating if we keep the documents or not as outputs.

We filter web documents on:
- The number of images;
- The number of words;
- The character repetition ratio;
- The word repetition ratio;
- The special character ratio;
- The stop word ratio;
- The flagged word ratio;
- The ratio of punctuation characters vs number of words;
- The common word ratio;
- The language identification prediction score;
- The perplexity score.

See details below for the calculations of these quantities.


## Keeping a high diversity

Some web documents, even if they are rare, could be extremely long, or contain an inordinate number of images.

Imagine we have a dataset of 1000 documents, 999 of these documents contain 10 words and 1 image each, and the remaining document contains 10000 words and 1000 images.

Then, we have the feeling that we have diversity in our dataset, since we took 1000 random documents from the internet, but half of the content is from one single document, which is likely to be on the same topic (if it is not spam, which is highly possible).

To remove these outliers (that exist, and take all the place in the dataset), we remove documents with too many words or images.

https://github.com/huggingface/m4/blob/4e95b234c1206355848faf0e77492717e3e70154/m4/sourcing/data_collection/processors/web_document_filtering.py#L51

https://github.com/huggingface/m4/blob/4e95b234c1206355848faf0e77492717e3e70154/m4/sourcing/data_collection/processors/web_document_filtering.py#L137


## Character repetition ratio calculation

For a given $n$, we count the occurrence of each *character* $n$-gram present in the text. We denote $r$ the number of character $n$-grams with at least 2 occurences. We define the character repetition ratio as the ratio of the sum of the $min(k, r)$ largest occurrences ($k$ is defined just below) by the sum of all occurrences, and we discarded texts with a too high ratio.

If $k=1$, short sentences are much more likely to have a high character repetition ratio, since the most frequent character $n$-gram represents a larger proportion of the sentence. If $k$ is the number of occurrences greater than or equal to $2$, very long texts, but not necessarily including repetitions, tend to have a high character repetition ratio, since these texts inherently have a wide diversity of character $n$-grams. $k=\lfloor \sqrt{N} \rfloor $, with $N$ the number of different character $n$-grams found in the text, counterbalances well this effect in practice.

*Example:* Take the sentence `ok_ok_good_ok` and $n=3$. Character $n$-grams, with their frequencies, are given in the following table.

| `ok_` | `_ok` | `k\_o` | `k\_g` | `\_go` | `goo` | `ood` | `od\_` | `d\_o` |
| - | - | - | - | - | - | - | - | - |
| 2 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |


Since we have 9 different character $n$-grams, $N=9$ and $k = \lfloor \sqrt{N} \rfloor =3$.

We have two character $n$-grams with at least two occurrences, so $r=2$.

Then, $min(k, r)=2$.

The sum of the $min(k, r)$ largest occurrences is $2+2=4$ and the sum of all occurrences is $11$. Thus, the character repetition ratio for this sentence is $\frac{4}{11}$.

https://github.com/huggingface/m4/blob/4e95b234c1206355848faf0e77492717e3e70154/m4/sourcing/data_collection/processors/web_document_filtering.py#L167


## Word repetition ratio calculation

As a complement to the previous filter, we remove texts that have commonly repeated similar long sentences. More specifically, we create a filter for the repetitions by looking this time at the occurrences of the *word* $n$-grams, for a chosen $n$ parameter. We define the word repetition ratio as the ratio of the sum of the occurrences greater than or equal to 2 to the sum of all occurrences, and we discard texts with too high of a ratio. Contrary to the filter on the character repetition ratios, I did not find a bias of this method giving systematically higher or lower scores to longer or short texts. This filter is more robust in finding texts with long exact duplicated sentences in them, while the previous one is used to find short to medium sized repetitions.

*Example:* Take the sentence `My name is Hugo. What is your name? My name is Paul.` and $n=2$. Word $n$-grams, with their frequencies, are given in the following table.

| `My name` | `name is` | `is Hugo` | `Hugo What` | `What is` | `is your` | `your name` | `name My` | `is Paul` |
| - | - | - | - | - | - | - | - | - |
| 2 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

We have two word $n$-grams with at least two occurrences, for a total number of $2+2=4$ occurences.
The sum of all occurrences is $11$, so the word repetition ratio for this sentence is $\frac{4}{11}$

https://github.com/huggingface/m4/blob/4e95b234c1206355848faf0e77492717e3e70154/m4/sourcing/data_collection/processors/web_document_filtering.py#L197


## Special character ratio calculation

The list of special characters was defined by taking an existing list of special characters, then finding in many web texts non ASCII characters that were not present in this list and count their occurences, and finally adding the most frequent ones to the original list. Emojis are also added to the list. We simply discard texts with a special character ratio above a certain threshold.

https://github.com/huggingface/m4/blob/4e95b234c1206355848faf0e77492717e3e70154/m4/sourcing/data_collection/processors/web_document_filtering.py#L217


## Stop word ratio calculation

Having a low stop word (or closed class word) ratio in a text is one of the best indicators of a non-human generated content. The list of stop words was built by taking pre-existing lists, for example from Universal Dependencies. We discard texts with a too low closed class word ratio.

https://github.com/huggingface/m4/blob/4e95b234c1206355848faf0e77492717e3e70154/m4/sourcing/data_collection/processors/web_document_filtering.py#L236


## Flagged word ratio calculation

To build a list of flagged words, we used a circular method of 1 step:
- We started with a concatenation of pre-existing (even if not perfect) lists of flagged words found on the internet;
- Then, we computed the flagged word ratios of many documents from the internet, using this list;
- We used these scores to build a database containing the documents with the highest flagged words ratios;
- Then, we manually inspected these documents to discover new words to add to the list;
- Finally, the list was filtered with precise instructions below.

*Instructions for building the lists of flagged words:* Keep only the words associated with porn and systematically used in a sexual context. Remove words that can be used in medical, scientific, colloquial (without referring systematically to porn), or everyday contexts. Remove all insults. Remove all words referring to race or sexual orientation.

We are then able to compute the flagged word ratio of a text and discard it if it is too high.

https://github.com/huggingface/m4/blob/4e95b234c1206355848faf0e77492717e3e70154/m4/sourcing/data_collection/processors/web_document_filtering.py#L255


## Punctuation ratio calculation

With a regular expression, we split a text string into a list of words and punctuations (predefined list of punctuation characters used in English). We then count the ratio between number of punctuations and number of words. We discard texts with too low punctuation ratio, as they are usually indicative of poor quality text.


## Common word ratio calculation

We analyze a large amount of text from the Common Crawl using the Oscar dataset, extracting and counting words and removing those that occur only once. We calculate the common word ratio of a text to identify machine-generated content, removing texts with a low ratio.

https://github.com/huggingface/m4/blob/57bda9f70eec539401046b5127ecdff5ae6b4e71/m4/sourcing/data_collection/processors/web_document_filtering.py#L317


## Language identification prediction score calculation

FastText is used to perform language identification and getting confidence scores for a text. If a score is below a specific threshold, we discard the text.

https://github.com/huggingface/m4/blob/4e95b234c1206355848faf0e77492717e3e70154/m4/sourcing/data_collection/processors/web_document_filtering.py#L279


## Perplexity score calculation

SentencePiece unigram tokenizers followed by KenLM 5-gram models after tokenization were trained on Wikipedia article openings. We discarded texts with too high perplexity scores.

https://github.com/huggingface/m4/blob/4e95b234c1206355848faf0e77492717e3e70154/m4/sourcing/data_collection/processors/web_document_filtering.py#L363
