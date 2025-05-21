This folder traces the exploration of additional cleaning that could be brought to the CM4 dataset.

As a result of this exploration phase, 2 potential improvements have been identified:
1. Remove HTML nodes (and their descendants) whose tag class attribute value contains either "footer" or "site-info". From the exploration, this would correspond to "web" parts of the web page
2. Splitting the html at the level of the continue reading occurrence, which is often characterized by the fact that the class attribute value of the tag contains "more-link".

**Before being fully implemented**, we tested the suitability of 2. by creating a filtered version of CM4 that excluded all documents that would have had an occurance of continuous reading (`04_get_banned_url.slurm` and `05_filter_cm4.slurm`).

The explore folder contains streamlint spaces that have been used to find new possible cleaning rules.
