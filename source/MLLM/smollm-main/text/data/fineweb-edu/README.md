# 📐 FineMath pipeline

![image/png](https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/0GAdY8wZx6bGtUzqX4Lvi.png)


Here you can find the information on the curation of [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) and the code for training its math reasoning [classifier](https://huggingface.co/HuggingFaceTB/finemath-classifier).

## Dataset curation

Recent language models like DeepSeekMath and MathStral have demonstrated strong mathematical capabilities, trained on specialized datasets that aren't publicly available. We developed a pipeline to identify and extract high-quality mathematical content from CommonCrawl, with several iterations of refinement to improve quality.

### Phase 1: Initial content extraction and classification
We began by re-extracting pages from CommonCrawl WARCs using URLs from the FineWeb dataset, collecting both the latest and largest versions of each page to capture the evolution of pages across the years. 
Unlike FineWeb which uses Trafilatura, we employed Resiliparse for text extraction as it better preserves forum discussions and QA answers that often contain crucial reasoning steps and solutions.

For initial quality assessment, we used [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) to generate annotations on a 3-point scale:
1. Contains general mathematical content
2. Shows logical reasoning in mathematical context
3. Contains clear step-by-step solutions at appropriate level

A `multilingual-e5-small`-based classifier finetuned on these annotations was used to score the initial corpus. 
However, this first version performed below the OpenWebMath baseline, leading to several important refinements.

### Phase 2: Recalling more candidate pages
Analysis revealed that FineWeb's C4 filter removes pages containing '{' characters, inadvertently filtering out content with LaTeX notation. To address this and expand coverage, we:

1. Identified promising website domains by selecting those where at least 10% of pages received a classifier score ≥ 2
2. Added URLs from OpenWebMath and InfiMM-WebMath datasets
3. Recovered URLs of pages filtered by FineWeb's '{' rule from its rejection logs
4. Re-extracted all content from scratch using the [OpenWebMath pipeline](https://github.com/keirp/OpenWebMath), which properly handles mathematical notation across various HTML markup formats and standardizes them to LaTeX

### Phase 3: Refined quality assessment
The expanded corpus underwent a more fine-grained quality evaluation:

Once again, we used LLama-3.1-70B-Instruct to score a sample of newly extracted pages on a 5-point scale (full prompt available in [here](assets/prompt.txt)):
We finetuned a new [classifier](https://huggingface.co/HuggingFaceTB/finemath-classifier) on these annotations and scored the entire corpus.
After leaving only pages with a score of 3 or higher, and deduplicating the samples using simple single-band MinHash-LSH, we obtained FineMath-3+ with 34B tokens.

The same classifier was applied to InfiMM-WebMath's text content, focusing more on reasoning rather than advanced mathematics.

Both datasets were additionally filtered using FineWeb's language classification pipeline to remove non-English content.

### Decontamination
Following Qwen2.5-Math's approach, we removed samples with 13-gram overlaps against test sets from GSM8k, MATH, MMLU and ARC. Decontamination logs are available at [HuggingFaceTB/finemath_contamination_report](https://huggingface.co/datasets/HuggingFaceTB/finemath_contamination_report).

## Training the classifier

Todo: share step 2 annotations and finetuning code.