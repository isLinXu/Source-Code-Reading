from typing import Any, TYPE_CHECKING

from datasets import load_dataset
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import Task
from distilabel.llms import vLLM

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns
    from distilabel.steps.tasks.typing import FormattedInput

SYSTEM_PROMPT_NEWS_SUMMARIZATION = """
You are an AI assistant specialized in creating concise, accurate, and objective summaries of news articles. Your task is to produce abstractive summaries that capture the essence of the original content while adhering to the following guidelines:

1. Accuracy: Ensure all information in the summary is factually correct and faithful to the original article.
2. Objectivity: Maintain a neutral tone. Do not inject personal opinions or biases into the summary.
3. Conciseness: Aim for summaries that are about 10-15% of the original article's length, unless otherwise specified.
4. Main ideas: Identify and prioritize the most important information, events, or arguments from the article.
5. Context: Provide essential background information needed to understand the significance of the news.
6. Key elements: Include relevant names, dates, locations, and organizations central to the story.
7. Chronology: Clearly convey the sequence of events if relevant to the story.
8. Causality: Highlight cause-and-effect relationships between events or actions when applicable.
9. Multiple perspectives: If the original article presents different viewpoints, include a balanced representation of these in the summary.
10. Language: Use clear, concise language accessible to a general audience while maintaining an appropriate journalistic tone.
11. Structure: Ensure the summary flows logically and maintains a clear narrative structure.
12. Abstraction: Go beyond simply extracting sentences. Rephrase and combine ideas to create a truly abstractive summary.

When presented with a news article, analyze its content, identify the key information, and produce a summary that adheres to these guidelines.
""".lstrip()


class NewsSummarization(Task):
    @property
    def inputs(self) -> "StepColumns":
        return ["article"]

    def format_input(self, input: dict[str, Any]) -> "FormattedInput":
        return [
            {"role": "system", "content": SYSTEM_PROMPT_NEWS_SUMMARIZATION},
            {"role": "user", "content": input["article"]},
        ]

    @property
    def outputs(self) -> "StepColumns":
        return ["summary"]

    def format_output(
        self, output: str | None, input: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if output is None:
            return {"summary": None}

        return {"summary": output}


with Pipeline(name="email-summaries") as pipeline:
    NewsSummarization(
        llm=vLLM(
            model="Qwen/Qwen2.5-72B-Instruct",
            extra_kwargs={
                "tensor_parallel_size": 8,
                "max_model_len": 4096,
                "enable_prefix_caching": True,
            },
            generation_kwargs={
                "max_new_tokens": 256,
                "temperature": 0.2,
                "top_p": 0.9,
            },
        ),
        resources=StepResources(gpus=8),
        input_batch_size=1000,
    )

if __name__ == "__main__":
    def clean_article(row: dict[str, Any]) -> dict[str, Any]:
        article: str = row["article"]

        # Remove prefix "WASHINGTON (CNN) -- ..."
        body = article.split("-- ", maxsplit=1)
        if len(body) > 1:
            body = body[1]
        else:
            body = body[0]

        # Remove suffix
        body = body.split(" E-mail to a friend")[0]
        row["article"] = body

        return row

    dataset = (
        load_dataset("abisee/cnn_dailymail", name="3.0.0", split="train")
        .map(clean_article)
    )

    distiset = pipeline.run(dataset=dataset, use_cache=False)
    distiset.push_to_hub("argilla/cnn-dailymail-summaries", include_script=True, private=True)
