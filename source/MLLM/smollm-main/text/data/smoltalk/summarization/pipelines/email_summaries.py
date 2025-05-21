from typing import Any, TYPE_CHECKING

import re
from datasets import load_dataset
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import Task
from distilabel.llms import vLLM

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns
    from distilabel.steps.tasks.typing import FormattedInput

SYSTEM_PROMPT_EMAIL_SUMMARIZATION = """
You are an AI assistant designed to summarize emails for the recipient of the email. Your task is to create concise, objective summaries that capture the essential information communicated by the sender, from the recipient's perspective but without directly addressing or mentioning the recipient.

## Key points

1. Do not use phrases like "you" or "the recipient" in the summary.
2. Do not use the recipient name.
3. Do not use the third person.
4. Focus on the sender's actions and intentions.
5. Summarize as if describing the email to a third party.

For example, instead of "Alex is reaching out to you to collaborate" or "Alex is reaching out Samantha to collaborate", use "Alex is reaching out to collaborate".

## Output Requirements

Provide two types of summaries:
1. A maximum brevity summary: extract the main key point of the conversation and present it in one very very short sentence. Include details such as dates, cities, venues, etc if required.
2. A more detailed summary (up to three sentences).

## Output Format:

```markdown
## Maximum brevity summary

[One-sentence summary here]

## Summary

[Up to three-sentence summary here]
```
""".lstrip()

EXTRACT_SUMMARIES_REGEX = re.compile(
    r"## Maximum brevity summary\s+(.*?)\s+## Summary\s+(.*)", re.DOTALL | re.IGNORECASE
)


class EmailSummarization(Task):
    @property
    def inputs(self) -> "StepColumns":
        return ["email"]

    def format_input(self, input: dict[str, Any]) -> "FormattedInput":
        return [
            {"role": "system", "content": SYSTEM_PROMPT_EMAIL_SUMMARIZATION},
            {"role": "user", "content": input["email"]},
        ]

    @property
    def outputs(self) -> "StepColumns":
        return ["maximum_brevity_summary", "summary"]

    def format_output(
        self, output: str | None, input: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if output is None:
            return {"maximum_brevity_summary": None, "summary": None}

        match = EXTRACT_SUMMARIES_REGEX.match(output)
        if not match:
            return {"maximum_brevity_summary": None, "summary": None}

        return {
            "maximum_brevity_summary": match.group(1).strip(),
            "summary": match.group(2).strip(),
        }


with Pipeline(name="email-summaries") as pipeline:
    EmailSummarization(
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
        input_batch_size=1000,
        resources=StepResources(gpus=8)
    )

if __name__ == "__main__":
    dataset = load_dataset(
        "argilla/FinePersonas-Synthetic-Email-Conversations", split="train"
    )

    def explode_emails(rows: dict[str, list[Any]]) -> dict[str, list[Any]]:
        formatted_emails = rows["formatted_emails"]
        exploded_rows = {"conversation_id": [], "email": []}

        for i, emails in enumerate(formatted_emails):
            if not emails:
                continue

            for email in emails:
                subject = email["subject"]
                body = email["body"]
                exploded_rows["conversation_id"].append(i)
                exploded_rows["email"].append(f"Subject: {subject}\n\n{body}")

        return exploded_rows

    dataset = dataset.map(
        explode_emails,
        batched=True,
        remove_columns=dataset.column_names,
    )

    distiset = pipeline.run(dataset=dataset, use_cache=False)

    distiset.push_to_hub("argilla-warehouse/Email-Summaries", include_script=True, private=True)
