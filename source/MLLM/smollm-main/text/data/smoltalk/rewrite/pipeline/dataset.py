from typing import TYPE_CHECKING, Any

from datasets import concatenate_datasets, load_dataset

if TYPE_CHECKING:
    from datasets import Dataset

REWRITE_CONCISE_SYSTEM_PROMPT = "You're an AI assistant for text re-writing. Rewrite the input text to make it more concise while preserving its core meaning."

REWRITE_FRIENDLY_SYSTEM_PROMPT = "You're an AI assistant for text re-writing. Rewrite the input text to make it more friendly and approachable while maintaining its main points."

REWRITE_PROFESSIONAL_SYSTEM_PROMPT = "You're an AI assistant for text re-writing. Rewrite the input text to make it more professional and formal while retaining its essential content."

CONFIG_NAME_TO_SYSTEM_PROMPT = {
    "default": REWRITE_CONCISE_SYSTEM_PROMPT,
    "unfriendly_email_conversations": REWRITE_FRIENDLY_SYSTEM_PROMPT,
    "unprofessional_email_conversations": REWRITE_PROFESSIONAL_SYSTEM_PROMPT,
}


def get_dataset() -> "Dataset":
    concise = get_finepersonas_emails_dataset("default")
    unfriendly = get_finepersonas_emails_dataset("unfriendly_email_conversations")
    unprofessional = get_finepersonas_emails_dataset(
        "unprofessional_email_conversations"
    )
    linkedin_posts_concise = get_finepersonas_linkedin_posts(
        system_prompt=REWRITE_CONCISE_SYSTEM_PROMPT
    )
    linkedin_posts_professional = get_finepersonas_linkedin_posts(
        system_prompt=REWRITE_PROFESSIONAL_SYSTEM_PROMPT
    )
    tweets_professional = get_finepersonas_tweets(
        system_prompt=REWRITE_PROFESSIONAL_SYSTEM_PROMPT
    )
    return concatenate_datasets(
        [
            concise,
            unfriendly,
            unprofessional,
            linkedin_posts_concise,
            linkedin_posts_professional,
            tweets_professional,
        ]
    )


def get_finepersonas_emails_dataset(
    config_name: str, n: int | None = None
) -> "Dataset":
    def get_first_email(row: dict[str, Any]) -> dict[str, Any]:
        formatted_emails = row["formatted_emails"]
        if not formatted_emails:
            return {"system_prompt": "", "instruction": "", "kind_of_content": ""}

        email = formatted_emails[0]
        return {
            "system_prompt": CONFIG_NAME_TO_SYSTEM_PROMPT[config_name],
            "instruction": email["body"],
            "kind_of_content": "email",
        }

    dataset: "Dataset" = load_dataset(  # type: ignore
        "argilla/FinePersonas-Synthetic-Email-Conversations",
        name=config_name,
        split="train",
    )
    dataset = dataset.map(get_first_email, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: x["system_prompt"] != "")
    if n is not None:
        dataset = select_n(dataset, n)
    return dataset


def get_finepersonas_linkedin_posts(
    n: int | None = None, system_prompt: str = REWRITE_CONCISE_SYSTEM_PROMPT
) -> "Dataset":
    dataset: "Dataset" = load_dataset(  # type: ignore
        "argilla-warehouse/FinePersonas-Extra-Stuff",
        name="linkedin_posts",
        split="train",
    )
    dataset = dataset.select_columns(["formatted_linkedin_post"])
    dataset = dataset.add_column("system_prompt", [system_prompt] * len(dataset))
    dataset = dataset.rename_column("formatted_linkedin_post", "instruction")
    dataset = dataset.add_column("kind_of_content", ["linkedin_post"] * len(dataset))
    if n is not None:
        dataset = select_n(dataset, n)
    return dataset


def get_finepersonas_tweets(
    n: int | None = None, system_prompt: str = REWRITE_CONCISE_SYSTEM_PROMPT
) -> "Dataset":
    dataset: "Dataset" = load_dataset(  # type: ignore
        "argilla-warehouse/FinePersonas-Extra-Stuff", name="tweets", split="train"
    )
    dataset = dataset.select_columns(["formatted_tweet"])
    dataset = dataset.add_column("system_prompt", [system_prompt] * len(dataset))
    dataset = dataset.rename_column("formatted_tweet", "instruction")
    dataset = dataset.add_column("kind_of_content", ["tweet"] * len(dataset))
    if n is not None:
        dataset = select_n(dataset, n)
    return dataset


def select_n(dataset: "Dataset", n: int) -> "Dataset":
    return dataset.select(range(n))
