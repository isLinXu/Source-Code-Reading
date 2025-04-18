# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from jinja2 import Template
from pydantic import Field, PrivateAttr

from distilabel.errors import DistilabelUserError
from distilabel.steps.tasks.base import Task
from distilabel.utils.chat import is_openai_format
from distilabel.utils.template import check_column_in_template

if TYPE_CHECKING:
    from distilabel.typing import ChatType, StepColumns


class TextGeneration(Task):
    """Text generation with an `LLM` given a prompt.

    `TextGeneration` is a pre-defined task that allows passing a custom prompt using the
    Jinja2 syntax. By default, a `instruction` is expected in the inputs, but the using
    `template` and `columns` attributes one can define a custom prompt and columns expected
    from the text. This task should be good enough for tasks that don't need post-processing
    of the responses generated by the LLM.

    Attributes:
        system_prompt: The system prompt to use in the generation. If not provided, then
            it will check if the input row has a column named `system_prompt` and use it.
            If not, then no system prompt will be used. Defaults to `None`.
        template: The template to use for the generation. It must follow the Jinja2 template
            syntax. If not provided, it will assume the text passed is an instruction and
            construct the appropriate template.
        columns: A string with the column, or a list with columns expected in the template.
            Take a look at the examples for more information. Defaults to `instruction`.
        use_system_prompt: DEPRECATED. To be removed in 1.5.0. Whether to use the system
            prompt in the generation. Defaults to `True`, which means that if the column
            `system_prompt` is defined within the input batch, then the `system_prompt`
            will be used, otherwise, it will be ignored.

    Input columns:
        - dynamic (determined by `columns` attribute): By default will be set to `instruction`.
            The columns can point both to a `str` or a `List[str]` to be used in the template.

    Output columns:
        - generation (`str`): The generated text.
        - model_name (`str`): The name of the model used to generate the text.

    Categories:
        - text-generation

    References:
        - [Jinja2 Template Designer Documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/)

    Examples:
        Generate text from an instruction:

        ```python
        from distilabel.steps.tasks import TextGeneration
        from distilabel.models import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        text_gen = TextGeneration(
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            )
        )

        text_gen.load()

        result = next(
            text_gen.process(
                [{"instruction": "your instruction"}]
            )
        )
        # result
        # [
        #     {
        #         'instruction': 'your instruction',
        #         'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
        #         'generation': 'generation',
        #     }
        # ]
        ```

        Use a custom template to generate text:

        ```python
        from distilabel.steps.tasks import TextGeneration
        from distilabel.models import InferenceEndpointsLLM

        CUSTOM_TEMPLATE = '''Document:
        {{ document }}

        Question: {{ question }}

        Please provide a clear and concise answer to the question based on the information in the document and your general knowledge:
        '''.rstrip()

        text_gen = TextGeneration(
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            ),
            system_prompt="You are a helpful AI assistant. Your task is to answer the following question based on the provided document. If the answer is not explicitly stated in the document, use your knowledge to provide the most relevant and accurate answer possible. If you cannot answer the question based on the given information, state that clearly.",
            template=CUSTOM_TEMPLATE,
            columns=["document", "question"],
        )

        text_gen.load()

        result = next(
            text_gen.process(
                [
                    {
                        "document": "The Great Barrier Reef, located off the coast of Australia, is the world's largest coral reef system. It stretches over 2,300 kilometers and is home to a diverse array of marine life, including over 1,500 species of fish. However, in recent years, the reef has faced significant challenges due to climate change, with rising sea temperatures causing coral bleaching events.",
                        "question": "What is the main threat to the Great Barrier Reef mentioned in the document?"
                    }
                ]
            )
        )
        # result
        # [
        #     {
        #         'document': 'The Great Barrier Reef, located off the coast of Australia, is the world's largest coral reef system. It stretches over 2,300 kilometers and is home to a diverse array of marine life, including over 1,500 species of fish. However, in recent years, the reef has faced significant challenges due to climate change, with rising sea temperatures causing coral bleaching events.',
        #         'question': 'What is the main threat to the Great Barrier Reef mentioned in the document?',
        #         'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
        #         'generation': 'According to the document, the main threat to the Great Barrier Reef is climate change, specifically rising sea temperatures causing coral bleaching events.',
        #     }
        # ]
        ```

        Few shot learning with different system prompts:

        ```python
        from distilabel.steps.tasks import TextGeneration
        from distilabel.models import InferenceEndpointsLLM

        CUSTOM_TEMPLATE = '''Generate a clear, single-sentence instruction based on the following examples:

        {% for example in examples %}
        Example {{ loop.index }}:
        Instruction: {{ example }}

        {% endfor %}
        Now, generate a new instruction in a similar style:
        '''.rstrip()

        text_gen = TextGeneration(
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            ),
            template=CUSTOM_TEMPLATE,
            columns="examples",
        )

        text_gen.load()

        result = next(
            text_gen.process(
                [
                    {
                        "examples": ["This is an example", "Another relevant example"],
                        "system_prompt": "You are an AI assistant specialised in cybersecurity and computing in general, you make your point clear without any explanations."
                    }
                ]
            )
        )
        # result
        # [
        #     {
        #         'examples': ['This is an example', 'Another relevant example'],
        #         'system_prompt': 'You are an AI assistant specialised in cybersecurity and computing in general, you make your point clear without any explanations.',
        #         'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
        #         'generation': 'Disable the firewall on the router',
        #     }
        # ]
        ```
    """

    system_prompt: Union[str, None] = None
    use_system_prompt: bool = Field(default=True, deprecated=True)
    template: str = Field(
        default="{{ instruction }}",
        description=(
            "This is a template or prompt to use for the generation. "
            "If not provided, it is assumed a `instruction` is placed in the inputs, "
            "to be used as is."
        ),
    )
    columns: Union[str, List[str]] = Field(
        default="instruction",
        description=(
            "Custom column or list of columns to include in the input. "
            "If a `template` is provided which needs custom column names, "
            "then they should be provided here. By default it will use `instruction`."
        ),
    )

    _can_be_used_with_offline_batch_generation = True
    _template: Optional["Template"] = PrivateAttr(default=...)

    def model_post_init(self, __context: Any) -> None:
        self.columns = [self.columns] if isinstance(self.columns, str) else self.columns
        super().model_post_init(__context)

    def load(self) -> None:
        super().load()

        for column in self.columns:
            check_column_in_template(column, self.template)

        self._template = Template(self.template)

    def unload(self) -> None:
        super().unload()
        self._template = None

    @property
    def inputs(self) -> "StepColumns":
        """The input for the task is the `instruction` by default, or the `columns` given as input."""
        columns = {column: True for column in self.columns}
        columns["system_prompt"] = False
        return columns

    def _prepare_message_content(self, input: Dict[str, Any]) -> "ChatType":
        """Prepares the content for the template and returns the formatted messages."""
        fields = {column: input[column] for column in self.columns}
        return [{"role": "user", "content": self._template.render(**fields)}]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        # Handle the previous expected errors, in case of custom columns there's more freedom
        # and we cannot check it so easily.
        if self.columns == ["instruction"]:
            if is_openai_format(input["instruction"]):
                raise DistilabelUserError(
                    "Providing `instruction` formatted as an OpenAI chat / conversation is"
                    " deprecated, you should use `ChatGeneration` with `messages` as input instead.",
                    page="components-gallery/tasks/textgeneration/",
                )

            if not isinstance(input["instruction"], str):
                raise DistilabelUserError(
                    f"Input `instruction` must be a string. Got: {input['instruction']}.",
                    page="components-gallery/tasks/textgeneration/",
                )

        messages = self._prepare_message_content(input)

        row_system_prompt = input.get("system_prompt")
        if row_system_prompt:
            messages.insert(0, {"role": "system", "content": row_system_prompt})

        if self.system_prompt and not row_system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages  # type: ignore

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        return ["generation", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        """The output is formatted as a dictionary with the `generation`. The `model_name`
        will be automatically included within the `process` method of `Task`."""
        return {"generation": output}


class ChatGeneration(Task):
    """Generates text based on a conversation.

    `ChatGeneration` is a pre-defined task that defines the `messages` as the input
    and `generation` as the output. This task is used to generate text based on a conversation.
    The `model_name` is also returned as part of the output in order to enhance it.

    Input columns:
        - messages (`List[Dict[Literal["role", "content"], str]]`): The messages to generate the
            follow up completion from.

    Output columns:
        - generation (`str`): The generated text from the assistant.
        - model_name (`str`): The model name used to generate the text.

    Categories:
        - chat-generation

    Icon:
        `:material-chat:`

    Examples:
        Generate text from a conversation in OpenAI chat format:

        ```python
        from distilabel.steps.tasks import ChatGeneration
        from distilabel.models import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        chat = ChatGeneration(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            )
        )

        chat.load()

        result = next(
            chat.process(
                [
                    {
                        "messages": [
                            {"role": "user", "content": "How much is 2+2?"},
                        ]
                    }
                ]
            )
        )
        # result
        # [
        #     {
        #         'messages': [{'role': 'user', 'content': 'How much is 2+2?'}],
        #         'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
        #         'generation': '4',
        #     }
        # ]
        ```
    """

    @property
    def inputs(self) -> List[str]:
        """The input for the task are the `messages`."""
        return ["messages"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType` assuming that the messages provided
        are already formatted that way i.e. following the OpenAI chat format."""

        if not is_openai_format(input["messages"]):
            raise DistilabelUserError(
                "Input `messages` must be an OpenAI chat-like format conversation. "
                f"Got: {input['messages']}. Please check: 'https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models'.",
                page="components-gallery/tasks/chatgeneration/",
            )

        if input["messages"][-1]["role"] != "user":
            raise DistilabelUserError(
                "The last message must be from the user. Please check: "
                "'https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models'.",
                page="components-gallery/tasks/chatgeneration/",
            )

        return input["messages"]

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        return ["generation", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        """The output is formatted as a dictionary with the `generation`. The `model_name`
        will be automatically included within the `process` method of `Task`."""
        return {"generation": output}
