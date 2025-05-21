import json
from typing import TYPE_CHECKING, Any

from distilabel.steps.tasks import Task
from system_prompts import (
    IFEVAL_INSTRUCTION_ID_LIST_ASSIGNATOR_SYSTEM_PROMPT,
    IFEVAL_KWARGS_ASSIGNATOR_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from distilabel.typing import ChatType, StepColumns


class IFEvalInstructionIdListAssignator(Task):
    @property
    def inputs(self) -> "StepColumns":
        return ["instruction"]

    def format_input(self, input: dict[str, Any]) -> "ChatType":
        instruction = input["instruction"]

        return [
            {
                "role": "system",
                "content": IFEVAL_INSTRUCTION_ID_LIST_ASSIGNATOR_SYSTEM_PROMPT,
            },
            {"role": "user", "content": instruction},
        ]

    @property
    def outputs(self) -> "StepColumns":
        return ["instruction_id_list"]

    def format_output(
        self, output: str | None, input: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if output is None:
            return {"instruction_id_list": None}

        return json.loads(output)


class IFEvalKwargsAssignator(Task):
    @property
    def inputs(self) -> "StepColumns":
        return ["instruction", "instruction_id_list"]

    def format_input(self, input: dict[str, Any]) -> "ChatType":
        instruction = input["instruction"]
        instruction_id_list = "\n".join(input["instruction_id_list"])

        return [
            {"role": "system", "content": IFEVAL_KWARGS_ASSIGNATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"## Instruction\n\n{instruction}## Instruction ID List\n\n{instruction_id_list}",
            },
        ]

    @property
    def outputs(self) -> "StepColumns":
        return ["kwargs"]

    def format_output(
        self, output: str | None, input: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if output is None:
            return {"kwargs": None}

        return {"kwargs": output}
