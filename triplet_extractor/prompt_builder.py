"""
Prompt builder for relation extraction.

This module is responsible only for creating prompts
for the LLM.

Input:
    - text
    - two entities

Output:
    - formatted prompt string

The model itself is not called here.
"""


from pathlib import Path


class PromptBuilder:

    def __init__(self, template_path: str):

        self.template_path = template_path

        self.template = self._load_template()

    def _load_template(self) -> str:

        with open(
            self.template_path,
            "r",
            encoding="utf-8"
        ) as file:

            template = file.read()


        return template



    def build(
        self,
        text: str,
        subject: str,
        object_: str
    ) -> str:

        prompt = self.template.replace(
            "{text}",
            text
        )


        prompt = prompt.replace(
            "{subject}",
            subject
        )


        prompt = prompt.replace(
            "{object}",
            object_
        )


        return prompt