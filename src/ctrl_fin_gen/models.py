"""The prepared inputs shared by creation, prompting, and generation."""

from dataclasses import dataclass


@dataclass
class ExperimentInputs:
    selection: dict
    template: dict
    controls: dict
    prompts: dict
