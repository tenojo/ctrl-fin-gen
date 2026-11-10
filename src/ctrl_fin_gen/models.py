"""The data container shared by preparation, prompting, and generation.

The fields remain ordinary dictionaries so they match the project's JSON files.
Validation and default resolution live in prepare.py, not in this container.
"""

from dataclasses import dataclass, field


@dataclass
class ExperimentInputs:
    """Bundle an experiment's resolved values and the text needed to render them.

    selection: Input choices/recipes after overrides, before default resolution;
        saved as selection.json. It may contain vocabulary file references.
    template: Realized constraints, audience, theme, and source references;
        saved as template.json. Vocabulary items or the sampling pool are frozen.
    controls: Shared catalogue definitions, grouped by constraint category;
        prompts.py uses their Finnish instructions. This contains more than just
        the selected-control snapshots saved in control_definitions.json.
    prompts: The actual strings under system, user_prefix, user_reminder, layout;
        these are text contents, whereas template['prompts'] holds configuration.
    """

    selection: dict
    template: dict
    controls: dict
    prompts: dict


# Generation records deliberately contain no SDK types or native API payloads.

SCHEMA_VERSION = '1.0'
RESULT_STATUSES = {'completed', 'incomplete', 'refused', 'failed', 'cancelled'}
JOB_TERMINAL = {'completed', 'failed', 'cancelled', 'expired'}


@dataclass
class GenerationSettings:
    """Common execution choices; only a provider interprets provider_options."""
    backend: str
    model: str
    max_output_tokens: int
    mode: str = 'immediate'
    provider_options: dict = field(default_factory=dict)


@dataclass
class GenerationRequest:
    """A frozen planned text, reusable across runs without changing its vocabulary."""
    request_id: str
    configuration_id: str
    repetition_index: int
    system_prompt: str
    user_prompt: str
    conditions: dict
    metadata: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass
class GenerationAttempt:
    """One execution of a request. Retrying changes attempt_id, not request_id."""
    request: GenerationRequest
    submission_id: str
    attempt_id: str


@dataclass
class GenerationResult:
    """One terminal attempt outcome; completed does not imply linguistic quality."""
    submission_id: str
    request_id: str
    attempt_id: str
    configuration_id: str
    repetition_index: int
    status: str
    text: str = ''
    finish_reason: str | None = None
    model: str | None = None
    usage: dict | None = None
    error: dict | None = None
    native_artifacts: list = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


@dataclass
class ProviderEvent:
    """A common started/text_delta/checkpoint event; checkpoint contents are opaque."""
    kind: str
    attempt_id: str | None = None
    text: str = ''
    checkpoint: dict = field(default_factory=dict)


@dataclass
class BatchJob:
    """Normalized job status with a handle interpreted exclusively by its provider."""
    status: str
    handle: dict
    counts: dict | None = None


@dataclass
class RecoveryReport:
    """finished/running/safe_to_retry/unknown, with recovered output when available."""
    outcome: str
    result: GenerationResult | None = None
    job: BatchJob | None = None
