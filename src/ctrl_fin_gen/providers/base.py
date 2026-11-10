"""Provider contract. This module imports no service SDKs.

Providers own native requests, responses, streams, errors, and recovery evidence.
The shared runner sees only records from models.py. Native artifacts belong in
workspace, a dedicated provider/ directory inside the submission.
"""
from abc import ABC, abstractmethod
from pathlib import Path

from ..models import RecoveryReport


class ProviderError(RuntimeError):
    """A service operation could not finish; preserve checkpoints before raising."""


class Provider(ABC):
    """Implement the four core methods; override optional capabilities as needed."""
    streaming = False
    batch_jobs = False

    def __init__(self, settings, workspace):
        """Store configuration without loading an inference engine or opening a client."""
        self.settings = settings
        self.workspace = Path(workspace)

    @classmethod
    def add_cli_arguments(cls, parser):
        """Optionally declare this provider's CLI flags, without initializing it."""

    @classmethod
    def options_from_args(cls, args):
        """Return provider-owned options; the default provider has no extra flags."""
        return {}

    @abstractmethod
    def validate_settings(self):
        """Reject unsupported native settings before reserving a submission."""

    @abstractmethod
    def format_request(self, request):
        """Translate one shared request into native input; never called by the runner."""

    @abstractmethod
    def generate(self, attempt, on_event):
        """Return one terminal result, emitting shared events synchronously."""

    def generate_many(self, attempts, on_event):
        """Yield results sequentially; local engines can override inference grouping."""
        for attempt in attempts:
            yield self.generate(attempt, on_event)

    @abstractmethod
    def format_result(self, native_output, attempt):
        """Extract a common result, retaining native evidence separately."""

    def recover_request(self, attempt, checkpoint):
        """Default to unknown: lack of recovery support is not permission to resend."""
        return RecoveryReport('unknown')

    def submit_batch(self, attempts, submission_id, on_event):
        """Optional background-job submission, distinct from local inference grouping."""
        raise ProviderError('This provider does not support background batch jobs.')

    def get_batch_status(self, job):
        """Optional job lookup; return a normalized BatchJob."""
        raise ProviderError('This provider does not support background batch jobs.')

    def collect_batch(self, job, attempts, on_event):
        """Optional collection of normalized results, matched to original attempts."""
        raise ProviderError('This provider does not support background batch jobs.')

    def recover_batch(self, submission_id, checkpoint):
        """Default reconciliation cannot establish whether submission succeeded."""
        return RecoveryReport('unknown')

    def close(self):
        """Release resources owned by this instance; default has nothing to release."""
