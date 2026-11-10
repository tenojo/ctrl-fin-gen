"""Skeleton for an in-process vLLM engine; implementation choices are pending.

Keep model loading, chat templates, SamplingParams, output conversion and engine
cleanup here. No vLLM/GPU dependency is imported until this adapter is implemented.
It is deliberately absent from the provider registry until it can generate texts.
"""
from .base import Provider, ProviderError


class VLLMPythonProvider(Provider):
    """The extension points for local Python inference, without engine assumptions."""

    def validate_settings(self):
        """Later validate engine and sampling options before loading a model."""
        raise ProviderError('vLLM Python is a skeleton awaiting engine and sampling specifications.')

    def format_request(self, request):
        """Later apply the model's chat template to the shared system/user prompts."""
        raise ProviderError('vLLM Python request formatting is not implemented yet.')

    def generate(self, attempt, on_event):
        """Later execute local inference, preserving native output and shared events."""
        raise ProviderError('vLLM Python generation is not implemented yet.')

    def format_result(self, native_output, attempt):
        """Later convert one native output to a shared GenerationResult."""
        raise ProviderError('vLLM Python result formatting is not implemented yet.')

    # Override generate_many for local inference grouping, separate from cloud
    # background jobs. Override close when the adapter owns an engine to release.
