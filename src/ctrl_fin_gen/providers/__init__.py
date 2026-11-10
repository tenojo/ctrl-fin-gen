"""Explicit provider registration; adding a backend changes only this package.

Provider modules are lightweight: service SDKs/engines are loaded on execution,
so importing these classes for CLI flags requires no inference environment.
Implement native operations in the corresponding module and register its class
below. Shared generation code does not need a backend-specific branch.
"""
from .base import Provider, ProviderError
from .openai import OpenAIProvider
from .google import GeminiProvider

# Keep the earlier CLI spelling as an alias; both names use Gemini API keys.
PROVIDERS = {'openai': OpenAIProvider, 'gemini': GeminiProvider, 'google_cloud': GeminiProvider}


def get_provider_class(name):
    """Return a registered class without constructing a client or engine."""
    if name not in PROVIDERS:
        raise ValueError(f'Backend {name!r} is not implemented yet.')
    return PROVIDERS[name]


def provider_classes():
    """Return the classes contributing optional CLI flags."""
    return list(dict.fromkeys(PROVIDERS.values()))


def create_provider(settings, workspace):
    """Construct the selected provider for generation, resume, status or collection."""
    return get_provider_class(settings.backend)(settings, workspace)
