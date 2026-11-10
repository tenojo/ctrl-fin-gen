"""The pending local engine must not load dependencies or reserve a submission."""
from ctrl_fin_gen.models import GenerationSettings
from ctrl_fin_gen.providers import get_provider_class
from ctrl_fin_gen.providers.base import ProviderError
from ctrl_fin_gen.providers.vllm_python import VLLMPythonProvider
from support import WorkspaceTest


class VLLMPythonSkeletonTests(WorkspaceTest):
    def test_skeleton_is_lazy_and_explicit_about_pending_work(self):
        workspace = self.root / 'not-created'
        provider = VLLMPythonProvider(GenerationSettings('vllm_python', 'test-model', 100), workspace)
        with self.assertRaisesRegex(ProviderError, 'engine and sampling'):
            provider.validate_settings()
        with self.assertRaisesRegex(ProviderError, 'not implemented'):
            provider.generate(None, lambda event: None)
        provider.close()
        self.assertFalse(workspace.exists())

    def test_skeleton_is_not_registered_for_execution(self):
        with self.assertRaisesRegex(ValueError, 'not implemented'):
            get_provider_class('vllm_python')
