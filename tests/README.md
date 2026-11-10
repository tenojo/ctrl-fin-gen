# Tests

Run from the repository root, with the package installed in your Python environment:

```bash
python -m unittest discover -s tests -v
```

The suite uses temporary directories and simulated service I/O. It does not
require API credentials, a vLLM installation, a GPU, or a running server.

| File | Responsibility |
| --- | --- |
| `test_responses_provider.py` | Shared Responses behavior, exercised through OpenAI and vLLM server subclasses using `subTest` |
| `test_openai_provider.py` | OpenAI request options, client configuration, batch files, and job reconciliation |
| `test_vllmserver_provider.py` | vLLM options, URL validation, credentials, capabilities, CLI and runner integration |
| `test_google_provider.py` | Gemini API key options, real SDK serialization, streaming, local recovery, batch correlation and collection |
| `test_vllm_python_provider.py` | The pending local-engine skeleton stays lazy and unregistered |
| `test_generation.py` | Provider-independent execution, retries, result retention, and recovery |
| `test_preparation.py` | Template bounds, prompt rendering, vocabulary import and sampling |
| `test_cli.py` | General CLI behavior and provider registration |
| `test_nlp.py` | Lazy parser imports, isolated worker protocol, missing/corrupted resources, and opt-in real Finnish-TDT integration |
| `test_linguistic_eval.py` | Independent generation/evaluation imports, CLI argument validation and error handling |
| `test_deterministic_workflow.py` | Three-stage file workflow, frozen constraints, exact feature matching, lengths, failures, replacement and incremental runs |

The provider tests use the actual SDK with `httpx2.MockTransport`: outgoing
requests and incoming responses go through SDK serialization/parsing, but are
handled in memory. Constructor tests separately cover settings that an injected
client bypasses, including server URLs and credential selection.

Google SDK tests require the optional dependencies (`python -m pip install -e ".[google]"`)
and are skipped when these are absent. They exercise the real Gen AI SDK over
`httpx2.MockTransport`, including resumable Files API upload and output download. Coverage includes
out-of-order batch outputs with repeated prompts, refusal/incomplete/error rows,
unknown network outcomes, checkpoint reconciliation and CLI option persistence.
These tests make no cloud calls. The project maintainer confirmed successful live
Gemini streaming generation on 2026-11-10. Live non-streaming, batch, and
interruption recovery checks remain pending.

Shared streaming tests exercise the inherited protocol for both adapters. vLLM
runner integration additionally checks that streaming is enabled at the
application boundary. vLLM background batch jobs remain unsupported.

The integration tests temporarily register vLLM within the test process. They do
not change the production provider registry. Live server compatibility still
requires a separate check against the deployed vLLM version and model.

The Trankit integration test is skipped by default. After
`python3.11 scripts/setup_trankit.py`, run it explicitly with
`CTRL_FIN_GEN_TEST_TRANKIT=1 python -m unittest discover -s tests -p 'test_nlp.py' -v`.
It exercises real offline CPU inference and compares repeated annotations,
including after a worker restart. The remaining parser tests require no NLP
dependencies or model downloads.

To run only the shared Responses tests:

```bash
python -m unittest discover -s tests -p 'test_responses_provider.py' -v
```
