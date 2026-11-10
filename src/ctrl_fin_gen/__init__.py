"""Controlled Finnish generation: start reading at cli.run_command.

prepare.py resolves template choices and freezes GenerationRequest records.
prompts.py renders Finnish system/user text. generation.py manages shared files,
attempts, events, and progress. storage.py handles persistence and locks.
providers/base.py defines the execution contract; providers/openai.py owns all
OpenAI translation, SDK calls, native artifacts and recovery. Register future
implementations in providers/__init__.py without changing the runner.

Configurations group realized conditions, requests identify planned texts, and
attempts identify retries. All terminal outcomes are retained without linguistic
filtering. Shared schema 1.0 requires fresh submissions; older runs stay intact.
"""
