import json

import pytest

from isthmus.demo import build_demo
from isthmus.generation import OpenAIAnswerProvider, generate_answer
from conftest import FakeEmbeddingProvider


class FakeAnswerProvider:
    def __init__(self, response, assessment='{"supported": true}'):
        self.response = response
        self.assessment = assessment
        self.prompt = None
        self.assessment_prompt = None

    def generate(self, prompt):
        self.prompt = json.loads(prompt)
        return self.response

    def assess(self, prompt):
        self.assessment_prompt = json.loads(prompt)
        return self.assessment


def test_generate_answer_resolves_citations_to_retrieved_sources():
    pipeline = build_demo(FakeEmbeddingProvider())
    result = pipeline.query("What are the travel reimbursement rules?")
    provider = FakeAnswerProvider('{"answer": "Travel is reimbursed [S1]."}')

    answer = generate_answer(result, pipeline.index.documents, provider)

    assert answer.citations[0].chunk_id == result.context[0].id
    assert answer.citations[0].location == result.context[0].source_location
    assert provider.prompt["question"] == result.query_plan.original_query
    assert provider.prompt["evidence"][0]["content"] == result.context[0].content
    assert provider.assessment_prompt["answer"] == answer.text
    assert provider.assessment_prompt["evidence"][0]["content"] == result.context[0].content


@pytest.mark.parametrize("assessment", [
    '{"supported": false}', '{"supported": "true"}', '{}', 'not json',
])
def test_generate_answer_rejects_failed_or_invalid_assessment(assessment):
    pipeline = build_demo(FakeEmbeddingProvider())
    result = pipeline.query("travel reimbursement")
    provider = FakeAnswerProvider('{"answer": "Unsupported claim [S1]."}', assessment)
    with pytest.raises(ValueError, match="assessment"):
        generate_answer(result, pipeline.index.documents, provider)


@pytest.mark.parametrize("response", [
    '{"answer": "Invented source [S999]."}',
    '{"answer": "Uncited factual claim."}',
    '{"answer": 123}',
    'not json',
])
def test_generate_answer_rejects_invalid_or_unverifiable_citations(response):
    pipeline = build_demo(FakeEmbeddingProvider())
    result = pipeline.query("travel reimbursement")
    with pytest.raises(ValueError):
        generate_answer(result, pipeline.index.documents, FakeAnswerProvider(response))


def test_generate_answer_with_no_context_does_not_call_model():
    pipeline = build_demo(FakeEmbeddingProvider())
    result = pipeline.query("travel")
    from dataclasses import replace

    provider = FakeAnswerProvider("invalid")
    answer = generate_answer(replace(result, context=()), pipeline.index.documents, provider)
    assert "Insufficient evidence" in answer.text
    assert provider.prompt is None


def test_openai_answer_provider_uses_selected_model(monkeypatch):
    from types import SimpleNamespace

    calls = []

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"ok [S1]"}'))])

    monkeypatch.setattr("isthmus.generation.OpenAI", Client)
    monkeypatch.setenv("ISTHMUS_ANSWER_API_KEY", "test")
    provider = OpenAIAnswerProvider("my-model")
    assert provider.generate("prompt") == '{"answer":"ok [S1]"}'
    provider.assess("assessment")
    assert calls[0]["model"] == "my-model"
    assert "supported" in calls[1]["messages"][0]["content"]
    assert calls[1]["messages"][1]["content"] == "assessment"


def test_anthropic_answer_provider_uses_selected_model(monkeypatch):
    from types import SimpleNamespace
    from isthmus.generation import AnthropicAnswerProvider

    calls = []

    class Client:
        def __init__(self, **kwargs):
            self.messages = self

        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"answer":"ok [S1]"}')])

    monkeypatch.setattr("isthmus.generation.Anthropic", Client)
    monkeypatch.setenv("ISTHMUS_ANSWER_API_KEY", "test")
    provider = AnthropicAnswerProvider("chosen-model")
    assert provider.generate("prompt") == '{"answer":"ok [S1]"}'
    provider.assess("assessment")
    assert calls[0]["model"] == "chosen-model"
    assert "supported" in calls[1]["system"]
    assert calls[1]["messages"][0]["content"] == "assessment"
