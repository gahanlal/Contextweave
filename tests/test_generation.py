import json

import pytest

from isthmus.demo import build_demo
from isthmus.generation import OpenAIAnswerProvider, generate_answer


class FakeAnswerProvider:
    def __init__(self, response):
        self.response = response
        self.prompt = None

    def generate(self, prompt):
        self.prompt = json.loads(prompt)
        return self.response


def test_generate_answer_resolves_citations_to_retrieved_sources():
    pipeline = build_demo()
    result = pipeline.query("What are the travel reimbursement rules?")
    provider = FakeAnswerProvider('{"answer": "Travel is reimbursed [S1]."}')

    answer = generate_answer(result, pipeline.index.documents, provider)

    assert answer.citations[0].chunk_id == result.context[0].id
    assert answer.citations[0].location == result.context[0].source_location
    assert provider.prompt["question"] == result.query_plan.original_query
    assert provider.prompt["evidence"][0]["content"] == result.context[0].content


@pytest.mark.parametrize("response", [
    '{"answer": "Invented source [S999]."}',
    '{"answer": "Uncited factual claim."}',
    '{"answer": 123}',
    'not json',
])
def test_generate_answer_rejects_invalid_or_unverifiable_citations(response):
    pipeline = build_demo()
    result = pipeline.query("travel reimbursement")
    with pytest.raises(ValueError):
        generate_answer(result, pipeline.index.documents, FakeAnswerProvider(response))


def test_generate_answer_with_no_context_does_not_call_model():
    pipeline = build_demo()
    result = pipeline.query("travel")
    from dataclasses import replace

    provider = FakeAnswerProvider("invalid")
    answer = generate_answer(replace(result, context=()), pipeline.index.documents, provider)
    assert "Insufficient evidence" in answer.text
    assert provider.prompt is None


def test_openai_answer_provider_uses_gpt_4o_mini(monkeypatch):
    from types import SimpleNamespace

    calls = []

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"ok [S1]"}'))])

    monkeypatch.setattr("isthmus.generation.OpenAI", Client)
    assert OpenAIAnswerProvider().generate("prompt") == '{"answer":"ok [S1]"}'
    assert calls[0]["model"] == "gpt-4o-mini"
