from __future__ import annotations

import re

from contextweave.models import QueryPlan, QueryRequirement

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "with",
}
_EXCEPTION_TERMS = {
    "exception", "exceptions", "except", "unless", "qualification",
    "qualifications", "condition", "conditions", "provided", "subject",
}


def terms(text: str) -> frozenset[str]:
    return frozenset(
        word for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in _STOP_WORDS and len(word) > 1
    )


def understand_query(query: str) -> QueryPlan:
    normalized = query.strip()
    lowered = normalized.lower()
    asks_exceptions = any(term in lowered for term in _EXCEPTION_TERMS)
    is_multi_part = bool(re.search(r"\b(and|also|compare|versus|vs\.?|while)\b", lowered))
    if any(word in lowered for word in ("compare", "versus", "vs.")):
        intent = "comparative"
    elif lowered.startswith(("what is", "define", "what does")):
        intent = "factual"
    elif is_multi_part or lowered.startswith(("how", "why", "explain")):
        intent = "explanatory"
    else:
        intent = "factual"

    requirements = [
        QueryRequirement("main", normalized, terms(normalized) - _EXCEPTION_TERMS)
    ]
    if asks_exceptions:
        requirements.append(
            QueryRequirement("exceptions", "Exceptions and qualifications", _EXCEPTION_TERMS)
        )
    return QueryPlan(
        original_query=normalized,
        intent=intent,
        requirements=tuple(requirements),
        asks_exceptions=asks_exceptions,
        is_multi_part=is_multi_part,
    )
