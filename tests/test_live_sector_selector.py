import os

import pytest
from dotenv import load_dotenv

from isthmus.query import sector_selector_from_environment


@pytest.mark.skipif(
    os.getenv("ISTHMUS_RUN_LIVE_LLM_TESTS") != "1",
    reason="set ISTHMUS_RUN_LIVE_LLM_TESTS=1 to run the real GPT-4o-mini integration test",
)
def test_live_gpt_4o_mini_sector_selection():
    load_dotenv()
    selector = sector_selector_from_environment(enabled=True)
    sector_scores = {
        "finance": 0.83,
        "legal": 0.61,
        "human-resources": 0.34,
    }

    selected = selector.select(
        "Which documents describe the process for approving travel expenses?",
        sector_scores,
        ("finance",),
    )

    print("gpt-4o-mini selected sectors:", ", ".join(selected))
    assert 1 <= len(selected) <= 3
    assert set(selected) <= set(sector_scores)
