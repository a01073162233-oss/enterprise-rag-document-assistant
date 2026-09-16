import pytest

from app.services.retrieval import bm25_scores, lexical_tokens


def record(point_id: str, text: str) -> dict[str, object]:
    return {"id": point_id, "payload": {"text": text}}


def test_lexical_tokens_normalize_english_and_add_chinese_bigrams() -> None:
    tokens = lexical_tokens("Ａnnual LEAVE 员工年假")

    assert "annual" in tokens
    assert "leave" in tokens
    assert "员工" in tokens
    assert "工年" in tokens
    assert "年假" in tokens


def test_bm25_ranks_bilingual_exact_evidence_above_unrelated_documents() -> None:
    records = [
        record("leave", "员工年假 annual leave 根据工龄计算，每年最高十五天。"),
        record("security", "信息安全和密码管理制度 cyber security password policy"),
        record("expense", "差旅费报销需要发票 travel expense receipt"),
    ]

    scores = bm25_scores("年假 annual leave", records)

    assert scores["leave"] > 0
    assert "security" not in scores
    assert "expense" not in scores
    assert max(scores, key=scores.get) == "leave"


def test_bm25_rewards_a_rare_matching_term() -> None:
    records = [
        record("specific", "policy policy policy 量子加密"),
        record("common-1", "policy handbook"),
        record("common-2", "policy process"),
    ]

    scores = bm25_scores("量子加密 policy", records)

    assert scores["specific"] > scores["common-1"]
    assert scores["specific"] > scores["common-2"]


def test_bm25_ignores_unrelated_chinese_pages_with_only_single_char_overlap() -> None:
    records = [
        record(
            "leave",
            "未使用的年假最多可结转五天，结转假期须在次年三月三十一日前使用。",
        ),
        record(
            "security",
            "发现疑似数据泄露后，应在一小时内向信息安全团队报告。",
        ),
    ]

    scores = bm25_scores("年假可以结转几天，最晚什么时候使用？", records)

    assert scores["leave"] > 0
    assert "security" not in scores


@pytest.mark.parametrize(
    ("query", "records"),
    [
        ("", [record("one", "content")]),
        ("question", []),
        ("unmatched", [record("one", "完全无关")]),
    ],
)
def test_bm25_returns_no_scores_without_usable_matches(
    query: str, records: list[dict[str, object]]
) -> None:
    assert bm25_scores(query, records) == {}
