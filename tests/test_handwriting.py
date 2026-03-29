"""Tests for handwriting reading strategies module."""


def test_disambiguation_pairs_count():
    from handwriting_engine.handwriting import get_disambiguation_pairs
    pairs = get_disambiguation_pairs()
    assert len(pairs) >= 25, f"Expected at least 25 confusion pairs, got {len(pairs)}"


def test_reading_strategies_not_empty():
    from handwriting_engine.handwriting import get_reading_strategies
    strategies = get_reading_strategies(domain="biology")
    assert len(strategies) > 500, "Reading strategies should be substantial text"
    assert "disambiguation" in strategies.lower() or "confusion" in strategies.lower()


def test_grading_rules_not_empty():
    from handwriting_engine.handwriting import get_grading_handwriting_rules
    rules = get_grading_handwriting_rules(domain="biology")
    assert len(rules) > 200


def test_biology_domain_rules():
    from handwriting_engine.handwriting import get_reading_strategies
    bio = get_reading_strategies(domain="biology")
    assert "DNA" in bio or "enzyme" in bio or "mitochondri" in bio


def test_character_disambiguation_constant():
    from handwriting_engine.handwriting import CHARACTER_DISAMBIGUATION
    assert isinstance(CHARACTER_DISAMBIGUATION, str)
    assert len(CHARACTER_DISAMBIGUATION) > 100
