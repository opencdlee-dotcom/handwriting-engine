"""Tests for LLM output post-processing in vision.py."""

from handwriting_engine.vision import _postprocess_output


class TestPreambleStripping:
    def test_strips_here_is(self):
        text = "Here is the transcribed text:\nHello world"
        assert _postprocess_output(text) == "Hello world"

    def test_strips_transcription_colon(self):
        text = "Transcription:\nLine one\nLine two"
        assert _postprocess_output(text) == "Line one\nLine two"

    def test_strips_i_can_see(self):
        text = "I can see the following handwritten text:\nActual content"
        assert _postprocess_output(text) == "Actual content"

    def test_strips_multiple_preamble_lines(self):
        text = "The image shows handwritten notes.\nHere is the text:\nActual content"
        assert _postprocess_output(text) == "Actual content"

    def test_preserves_content_starting_with_here(self):
        # "Here" as part of actual handwritten content shouldn't be stripped
        # if it's the only line
        text = "Here we go again"
        # This starts with "Here" but after stripping would be empty —
        # the function strips preamble lines that end with \n before content
        result = _postprocess_output(text)
        assert result  # Should not be empty


class TestTrailingStripping:
    def test_strips_note(self):
        text = "Hello world\nNote: some words were difficult to read"
        assert _postprocess_output(text) == "Hello world"

    def test_strips_please_note(self):
        text = "Content here\nPlease note that some characters are ambiguous"
        assert _postprocess_output(text) == "Content here"

    def test_strips_i_hope(self):
        text = "The answer is 42\nI hope this helps!"
        assert _postprocess_output(text) == "The answer is 42"

    def test_strips_let_me_know(self):
        text = "Mitochondria\nLet me know if you need anything else"
        assert _postprocess_output(text) == "Mitochondria"

    def test_preserves_note_in_content(self):
        # "Note" in middle of content should be preserved
        text = "Note the following results:\n42\n55"
        result = _postprocess_output(text)
        assert "42" in result
        assert "55" in result


class TestMarkdownRemoval:
    def test_strips_bold(self):
        text = "The **answer** is **42**"
        assert _postprocess_output(text) == "The answer is 42"

    def test_strips_headers(self):
        text = "# Title\nContent"
        assert _postprocess_output(text) == "Title\nContent"

    def test_strips_italic_underscores(self):
        text = "The _answer_ is here"
        assert _postprocess_output(text) == "The answer is here"


class TestDeduplication:
    def test_deduplicates_repeated_lines(self):
        text = "Hello\nHello\nHello\nHello\nWorld"
        result = _postprocess_output(text)
        assert result.count("Hello") == 2  # Keeps first 2
        assert "World" in result

    def test_preserves_unique_lines(self):
        text = "Line one\nLine two\nLine three"
        assert _postprocess_output(text) == text

    def test_preserves_blank_lines(self):
        text = "Line one\n\nLine two"
        assert _postprocess_output(text) == text


class TestEdgeCases:
    def test_empty_string(self):
        assert _postprocess_output("") == ""

    def test_whitespace_only(self):
        assert _postprocess_output("   \n  ") == ""

    def test_none_safe(self):
        # Should not crash on falsy input
        assert _postprocess_output("") == ""

    def test_clean_text_passthrough(self):
        text = "The mitochondria is the powerhouse of the cell.\npH = 7.2\nOD600 = 0.45"
        assert _postprocess_output(text) == text

    def test_preserves_uncertainty_markers(self):
        text = "The [?] answer is [illegible: ~3 chars]"
        assert _postprocess_output(text) == text


class TestDomainSpellCorrection:
    def test_corrects_mitochondria_typo(self):
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("the mitocondria is important", "biology")
        assert "mitochondria" in result

    def test_no_correction_when_ambiguous(self):
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("the cat sat", "biology")
        assert result == "the cat sat"

    def test_preserves_numbers(self):
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("pH is 7.4 and temp is 37C", "biology")
        assert "7.4" in result
        assert "37C" in result

    def test_preserves_abbreviations(self):
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("ATP ADP NAD", "biology")
        assert "ATP" in result
        assert "ADP" in result

    def test_preserves_question_markers(self):
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("the [?] is here", "biology")
        assert "[?]" in result

    def test_exact_match_no_change(self):
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("mitosis is a type of cell division", "biology")
        assert "mitosis" in result

    def test_edit_distance_1_exact_match_returns_empty(self):
        from handwriting_engine.postprocess import _edit_distance_1_candidates, _BIOLOGY_TERMS
        candidates = _edit_distance_1_candidates("mitosis", _BIOLOGY_TERMS)
        assert candidates == []

    def test_unknown_domain_falls_back_to_general(self):
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("the experiment was successful", "unknown_domain")
        assert isinstance(result, str)

    def test_empty_string(self):
        from handwriting_engine.postprocess import correct_domain_terms
        assert correct_domain_terms("", "biology") == ""
