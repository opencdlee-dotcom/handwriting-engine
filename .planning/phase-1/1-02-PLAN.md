<plan phase="1" index="02" requirement="REQ-002">
  <objective>Extend _smart_route() to count [?] markers and auto-escalate to self_correct when uncertainty exceeds threshold, and add comprehensive tests for both new features</objective>

  <files>
    <modify>handwriting_engine/consensus.py</modify>
    <modify>tests/test_consensus.py</modify>
  </files>

  <tasks>
    <task type="auto">
      <name>Extend _smart_route() with uncertainty-gated self-correction</name>
      <action>
        In handwriting_engine/consensus.py, update the _smart_route() function.

        Current signature:
        def _smart_route(image_b64, media_type, prompt, system_prompt, quality_assessment, content_type, max_tokens)

        Update the read_with_consensus() call to pass uncertainty_threshold through, and update _smart_route() to accept it:

        def _smart_route(
            image_b64: str, media_type: str, prompt: str, system_prompt: str,
            quality_assessment: dict, content_type: str, max_tokens: int,
            uncertainty_threshold: int = 3,
        ) -> ConsensusResult:

        After the difficulty == "easy" branch returns _best_of(), add uncertainty checking:

        In the "easy" path:
          result = _best_of(image_b64, media_type, prompt, system_prompt, content_type, max_tokens)
          marker_count = _count_uncertainty_markers(result.text)
          if marker_count > uncertainty_threshold:
              logger.info(f"Smart route: escalating to self_correct ({marker_count} uncertainty markers > threshold {uncertainty_threshold})")
              corrected = _self_correct(image_b64, media_type, prompt, system_prompt, content_type, max_tokens)
              corrected.strategy_used = f"smart→self_correct"
              return corrected
          return result

        In the "medium" path:
          result = _cascade(image_b64, media_type, prompt, system_prompt, 0.7, max_tokens)
          marker_count = _count_uncertainty_markers(result.text)
          if marker_count > uncertainty_threshold:
              logger.info(f"Smart route medium: escalating to self_correct ({marker_count} markers)")
              corrected = _self_correct(image_b64, media_type, prompt, system_prompt, content_type, max_tokens)
              corrected.strategy_used = "smart→self_correct"
              return corrected
          return result

        Also update the read_with_consensus() → _smart_route() call to pass uncertainty_threshold:
          return _smart_route(image_b64, media_type, prompt, system_prompt, quality_assessment, content_type, max_tokens, uncertainty_threshold)
      </action>
      <verify>python -c "from handwriting_engine.consensus import _smart_route; print('OK')"</verify>
      <done>_smart_route() escalates to self_correct when [?] markers exceed threshold</done>
    </task>

    <task type="auto">
      <name>Add self_correct and smart escalation tests to test_consensus.py</name>
      <action>
        In tests/test_consensus.py, add the following imports at the top (if not present):
          from handwriting_engine.consensus import _count_uncertainty_markers

        Then append the following test classes at the end of the file:

        # --- _count_uncertainty_markers ---

        def test_count_uncertainty_markers_none():
            assert _count_uncertainty_markers("The student wrote mitosis") == 0

        def test_count_uncertainty_markers_question():
            assert _count_uncertainty_markers("The [?] wrote [illegible: 3 chars]") == 2

        def test_count_uncertainty_markers_multiple():
            text = "[?] [?] [unclear] ??? [illegible]"
            count = _count_uncertainty_markers(text)
            assert count >= 3


        # --- self_correct strategy ---

        class TestSelfCorrectStrategy:

            def test_self_correct_calls_provider_twice(self):
                """self_correct must call read_image twice: initial read + correction."""
                call_count = [0]
                responses = [
                    "The [?] wrote about mitosis",  # initial read with uncertainty
                    "The student wrote about mitosis",  # corrected read
                ]

                class CountingProvider:
                    name = "gemini"
                    _usage = {"input_tokens": 100, "output_tokens": 50}

                    def read_image(self, *args, **kwargs):
                        idx = call_count[0]
                        call_count[0] += 1
                        return responses[idx] if idx < len(responses) else responses[-1]

                    @property
                    def usage(self):
                        return dict(self._usage)

                with patch("handwriting_engine.consensus.available_providers", return_value=["gemini"]), \
                     patch("handwriting_engine.consensus.get_provider", return_value=CountingProvider()):
                    result = read_with_consensus("b64", "image/jpeg", "read", strategy="self_correct")

                assert call_count[0] == 2  # Initial + correction
                assert result.strategy_used == "self_correct_1pass"
                assert result.text == "The student wrote about mitosis"

            def test_self_correct_no_extra_pass_when_clean(self):
                """If initial read has no [?] markers, correction still runs once (max_rounds=1)."""
                call_count = [0]

                class CleanProvider:
                    name = "gemini"
                    _usage = {"input_tokens": 100, "output_tokens": 50}

                    def read_image(self, *args, **kwargs):
                        call_count[0] += 1
                        return "The student wrote about mitosis"

                    @property
                    def usage(self):
                        return dict(self._usage)

                with patch("handwriting_engine.consensus.available_providers", return_value=["gemini"]), \
                     patch("handwriting_engine.consensus.get_provider", return_value=CleanProvider()):
                    result = read_with_consensus("b64", "image/jpeg", "read", strategy="self_correct")

                assert call_count[0] == 2  # Always does at least 1 correction pass
                assert result.strategy_used == "self_correct_1pass"

            def test_self_correct_max_rounds_respected(self):
                """max_self_correct_rounds=2 should do up to 2 correction passes."""
                call_count = [0]
                responses = [
                    "The [?] wrote [?] mitosis",   # initial
                    "The [?] wrote about mitosis",  # round 1 — still has [?]
                    "The student wrote about mitosis",  # round 2 — clean
                ]

                class MultiRoundProvider:
                    name = "gemini"
                    _usage = {"input_tokens": 100, "output_tokens": 50}

                    def read_image(self, *args, **kwargs):
                        idx = call_count[0]
                        call_count[0] += 1
                        return responses[idx] if idx < len(responses) else responses[-1]

                    @property
                    def usage(self):
                        return dict(self._usage)

                with patch("handwriting_engine.consensus.available_providers", return_value=["gemini"]), \
                     patch("handwriting_engine.consensus.get_provider", return_value=MultiRoundProvider()):
                    result = read_with_consensus(
                        "b64", "image/jpeg", "read",
                        strategy="self_correct", max_self_correct_rounds=2
                    )

                assert call_count[0] == 3  # initial + 2 correction passes
                assert "2pass" in result.strategy_used
                assert result.text == "The student wrote about mitosis"

            def test_self_correct_strategy_used_field(self):
                """strategy_used should reflect self_correct."""
                mock = MagicMock()
                mock.name = "gemini"
                mock.read_image.return_value = "clean text without markers here"
                mock.usage = {"input_tokens": 50, "output_tokens": 25}

                with patch("handwriting_engine.consensus.available_providers", return_value=["gemini"]), \
                     patch("handwriting_engine.consensus.get_provider", return_value=mock):
                    result = read_with_consensus("b64", "image/jpeg", "read", strategy="self_correct")

                assert "self_correct" in result.strategy_used

            def test_self_correct_provider_results_has_initial_and_corrected(self):
                """provider_results must contain 'initial' and 'corrected' keys."""
                call_count = [0]

                class TwoPassProvider:
                    name = "gemini"
                    _usage = {"input_tokens": 100, "output_tokens": 50}

                    def read_image(self, *args, **kwargs):
                        call_count[0] += 1
                        return "initial text" if call_count[0] == 1 else "corrected text"

                    @property
                    def usage(self):
                        return dict(self._usage)

                with patch("handwriting_engine.consensus.available_providers", return_value=["gemini"]), \
                     patch("handwriting_engine.consensus.get_provider", return_value=TwoPassProvider()):
                    result = read_with_consensus("b64", "image/jpeg", "read", strategy="self_correct")

                assert "initial" in result.provider_results
                assert "corrected" in result.provider_results
                assert result.provider_results["initial"] == "initial text"
                assert result.provider_results["corrected"] == "corrected text"


        # --- smart strategy escalation ---

        class TestSmartEscalation:

            def test_smart_escalates_on_high_uncertainty(self):
                """smart strategy should escalate to self_correct when [?] count > threshold."""
                call_count = [0]
                responses = [
                    "[?] wrote [?] about [?] and [?] more",  # initial easy read — 4 [?] markers
                    "The student wrote about mitosis and cell division",  # self-correction
                ]

                class EscalatingProvider:
                    name = "gemini"
                    _usage = {"input_tokens": 100, "output_tokens": 50}

                    def read_image(self, *args, **kwargs):
                        idx = call_count[0]
                        call_count[0] += 1
                        return responses[idx] if idx < len(responses) else responses[-1]

                    @property
                    def usage(self):
                        return dict(self._usage)

                good_quality = {"quality": "good", "blur_score": 200, "contrast_score": 0.7, "faint_ink": False, "issues": []}

                with patch("handwriting_engine.consensus.available_providers", return_value=["gemini"]), \
                     patch("handwriting_engine.consensus.get_provider", return_value=EscalatingProvider()):
                    result = read_with_consensus(
                        "b64", "image/jpeg", "read",
                        strategy="smart",
                        quality_assessment=good_quality,
                        uncertainty_threshold=3,
                    )

                assert "self_correct" in result.strategy_used

            def test_smart_no_escalation_on_clean_output(self):
                """smart strategy should NOT escalate when output is clean."""
                mock = MagicMock()
                mock.name = "gemini"
                mock.read_image.return_value = "The student answered mitosis correctly"
                mock.usage = {"input_tokens": 100, "output_tokens": 50}

                good_quality = {"quality": "good", "blur_score": 200, "contrast_score": 0.7, "faint_ink": False, "issues": []}

                with patch("handwriting_engine.consensus.available_providers", return_value=["gemini"]), \
                     patch("handwriting_engine.consensus.get_provider", return_value=mock):
                    result = read_with_consensus(
                        "b64", "image/jpeg", "read",
                        strategy="smart",
                        quality_assessment=good_quality,
                        uncertainty_threshold=3,
                    )

                # Should be best_of (single read), not self_correct
                assert "self_correct" not in result.strategy_used
                # Only called once (best_of, no escalation)
                assert mock.read_image.call_count == 1
      </action>
      <verify>cd "/Users/user/Documents/VSCode Projects/handwriting-engine" && python -m pytest tests/test_consensus.py -x -q 2>&1 | tail -5</verify>
      <done>All consensus tests pass including new TestSelfCorrectStrategy and TestSmartEscalation</done>
    </task>
  </tasks>

  <dependencies>Plan 01 (self_correct and _count_uncertainty_markers must exist)</dependencies>
  <commit_message>feat(phase-1-02): smart escalation + self-correction tests (REQ-001, REQ-002)</commit_message>
</plan>
