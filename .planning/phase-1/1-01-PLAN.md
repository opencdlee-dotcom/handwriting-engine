<plan phase="1" index="01" requirement="REQ-001">
  <objective>Add SELF_CORRECTION_PROMPT to handwriting.py and implement _self_correct() strategy in consensus.py with read_with_consensus() dispatch</objective>

  <files>
    <modify>handwriting_engine/handwriting.py</modify>
    <modify>handwriting_engine/consensus.py</modify>
  </files>

  <tasks>
    <task type="auto">
      <name>Add SELF_CORRECTION_PROMPT to handwriting.py</name>
      <action>
        In handwriting_engine/handwriting.py, after the ANTI_HALLUCINATION_PROTOCOL block (around line 244+), add a new constant:

        SELF_CORRECTION_PROMPT = """
        === SELF-CORRECTION REVIEW ===
        You previously produced the following transcription of this handwritten image:

        ---
        {initial_transcription}
        ---

        Review your transcription against the actual image. Look specifically for:

        1. CHARACTER CONFUSION: Check every character against the top confusion pairs:
           - 1/l/I, 0/O, rn/m, 5/S, u/v, a/o, cl/d, n/h, e/c, B/D, 7/1, 3/8, t/+
        2. [?] MARKERS: For every [?] you marked, make a final determination using context
        3. NUMBERS: Re-read each number digit by digit — do not guess whole numbers
        4. WORD BOUNDARIES: Check every word produces a real word or known scientific term

        Output ONLY the corrected transcription. If a section was correct, reproduce it unchanged.
        Do NOT add commentary, explanations, or "I corrected X" notes.
        Do NOT silently fix spelling — preserve original spelling errors.
        """.strip()

        The constant should use {initial_transcription} as a format placeholder.
      </action>
      <verify>grep -n "SELF_CORRECTION_PROMPT" handwriting_engine/handwriting.py returns a match</verify>
      <done>SELF_CORRECTION_PROMPT constant defined in handwriting.py with {initial_transcription} placeholder</done>
    </task>

    <task type="auto">
      <name>Implement _self_correct() in consensus.py</name>
      <action>
        In handwriting_engine/consensus.py, after the _best_of() function and before _vote(), add the _self_correct() function.

        Also add _count_uncertainty_markers() helper at the top of the confidence scoring section (near _UNCERTAINTY_RE, around line 680):

        def _count_uncertainty_markers(text: str) -> int:
            """Count [?] and similar uncertainty markers in transcription output."""
            return len(_UNCERTAINTY_RE.findall(text))

        The _self_correct() function:

        def _self_correct(
            image_b64: str, media_type: str, prompt: str, system_prompt: str,
            content_type: str, max_tokens: int,
            max_rounds: int = 1,
        ) -> ConsensusResult:
            """Two-pass self-correction: read → review own output → correct.

            Based on Journal of Documentation 2025 (peer-reviewed): GPT-4o
            self-correction drops CER from 1.75% to 1.39% on IAM (~20% reduction).
            Same model reads, then reviews its own transcription against the image.

            Args:
                max_rounds: Number of correction passes (default 1, max 3).
                           Extra passes only trigger if [?] markers remain.
            """
            from handwriting_engine.handwriting import SELF_CORRECTION_PROMPT

            max_rounds = min(max(1, max_rounds), 3)

            provider_name = BEST_PROVIDER_ROUTING.get(content_type, BEST_PROVIDER_ROUTING["default"])
            avail = available_providers()
            if not avail:
                raise ValueError(
                    "No vision providers available. Install at least one: "
                    "pip install handwriting-engine[all]"
                )
            if provider_name not in avail:
                provider_name = avail[0]

            if circuit_breaker.is_open(provider_name):
                # Fall back to best_of if primary provider is open
                return _best_of(image_b64, media_type, prompt, system_prompt, content_type, max_tokens)

            try:
                provider = get_provider(provider_name)
                initial_text = provider.read_image(image_b64, media_type, prompt, system_prompt, max_tokens)
                circuit_breaker.record_success(provider_name)
            except Exception as e:
                logger.warning(f"self_correct: initial read failed for {provider_name}: {e}")
                circuit_breaker.record_failure(provider_name)
                return ConsensusResult(text="", confidence=0.0, strategy_used="self_correct_failed")

            if not initial_text.strip():
                return ConsensusResult(text="", confidence=0.0, strategy_used="self_correct_failed")

            corrected_text = initial_text
            rounds_done = 0

            for round_num in range(max_rounds):
                # Build the correction prompt with the current transcription
                correction_prompt = SELF_CORRECTION_PROMPT.format(
                    initial_transcription=corrected_text
                )

                try:
                    corrected_text = provider.read_image(
                        image_b64, media_type, correction_prompt, system_prompt, max_tokens
                    )
                    rounds_done += 1
                    circuit_breaker.record_success(provider_name)
                except Exception as e:
                    logger.warning(f"self_correct: correction pass {round_num + 1} failed: {e}")
                    break

                # Stop early if no uncertainty markers remain
                if _count_uncertainty_markers(corrected_text) == 0:
                    break

            if not corrected_text.strip():
                corrected_text = initial_text

            confidence = _single_text_confidence(corrected_text)

            return ConsensusResult(
                text=corrected_text,
                confidence=confidence,
                confidence_level=_derive_confidence_level(confidence, corrected_text),
                provider_results={
                    "initial": initial_text,
                    "corrected": corrected_text,
                },
                strategy_used=f"self_correct_{rounds_done}pass",
                tokens_used=provider.usage,
            )

        Note: _count_uncertainty_markers uses _UNCERTAINTY_RE which is defined later in the file.
        Move _UNCERTAINTY_RE and _count_uncertainty_markers before _self_correct() to avoid forward reference.
        Current _UNCERTAINTY_RE is at line ~680. Move it to just after the module-level constants (after CASCADE_ORDER ~line 46).
      </action>
      <verify>python -c "from handwriting_engine.consensus import _self_correct; print('OK')"</verify>
      <done>_self_correct() importable and _count_uncertainty_markers() defined</done>
    </task>

    <task type="auto">
      <name>Wire self_correct into read_with_consensus() dispatch</name>
      <action>
        In handwriting_engine/consensus.py, update read_with_consensus() signature and dispatch to include:

        1. Add `max_self_correct_rounds: int = 1` parameter to read_with_consensus()
        2. Add `uncertainty_threshold: int = 3` parameter to read_with_consensus()
        3. Add elif branch in the dispatch block:

        elif strategy == "self_correct":
            return _self_correct(image_b64, media_type, prompt, system_prompt, content_type, max_tokens, max_self_correct_rounds)

        4. Update the raise ValueError for unknown strategies to include "self_correct" in the list.

        Also update the docstring for read_with_consensus() to include:
        - self_correct: Single provider reads, then reviews and corrects its own output
        - uncertainty_threshold: [?] marker count that triggers auto-escalation in smart strategy
      </action>
      <verify>python -c "from handwriting_engine.consensus import read_with_consensus; import inspect; sig = inspect.signature(read_with_consensus); print('self_correct' in str(sig) or 'max_self_correct' in str(sig))"</verify>
      <done>read_with_consensus() accepts strategy="self_correct" and max_self_correct_rounds parameter</done>
    </task>
  </tasks>

  <dependencies>none</dependencies>
  <commit_message>feat(phase-1-01): add self-correction consensus strategy (REQ-001)</commit_message>
</plan>
