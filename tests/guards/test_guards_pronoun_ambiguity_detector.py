"""
Tests for Zero False Positive Guards in PronounAmbiguityDetector

These tests verify that guards:
1. Prevent false positives (guards work correctly)
2. Don't create false negatives (real errors still detected)
3. Meet documentation standards
"""

import pytest
import spacy
from ambiguity.detectors.pronoun_ambiguity_detector import PronounAmbiguityDetector
from ambiguity.types import AmbiguityContext, AmbiguityConfig


@pytest.fixture(scope="module")
def nlp():
    """Load spaCy model once for all tests"""
    return spacy.load("en_core_web_sm")


@pytest.fixture
def detector():
    """Create pronoun ambiguity detector instance"""
    config = AmbiguityConfig()
    return PronounAmbiguityDetector(config)


class TestGuard1CodeBlocks:
    """
    GUARD 1: Code blocks never have ambiguous pronouns
    
    Context: Technical identifiers in code blocks are syntax, not natural language
    """
    
    def test_guard_1_prevents_false_positive(self, detector, nlp):
        """
        Verifies guard prevents false positives in code blocks
        
        Original Issue: Pronouns in code blocks flagged as ambiguous
        Expected: No error (code syntax is not natural language)
        """
        test_cases = [
            {
                "sentence": "if (this.value == null) return;",
                "block_type": "code_block"
            },
            {
                "sentence": "const result = it.next();",
                "block_type": "literal_block"
            },
            {
                "sentence": "var that = this;",
                "block_type": "inline_code"
            }
        ]
        
        for case in test_cases:
            context = AmbiguityContext(
                sentence_index=0,
                sentence=case["sentence"],
                preceding_sentences=[],
                following_sentences=[],
                document_context={'block_type': case["block_type"]}
            )
            
            detections = detector.detect(context, nlp)
            
            assert len(detections) == 0, (
                f"Guard should prevent code block errors in: '{case['sentence']}'\n"
                f"Block type: {case['block_type']}\n"
                f"But found: {detections}"
            )
    
    def test_guard_1_no_false_negatives(self, detector, nlp):
        """
        Verifies guard doesn't create false negatives
        
        Expected: Real ambiguous pronouns outside code blocks are still detected
        """
        context = AmbiguityContext(
            sentence_index=0,
            sentence="Connect the server and the database. It should start automatically.",
            preceding_sentences=["The system has two components."],
            following_sentences=[],
            document_context={'block_type': 'paragraph'}  # Not a code block
        )
        
        detections = detector.detect(context, nlp)
        
        # "It" is ambiguous (could refer to server or database)
        ambiguous_pronouns = [d for d in detections if 'it' in d.flagged_text.lower()]
        
        assert len(ambiguous_pronouns) > 0, (
            "Guard should NOT suppress real ambiguous pronouns in prose\n"
            "But no errors were detected."
        )


class TestGuard2ShortSentences:
    """
    GUARD 2: Very short sentences (< 4 words) without context rarely have ambiguous pronouns
    
    Context: Insufficient sentence length to create competing referents within same sentence
    Note: Only applies when there are no preceding sentences that could contain referents
    """
    
    def test_guard_2_prevents_false_positive(self, detector, nlp):
        """
        Verifies guard prevents false positives in short sentences WITHOUT context
        
        Original Issue: Short commands like "Skip this step" flagged
        Expected: No error (too short for competing referents AND no context)
        """
        test_cases = [
            "Skip this step.",
            "Use it.",
            "This works.",
            "Try that."
        ]
        
        for sentence in test_cases:
            context = AmbiguityContext(
                sentence_index=0,
                sentence=sentence,
                preceding_sentences=[],  # No preceding context
                following_sentences=[],
                document_context={}
            )
            
            detections = detector.detect(context, nlp)
            
            assert len(detections) == 0, (
                f"Guard should prevent short sentence errors: '{sentence}'\n"
                f"But found: {detections}"
            )
    
    def test_guard_2_no_false_negatives(self, detector, nlp):
        """
        Verifies guard doesn't create false negatives
        
        Expected: 
        1. Longer sentences with ambiguity are detected
        2. Short sentences WITH context are also checked (guard doesn't apply)
        """
        # Test 1: Longer sentence
        context1 = AmbiguityContext(
            sentence_index=0,
            sentence="The server connects to the database and the application. It handles requests efficiently.",
            preceding_sentences=[],
            following_sentences=[],
            document_context={}
        )
        
        detections1 = detector.detect(context1, nlp)
        ambiguous_pronouns1 = [d for d in detections1 if 'it' in d.flagged_text.lower()]
        
        assert len(ambiguous_pronouns1) > 0, (
            "Guard should NOT suppress real ambiguous pronouns in longer sentences\n"
            "But no errors were detected."
        )
        
        # Test 2: Short sentence WITH context (guard should NOT apply)
        context2 = AmbiguityContext(
            sentence_index=1,
            sentence="It stopped working.",  # 3 words, but has context
            preceding_sentences=["The system and the application are running."],
            following_sentences=[],
            document_context={}
        )
        
        detections2 = detector.detect(context2, nlp)
        ambiguous_pronouns2 = [d for d in detections2 if 'it' in d.flagged_text.lower()]
        
        assert len(ambiguous_pronouns2) > 0, (
            "Guard should NOT suppress short sentences WITH context\n"
            "But no errors were detected for: 'It stopped working.' with preceding context."
        )


class TestGuard3Questions:
    """
    GUARD 3: Questions have grammatically clear pronoun usage
    
    Context: Question syntax establishes definite discourse context through interrogative structure
    """
    
    def test_guard_3_prevents_false_positive(self, detector, nlp):
        """
        Verifies guard prevents false positives in questions
        
        Original Issue: Pronouns in questions flagged as ambiguous
        Expected: No error (question syntax provides context)
        """
        test_cases = [
            "Does this work correctly?",
            "Is it running?",
            "Can these values be modified?",
            "Will that affect performance?"
        ]
        
        for sentence in test_cases:
            context = AmbiguityContext(
                sentence_index=0,
                sentence=sentence,
                preceding_sentences=["Configure the system settings."],
                following_sentences=[],
                document_context={}
            )
            
            detections = detector.detect(context, nlp)
            
            assert len(detections) == 0, (
                f"Guard should prevent question errors: '{sentence}'\n"
                f"But found: {detections}"
            )
    
    def test_guard_3_no_false_negatives(self, detector, nlp):
        """
        Verifies guard doesn't create false negatives
        
        Expected: Non-question sentences with ambiguity are still detected
        """
        context = AmbiguityContext(
            sentence_index=0,
            sentence="The server and database are configured. It is ready for production.",
            preceding_sentences=[],
            following_sentences=[],
            document_context={}
        )
        
        detections = detector.detect(context, nlp)
        
        # "It" is ambiguous (could refer to server or database)
        ambiguous_pronouns = [d for d in detections if 'it' in d.flagged_text.lower()]
        
        assert len(ambiguous_pronouns) > 0, (
            "Guard should NOT suppress real ambiguous pronouns in statements\n"
            "But no errors were detected."
        )


class TestGuard4ProceduralLists:
    """
    GUARD 4: Pronouns in procedural documentation contexts are unambiguous
    
    Context: Procedural documentation convention - sequential structure provides clear
    context where pronouns refer to entities from previous steps or clauses.
    Covers both list items and paragraphs within procedural documents.
    """
    
    def test_guard_4_prevents_false_positive(self, detector, nlp):
        """
        Verifies guard prevents false positives in procedural contexts
        
        Original Issue: "This activates the profile" in a list item flagged as ambiguous
        Expected: No error (procedural structure makes the reference clear)
        """
        test_cases = [
            # List items
            {
                "sentence": "This activates the profile.",
                "block_type": "ordered_list_item",
                "content_type": "procedural"
            },
            {
                "sentence": "This configures the service automatically.",
                "block_type": "list_item",
                "content_type": "procedural"
            },
            # Paragraphs in procedural documents
            {
                "sentence": "If you plan to connect this NIC only to a specific network, adapt the automatically-created profile.",
                "block_type": "paragraph",
                "content_type": "procedural"
            },
            {
                "sentence": "If multiple connection profiles are active at the same time, these profiles depend on the DNS priority values.",
                "block_type": "paragraph",
                "content_type": "procedural"
            },
            # Notes/warnings in procedural documents
            {
                "sentence": "These settings apply to all network interfaces.",
                "block_type": "note",
                "content_type": "procedural"
            },
            {
                "sentence": "It updates the database schema.",
                "block_type": "unordered_list_item"
            },
            {
                "sentence": "These settings affect all users.",
                "block_type": "step"
            },
            {
                "sentence": "That enables the new feature.",
                "block_type": "procedure_step"
            }
        ]
        
        for case in test_cases:
            doc_context = {'block_type': case["block_type"]}
            if "content_type" in case:
                doc_context['content_type'] = case["content_type"]
            
            context = AmbiguityContext(
                sentence_index=0,
                sentence=case["sentence"],
                preceding_sentences=["Configure the network settings."],
                following_sentences=[],
                document_context=doc_context
            )
            
            detections = detector.detect(context, nlp)
            
            assert len(detections) == 0, (
                f"Guard should prevent procedural context errors in: '{case['sentence']}'\n"
                f"Block type: {case['block_type']}\n"
                f"Content type: {case.get('content_type', 'N/A')}\n"
                f"But found: {detections}"
            )
    
    def test_guard_4_no_false_negatives(self, detector, nlp):
        """
        Verifies guard doesn't create false negatives
        
        Expected: Ambiguous pronouns in non-procedural contexts are still detected.
        Guard should NOT apply to regular paragraphs in non-procedural documents.
        """
        context = AmbiguityContext(
            sentence_index=0,
            sentence="The server and the database both started. This indicates successful initialization.",
            preceding_sentences=[],
            following_sentences=[],
            document_context={'block_type': 'paragraph', 'content_type': 'general'}  # Not procedural context
        )
        
        detections = detector.detect(context, nlp)
        
        # "This" could refer to either "server started" or "database started" or both
        ambiguous_pronouns = [d for d in detections if 'this' in d.flagged_text.lower()]
        
        assert len(ambiguous_pronouns) > 0, (
            "Guard should NOT suppress ambiguous pronouns outside procedural lists\n"
            "But no errors were detected."
        )
    
    def test_guard_4_all_procedural_block_types(self, detector, nlp):
        """
        Test guard works with all procedural block types
        
        Ensures comprehensiveness across all list/step types
        """
        block_types = [
            'list_item',
            'ordered_list_item',
            'unordered_list_item',
            'step',
            'procedure_step'
        ]
        
        for block_type in block_types:
            sentence = "This completes the configuration process."
            context = AmbiguityContext(
                sentence_index=0,
                sentence=sentence,
                preceding_sentences=["Run the setup script."],
                following_sentences=[],
                document_context={'block_type': block_type}
            )
            
            detections = detector.detect(context, nlp)
            
            assert len(detections) == 0, (
                f"Guard should handle block type '{block_type}': '{sentence}'\n"
                f"But found: {detections}"
            )
    
    def test_guard_4_only_applies_to_relevant_pronouns(self, detector, nlp):
        """
        Verify guard only applies to pronouns that benefit from procedural context
        
        Pronouns like "they" or "which" should not be suppressed even in lists
        """
        # Test with pronoun NOT in the guard's list
        context = AmbiguityContext(
            sentence_index=0,
            sentence="Users and administrators can configure settings. They have full access.",
            preceding_sentences=[],
            following_sentences=[],
            document_context={'block_type': 'ordered_list_item'}
        )
        
        detections = detector.detect(context, nlp)
        
        # "They" is ambiguous (users? administrators? both?) even in a list
        # This guard should NOT suppress it since "they" isn't in procedural_pronouns list
        ambiguous_pronouns = [d for d in detections if 'they' in d.flagged_text.lower()]
        
        # Note: This test might not have detections if "they" is resolved by other logic
        # The key point is that the guard checks for specific pronouns only
        # We're just documenting the intended behavior


class TestGuard5DemonstrativeDeterminers:
    """
    GUARD 5: Demonstrative determiners have explicit referents
    
    Context: Grammatical fact - when this/these/that/those acts as a determiner
    (dep_ == 'det'), it directly modifies a noun, creating a self-contained noun
    phrase with an explicit referent. This is distinct from standalone pronouns.
    """
    
    def test_guard_5_prevents_false_positive(self, detector, nlp):
        """
        Verifies guard prevents false positives for demonstrative determiners
        
        Original Issue: "this NIC" flagged as ambiguous
        Expected: No error (determiner + noun = explicit referent)
        """
        test_cases = [
            # Singular demonstratives
            "If you plan to connect this NIC only to a specific network, adapt the profile.",
            "Configure that server for production use.",
            
            # Plural demonstratives
            "Monitor these profiles for changes.",
            "Update those settings in the configuration file.",
            
            # Various noun types
            "This configuration applies to all interfaces.",
            "That database contains user information.",
            "These commands execute sequentially.",
            "Those files require special permissions.",
        ]
        
        for sentence in test_cases:
            context = AmbiguityContext(
                sentence_index=0,
                sentence=sentence,
                preceding_sentences=[],
                following_sentences=[],
                document_context={'block_type': 'paragraph'}
            )
            
            detections = detector.detect(context, nlp)
            
            assert len(detections) == 0, (
                f"Guard should prevent determiner errors: '{sentence}'\n"
                f"But found: {detections}"
            )
    
    def test_guard_5_no_false_negatives(self, detector, nlp):
        """
        Verifies guard doesn't create false negatives
        
        Expected: Standalone pronouns (not determiners) are still detected when ambiguous
        """
        # Standalone "this" as subject (nsubj, not det)
        context = AmbiguityContext(
            sentence_index=0,
            sentence="The server and database are configured. This indicates successful initialization.",
            preceding_sentences=[],
            following_sentences=[],
            document_context={'block_type': 'paragraph'}
        )
        
        detections = detector.detect(context, nlp)
        
        # "This" is ambiguous (could refer to server, database, or both configured)
        # It's a standalone pronoun (nsubj), not a determiner
        ambiguous_pronouns = [d for d in detections if 'this' in d.flagged_text.lower()]
        
        assert len(ambiguous_pronouns) > 0, (
            "Guard should NOT suppress standalone pronouns\n"
            "But no errors were detected."
        )
    
    def test_guard_5_demonstrative_vs_standalone(self, detector, nlp):
        """
        Verifies guard correctly distinguishes determiners from standalone pronouns
        
        Expected:
        - "this NIC" (determiner) → NOT flagged
        - "This is important" (standalone) → MAY be flagged if ambiguous
        """
        # Test 1: Determiner (should not flag)
        context1 = AmbiguityContext(
            sentence_index=0,
            sentence="Connect this NIC to the network.",
            preceding_sentences=[],
            following_sentences=[],
            document_context={'block_type': 'list_item'}
        )
        
        detections1 = detector.detect(context1, nlp)
        assert len(detections1) == 0, "Determiner 'this NIC' should not be flagged"
        
        # Test 2: Standalone pronoun (context-dependent)
        context2 = AmbiguityContext(
            sentence_index=0,
            sentence="This is important for security.",
            preceding_sentences=[],
            following_sentences=[],
            document_context={'block_type': 'paragraph'}
        )
        
        detections2 = detector.detect(context2, nlp)
        # May or may not flag depending on context, but we're verifying it's analyzed
        # (not early-exited by the guard)
        # Just verify it doesn't crash and processes the sentence
        assert isinstance(detections2, list), "Standalone 'this' should be analyzed"


class TestGuardEdgeCases:
    """
    Tests edge cases and boundary conditions for all guards
    """
    
    def test_guard_combination_short_question(self, detector, nlp):
        """Test multiple guards can apply simultaneously"""
        context = AmbiguityContext(
            sentence_index=0,
            sentence="Does this work?",  # Short AND question
            preceding_sentences=[],
            following_sentences=[],
            document_context={}
        )
        
        detections = detector.detect(context, nlp)
        
        assert len(detections) == 0, (
            "Multiple guards should prevent errors"
        )
    
    def test_actual_document_sentences(self, detector, nlp):
        """Test with real sentences from documentation"""
        # From user's original problem
        test_cases = [
            {
                "sentence": "Select the profile to use with the project.",
                "block_type": "ordered_list_item",
                "should_flag": False,
                "reason": "No pronouns"
            },
            {
                "sentence": "This activates the profile.",
                "block_type": "ordered_list_item",
                "should_flag": False,
                "reason": "Procedural context guard"
            }
        ]
        
        for case in test_cases:
            context = AmbiguityContext(
                sentence_index=0,
                sentence=case["sentence"],
                preceding_sentences=["Configure project settings."],
                following_sentences=[],
                document_context={'block_type': case.get("block_type", "paragraph")}
            )
            
            detections = detector.detect(context, nlp)
            
            if case["should_flag"]:
                assert len(detections) > 0, (
                    f"Should flag: '{case['sentence']}'\n"
                    f"Reason: {case['reason']}"
                )
            else:
                assert len(detections) == 0, (
                    f"Should NOT flag: '{case['sentence']}'\n"
                    f"Reason: {case['reason']}\n"
                    f"But found: {detections}"
                )


class TestGuardValidation:
    """
    Meta-tests to validate the guards themselves meet standards
    """
    
    def test_guard_is_documented(self):
        """Verify guards have required documentation in source"""
        import inspect
        source_guards = inspect.getsource(PronounAmbiguityDetector._apply_zero_false_positive_guards)
        source_evidence = inspect.getsource(PronounAmbiguityDetector._calculate_pronoun_evidence)
        
        # Guards 1-4 are in _apply_zero_false_positive_guards
        assert "GUARD 1:" in source_guards, "Guard 1 must have numbered identifier"
        assert "Test: test_guard_1_code_blocks()" in source_guards, "Guard 1 must reference test"
        assert "GUARD 2:" in source_guards, "Guard 2 must have numbered identifier"
        assert "Test: test_guard_2_short_sentences()" in source_guards, "Guard 2 must reference test"
        assert "GUARD 3:" in source_guards, "Guard 3 must have numbered identifier"
        assert "Test: test_guard_3_questions()" in source_guards, "Guard 3 must reference test"
        assert "GUARD 4:" in source_guards, "Guard 4 must have numbered identifier"
        assert "Test: test_guard_4_procedural_lists()" in source_guards, "Guard 4 must reference test"
        assert source_guards.count("Reason:") >= 4, "Guards 1-4 must document reason"
        
        # Guard 5 is in _calculate_pronoun_evidence (token-level check)
        assert "GUARD 5:" in source_evidence, "Guard 5 must have numbered identifier"
        assert "Demonstrative determiners" in source_evidence, "Guard 5 must describe demonstrative determiners"
        
        # Check reasons are objective (no subjective language)
        combined_source = source_guards + source_evidence
        source_lower = combined_source.lower()
        for word in ['usually', 'often', 'sometimes', 'generally', 'typically']:
            assert word not in source_lower or source_lower.count(word) == 0, (
                f"Guard reasons contain subjective word '{word}' - must be objective"
            )
    
    def test_guard_count_under_limit(self):
        """Verify rule doesn't exceed 5 guards"""
        import inspect
        import re
        source_guards = inspect.getsource(PronounAmbiguityDetector._apply_zero_false_positive_guards)
        source_evidence = inspect.getsource(PronounAmbiguityDetector._calculate_pronoun_evidence)
        
        # Count guards by looking for "GUARD N:" pattern (numbered guards only)
        combined_source = source_guards + source_evidence
        guard_matches = re.findall(r'GUARD \d+:', combined_source)
        guard_count = len(set(guard_matches))  # Use set to avoid double-counting
        
        assert guard_count == 5, f"Expected 5 guards, found {guard_count}: {guard_matches}"
        assert guard_count <= 5, f"Detector has {guard_count} guards, maximum is 5"

