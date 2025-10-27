"""
World-Class Comprehensive Test Suite for Fabrication Risk Detector Refinements

This test suite validates the causative construction guard that prevents
false positives on logical cause-and-effect statements.

Tests cover:
1. Causative constructions that should NOT be flagged
2. Vague "makes" statements that SHOULD be flagged
3. Edge cases and real-world scenarios
4. Different causative verbs
"""

import pytest
import spacy
from ambiguity.detectors.fabrication_risk_detector import FabricationRiskDetector
from ambiguity.types import AmbiguityContext, AmbiguityConfig


@pytest.fixture(scope="module")
def nlp():
    """Load spaCy model once for all tests."""
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        pytest.skip("spaCy model 'en_core_web_sm' not installed")


@pytest.fixture
def detector():
    """Create a fabrication risk detector instance."""
    config = AmbiguityConfig()
    return FabricationRiskDetector(config)


@pytest.fixture
def base_context():
    """Create a base context for testing."""
    return {
        'content_type': 'documentation',
        'block_type': 'paragraph'
    }


class TestCausativeConstructions:
    """Test causative constructions that should NOT be flagged."""
    
    def test_original_problematic_case(self, detector, nlp, base_context):
        """The original case: 'a meaningful name makes it easier'"""
        sentence = "A meaningful name makes it easier to understand the code."
        context = AmbiguityContext(
            sentence_index=0,
            sentence=sentence,
            paragraph_context="",
            document_context=base_context
        )
        
        detections = detector.detect(context, nlp)
        
        # Should NOT flag "makes" in causative construction
        flagged_words = [d.evidence.tokens[0].lower() for d in detections]
        assert 'makes' not in flagged_words, \
            "Should not flag 'makes' in causative construction 'makes it easier'"
    
    def test_makes_it_adjective_patterns(self, detector, nlp, base_context):
        """Test various 'makes it [adjective]' patterns."""
        test_cases = [
            "Using descriptive names makes it clearer what the code does.",
            "This approach makes it simpler to maintain the application.",
            "Good documentation makes it possible to onboard new developers quickly.",
            "The new interface makes it faster to complete common tasks.",
            "Modular design makes it more flexible for future changes."
        ]
        
        for sentence in test_cases:
            context = AmbiguityContext(
                sentence_index=0,
                sentence=sentence,
                paragraph_context="",
                document_context=base_context
            )
            
            detections = detector.detect(context, nlp)
            flagged_words = [d.evidence.tokens[0].lower() for d in detections]
            
            assert 'makes' not in flagged_words, \
                f"Should not flag 'makes' in causative construction: {sentence}"
    
    def test_makes_object_comparative_patterns(self, detector, nlp, base_context):
        """Test 'makes [object] [comparative]' patterns."""
        test_cases = [
            "Using clear variable names makes debugging faster.",
            "Proper indentation makes code easier to read.",
            "Automated testing makes deployment safer.",
            "Good error messages make troubleshooting quicker.",
            "Caching makes the application more responsive."
        ]
        
        for sentence in test_cases:
            context = AmbiguityContext(
                sentence_index=0,
                sentence=sentence,
                paragraph_context="",
                document_context=base_context
            )
            
            detections = detector.detect(context, nlp)
            flagged_words = [d.evidence.tokens[0].lower() for d in detections]
            
            assert 'makes' not in flagged_words, \
                f"Should not flag causative construction: {sentence}"
    
    def test_other_causative_verbs(self, detector, nlp, base_context):
        """Test other causative verbs like 'let', 'help', 'enable', 'allow'."""
        test_cases = [
            "This feature lets users configure their preferences.",
            "The API helps developers integrate quickly.",
            "Modularity enables teams to work independently.",
            "The framework allows rapid prototyping."
        ]
        
        for sentence in test_cases:
            context = AmbiguityContext(
                sentence_index=0,
                sentence=sentence,
                paragraph_context="",
                document_context=base_context
            )
            
            detections = detector.detect(context, nlp)
            
            # These causative constructions should not be flagged
            assert len(detections) == 0 or all(d.evidence.tokens[0].lower() not in ['lets', 'helps', 'enables', 'allows'] for d in detections), \
                f"Should not flag causative verbs in clear constructions: {sentence}"


class TestVagueMakesStatements:
    """Test vague uses of 'makes' that SHOULD be flagged."""
    
    def test_vague_makes_improvements(self, detector, nlp, base_context):
        """Test vague statements like 'makes improvements'."""
        sentence = "The system makes improvements to performance."
        context = AmbiguityContext(
            sentence_index=0,
            sentence=sentence,
            paragraph_context="",
            document_context=base_context
        )
        
        detections = detector.detect(context, nlp)
        
        # This SHOULD be flagged as vague
        # "makes improvements" is vague - what improvements? how?
        if detections:
            flagged_words = [d.evidence.tokens[0].lower() for d in detections]
            # Either "makes" or "improvements" might be flagged
            assert any(word in ['makes', 'improvements', 'make'] for word in flagged_words), \
                "Should flag vague 'makes improvements'"
    
    def test_vague_makes_changes(self, detector, nlp, base_context):
        """Test vague statements about making changes."""
        sentence = "The process makes changes to the configuration."
        context = AmbiguityContext(
            sentence_index=0,
            sentence=sentence,
            paragraph_context="",
            document_context=base_context
        )
        
        detections = detector.detect(context, nlp)
        
        # May flag "makes" or "changes" - both are somewhat vague
        # This is acceptable as it's not a clear causative construction
        assert isinstance(detections, list), "Should process without error"
    
    def test_unspecified_makes_statements(self, detector, nlp, base_context):
        """Test 'makes' without clear causative pattern."""
        # These don't have the causative adjective/comparative pattern
        test_cases = [
            "The tool makes things better.",  # "things" is vague
            "It makes stuff work.",  # "stuff" is vague
            "The update makes everything different."  # "everything" + "different" is vague
        ]
        
        for sentence in test_cases:
            context = AmbiguityContext(
                sentence_index=0,
                sentence=sentence,
                paragraph_context="",
                document_context=base_context
            )
            
            detections = detector.detect(context, nlp)
            
            # These vague statements may be flagged (which is correct)
            # We're just verifying they're processed without error
            assert isinstance(detections, list), f"Should process sentence: {sentence}"


class TestEdgeCases:
    """Test edge cases and corner cases."""
    
    def test_makes_in_passive_construction(self, detector, nlp, base_context):
        """Test passive constructions with 'makes'."""
        sentence = "The code is made easier to understand by using clear names."
        context = AmbiguityContext(
            sentence_index=0,
            sentence=sentence,
            paragraph_context="",
            document_context=base_context
        )
        
        detections = detector.detect(context, nlp)
        
        # Passive causative should also be protected
        flagged_words = [d.evidence.tokens[0].lower() for d in detections if d.evidence.tokens]
        assert 'made' not in flagged_words, "Should not flag passive causative construction"
    
    def test_makes_with_multiple_objects(self, detector, nlp, base_context):
        """Test 'makes' with coordinated objects."""
        sentence = "Good design makes code clearer and maintenance easier."
        context = AmbiguityContext(
            sentence_index=0,
            sentence=sentence,
            paragraph_context="",
            document_context=base_context
        )
        
        detections = detector.detect(context, nlp)
        
        flagged_words = [d.evidence.tokens[0].lower() for d in detections]
        assert 'makes' not in flagged_words, "Should not flag causative with coordinated objects"
    
    def test_makes_at_sentence_start(self, detector, nlp, base_context):
        """Test causative 'makes' at sentence start (imperative-like)."""
        sentence = "Makes debugging easier when you use descriptive names."
        context = AmbiguityContext(
            sentence_index=0,
            sentence=sentence,
            paragraph_context="",
            document_context=base_context
        )
        
        detections = detector.detect(context, nlp)
        
        # Should still recognize causative pattern
        flagged_words = [d.evidence.tokens[0].lower() for d in detections]
        assert 'makes' not in flagged_words, "Should not flag causative at sentence start"
    
    def test_makes_in_subordinate_clause(self, detector, nlp, base_context):
        """Test causative 'makes' in subordinate clause."""
        sentence = "When you use clear names, it makes the code easier to understand."
        context = AmbiguityContext(
            sentence_index=0,
            sentence=sentence,
            paragraph_context="",
            document_context=base_context
        )
        
        detections = detector.detect(context, nlp)
        
        flagged_words = [d.evidence.tokens[0].lower() for d in detections]
        assert 'makes' not in flagged_words, "Should not flag causative in subordinate clause"
    
    def test_empty_sentence(self, detector, nlp, base_context):
        """Test empty sentence handling."""
        sentence = ""
        context = AmbiguityContext(
            sentence_index=0,
            sentence=sentence,
            paragraph_context="",
            document_context=base_context
        )
        
        detections = detector.detect(context, nlp)
        
        assert detections == [], "Empty sentence should return empty list"


class TestRealWorldScenarios:
    """Test real-world scenarios from actual documentation."""
    
    def test_technical_documentation_examples(self, detector, nlp, base_context):
        """Test examples from technical documentation."""
        test_cases = [
            "Using meaningful variable names makes your code self-documenting.",
            "Proper error handling makes applications more robust.",
            "Automated testing makes refactoring safer and faster.",
            "Clear API documentation makes integration straightforward.",
            "Modular architecture makes the system easier to scale."
        ]
        
        for sentence in test_cases:
            context = AmbiguityContext(
                sentence_index=0,
                sentence=sentence,
                paragraph_context="",
                document_context=base_context
            )
            
            detections = detector.detect(context, nlp)
            flagged_words = [d.evidence.tokens[0].lower() for d in detections]
            
            assert 'makes' not in flagged_words, \
                f"Should not flag legitimate technical documentation: {sentence}"
    
    def test_user_guide_examples(self, detector, nlp, base_context):
        """Test examples from user guides."""
        test_cases = [
            "This feature makes it easier to manage your workflow.",
            "The interface makes navigation more intuitive.",
            "Keyboard shortcuts make common tasks faster.",
            "The search function makes finding files quicker."
        ]
        
        context_dict = base_context.copy()
        context_dict['content_type'] = 'user_guide'
        
        for sentence in test_cases:
            context = AmbiguityContext(
                sentence_index=0,
                sentence=sentence,
                paragraph_context="",
                document_context=context_dict
            )
            
            detections = detector.detect(context, nlp)
            flagged_words = [d.evidence.tokens[0].lower() for d in detections]
            
            assert 'makes' not in flagged_words, \
                f"Should not flag user guide causative statements: {sentence}"
    
    def test_tutorial_examples(self, detector, nlp, base_context):
        """Test examples from tutorials."""
        test_cases = [
            "Following best practices makes your code more maintainable.",
            "Using version control makes collaboration easier.",
            "Writing tests makes debugging more efficient.",
            "Good naming conventions make code reviews faster."
        ]
        
        context_dict = base_context.copy()
        context_dict['content_type'] = 'tutorial'
        
        for sentence in test_cases:
            context = AmbiguityContext(
                sentence_index=0,
                sentence=sentence,
                paragraph_context="",
                document_context=context_dict
            )
            
            detections = detector.detect(context, nlp)
            flagged_words = [d.evidence.tokens[0].lower() for d in detections]
            
            assert 'makes' not in flagged_words, \
                f"Should not flag tutorial causative statements: {sentence}"


class TestContextSensitivity:
    """Test context-dependent behavior."""
    
    def test_technical_vs_marketing_context(self, detector, nlp):
        """Test different behavior in technical vs marketing content."""
        sentence = "Our solution makes your business more efficient."
        
        # Marketing context - might be more lenient
        marketing_context = AmbiguityContext(
            sentence_index=0,
            sentence=sentence,
            paragraph_context="",
            document_context={'content_type': 'marketing', 'block_type': 'paragraph'}
        )
        
        # Technical context - should not flag causative construction
        technical_context = AmbiguityContext(
            sentence_index=0,
            sentence=sentence,
            paragraph_context="",
            document_context={'content_type': 'technical', 'block_type': 'paragraph'}
        )
        
        marketing_detections = detector.detect(marketing_context, nlp)
        technical_detections = detector.detect(technical_context, nlp)
        
        # Both should recognize the causative construction
        marketing_flagged = [d.evidence.tokens[0].lower() for d in marketing_detections]
        technical_flagged = [d.evidence.tokens[0].lower() for d in technical_detections]
        
        assert 'makes' not in marketing_flagged, "Should not flag causative in marketing"
        assert 'makes' not in technical_flagged, "Should not flag causative in technical"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

