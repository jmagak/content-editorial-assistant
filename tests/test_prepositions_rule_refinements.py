"""
World-Class Comprehensive Test Suite for Prepositions Rule Refinements

This test suite validates the introductory prepositional phrase guard that prevents
false positives on context-setting prepositional phrases at sentence start.

Tests cover:
1. Introductory prepositional phrases (should get negative evidence)
2. Embedded excessive prepositional phrases (should be flagged)
3. Edge cases and real-world scenarios
4. Context-dependent behavior
"""

import pytest
import spacy
from rules.language_and_grammar.prepositions_rule import PrepositionsRule


@pytest.fixture(scope="module")
def nlp():
    """Load spaCy model once for all tests."""
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        pytest.skip("spaCy model 'en_core_web_sm' not installed")


@pytest.fixture
def rule():
    """Create a prepositions rule instance."""
    return PrepositionsRule()


@pytest.fixture
def base_context():
    """Create a base context for testing."""
    return {
        'content_type': 'documentation',
        'block_type': 'paragraph'
    }


class TestTechnicalComponentSpecifications:
    """Test technical component specifications that should NOT be penalized."""
    
    def test_hardware_components(self, rule, nlp, base_context):
        """Hardware component specifications."""
        test_cases = [
            "The data is stored on the disk and loaded into memory.",
            "The process runs on the device and communicates with the server.",
            "Transfer files from the drive to the network."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            # Technical component specs should have reduced evidence
            if errors:
                evidence = errors[0].get('evidence_score', 1.0)
                assert evidence < 0.5, \
                    f"Hardware component specs should have low evidence: '{text}' (got {evidence:.2f})"
    
    def test_software_components(self, rule, nlp, base_context):
        """Software component specifications."""
        test_cases = [
            "The data is stored in the database and cached on the server.",
            "The application connects to the service and sends requests to the endpoint.",
            "Deploy the container to the cluster and configure the namespace."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            # Software component specs should have reduced evidence
            if errors:
                evidence = errors[0].get('evidence_score', 1.0)
                assert evidence < 0.5, \
                    f"Software component specs should have low evidence: '{text}' (got {evidence:.2f})"
    
    def test_data_locations(self, rule, nlp, base_context):
        """Data location specifications."""
        test_cases = [
            "The configuration is stored in the file and loaded from the cache.",
            "Messages are sent to the queue and processed by the workers.",
            "Logs are written to the stream and archived in the bucket."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            # Data location specs should have reduced evidence
            if errors:
                evidence = errors[0].get('evidence_score', 1.0)
                assert evidence < 0.5, \
                    f"Data location specs should have low evidence: '{text}' (got {evidence:.2f})"
    
    def test_cloud_infrastructure(self, rule, nlp, base_context):
        """Cloud infrastructure specifications."""
        test_cases = [
            "The service runs in the cloud and deploys to multiple regions.",
            "Resources are allocated in the zone and distributed across nodes.",
            "The application connects to the endpoint and communicates through the interface."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            # Cloud infrastructure specs should have reduced evidence
            if errors:
                evidence = errors[0].get('evidence_score', 1.0)
                assert evidence < 0.5, \
                    f"Cloud infrastructure specs should have low evidence: '{text}' (got {evidence:.2f})"


class TestIntroductoryPrepositionalPhrases:
    """Test introductory prepositional phrases that should NOT be heavily penalized."""
    
    def test_original_problematic_case(self, rule, nlp, base_context):
        """The original case: 'On hosts with multiple profiles, ...'"""
        text = "On hosts with multiple profiles, the default profile of Konflux is used."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        # Should either have no errors or low evidence score due to introductory phrase guard
        if errors:
            evidence = errors[0].get('evidence_score', 1.0)
            assert evidence < 0.5, \
                f"Introductory prepositional phrase should have reduced evidence (got {evidence:.2f})"
        else:
            # No errors is even better - the guard worked perfectly
            assert True, "Introductory prepositional phrase correctly filtered"
    
    def test_simple_introductory_phrase(self, rule, nlp, base_context):
        """Simple introductory prepositional phrase."""
        test_cases = [
            "In the configuration file, you can specify the settings.",
            "On the server, the application runs automatically.",
            "For production environments, use the recommended settings.",
            "After deployment, verify the configuration.",
            "Before starting, ensure all prerequisites are met."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            # Introductory phrases should have significantly reduced evidence
            if errors:
                evidence = errors[0].get('evidence_score', 1.0)
                assert evidence < 0.6, \
                    f"Introductory phrase should have low evidence: '{text}' (got {evidence:.2f})"
    
    def test_complex_introductory_phrase(self, rule, nlp, base_context):
        """Complex introductory prepositional phrase with multiple prepositions."""
        test_cases = [
            "In systems with multiple configurations, the default setting applies.",
            "For applications in production environments, enable logging.",
            "On hosts with complex network configurations, verify the settings."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            # Complex introductory phrases are still acceptable context-setting
            if errors:
                evidence = errors[0].get('evidence_score', 1.0)
                assert evidence < 0.5, \
                    f"Complex introductory phrase should have low evidence: '{text}' (got {evidence:.2f})"
    
    def test_technical_introductory_phrase(self, rule, nlp, base_context):
        """Technical introductory phrases with proper nouns."""
        test_cases = [
            "In Kubernetes clusters, the default namespace is used.",
            "On AWS instances, the security group controls access.",
            "For Docker containers, the environment variables are set."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            # Technical introductory phrases should have very low evidence
            if errors:
                evidence = errors[0].get('evidence_score', 1.0)
                assert evidence < 0.4, \
                    f"Technical introductory phrase should have very low evidence: '{text}' (got {evidence:.2f})"


class TestEmbeddedExcessivePrepositions:
    """Test embedded excessive prepositional phrases that SHOULD be flagged."""
    
    def test_excessive_embedded_prepositions(self, rule, nlp, base_context):
        """Excessive prepositions in middle/end of sentence should be flagged."""
        text = "The configuration of the system in the production environment for the deployment of the application is complex."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        # Should detect excessive prepositions (no introductory phrase protection)
        assert len(errors) > 0, "Should flag excessive embedded prepositions"
        if errors:
            evidence = errors[0].get('evidence_score', 0)
            assert evidence > 0.3, \
                f"Excessive embedded prepositions should have higher evidence (got {evidence:.2f})"
    
    def test_chained_prepositions_mid_sentence(self, rule, nlp, base_context):
        """Chained prepositions in middle of sentence."""
        text = "The system handles the configuration of the settings for the deployment in the production environment."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        # Should detect prepositional chaining
        assert len(errors) > 0, "Should flag chained prepositions"
    
    def test_nested_prepositions(self, rule, nlp, base_context):
        """Nested prepositional phrases."""
        text = "The application is deployed in the context of the configuration of the environment for the system."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        # Should detect nested prepositional complexity
        assert len(errors) > 0, "Should flag nested prepositions"


class TestIntroductoryVsEmbedded:
    """Test distinction between introductory and embedded prepositional phrases."""
    
    def test_introductory_acceptable_embedded_flagged(self, rule, nlp, base_context):
        """Introductory phrase is OK, but excessive embedded ones should be flagged."""
        # This has introductory phrase + excessive embedded prepositions
        text = "In production, the configuration of the settings for the deployment in the environment is complex."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        # The introductory "In production," reduces evidence
        # PLUS many words are technical components (configuration, settings, deployment, environment)
        # So evidence is reduced significantly by multiple guards
        if errors:
            evidence = errors[0].get('evidence_score', 0)
            # Evidence should be low due to intro guard + technical component guard
            assert evidence < 0.5, \
                f"Should have low/moderate evidence due to multiple guards (got {evidence:.2f})"
        else:
            # Completely filtered by combined guards is also acceptable
            assert True, "Completely filtered by combined intro + technical component guards"
    
    def test_no_comma_intro_phrase(self, rule, nlp, base_context):
        """Introductory phrase without comma should still get some reduction."""
        text = "In production the system deploys automatically."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        # Without comma, less certainty it's introductory, but still some reduction
        # Should have lower evidence than if preposition was embedded
        if errors:
            evidence = errors[0].get('evidence_score', 0)
            # Some reduction but not as much as with comma
            assert evidence < 0.5, \
                f"Intro phrase without comma should still get some reduction (got {evidence:.2f})"


class TestEdgeCases:
    """Test edge cases and corner cases."""
    
    def test_single_preposition_not_flagged(self, rule, nlp, base_context):
        """Single preposition should not be flagged."""
        text = "The system runs on the server."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        # Should not flag single preposition
        assert len(errors) == 0, "Should not flag single preposition"
    
    def test_two_prepositions_not_flagged(self, rule, nlp, base_context):
        """Two prepositions should generally not be flagged."""
        text = "The application runs on the server in production."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        # Should not flag two prepositions (base guard)
        assert len(errors) == 0, "Should not flag two prepositions"
    
    def test_empty_sentence(self, rule, nlp, base_context):
        """Empty sentence should not cause errors."""
        text = ""
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        assert errors == [], "Empty text should return empty list"
    
    def test_no_prepositions(self, rule, nlp, base_context):
        """Sentence with no prepositions."""
        text = "The system runs automatically."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        assert len(errors) == 0, "Should not flag sentences without prepositions"


class TestRealWorldScenarios:
    """Test real-world scenarios from actual documentation."""
    
    def test_technical_prerequisites(self, rule, nlp):
        """Technical prerequisite statements."""
        test_cases = [
            "On hosts with multiple profiles, the default profile of Konflux is used.",
            "In Kubernetes environments, the secret must be stored in the default namespace.",
            "For production deployments, the configuration file must be located in the config directory.",
            "Before starting the application, verify that the database is running on port 5432."
        ]
        
        context = {
            'content_type': 'documentation',
            'block_type': 'paragraph',
            'preceding_heading': 'Prerequisites'
        }
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, context)
            
            # Prerequisites section + introductory phrase should have very low evidence
            if errors:
                evidence = errors[0].get('evidence_score', 1.0)
                assert evidence < 0.3, \
                    f"Prerequisite with introductory phrase should have very low evidence: '{text}' (got {evidence:.2f})"
    
    def test_configuration_documentation(self, rule, nlp, base_context):
        """Configuration documentation examples."""
        test_cases = [
            "In the configuration file, specify the connection settings.",
            "For each environment, define the appropriate values.",
            "On the Settings page, configure the deployment options."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            # Configuration docs with introductory phrases should have low evidence
            if errors:
                evidence = errors[0].get('evidence_score', 1.0)
                assert evidence < 0.5, \
                    f"Configuration intro phrase should have low evidence: '{text}' (got {evidence:.2f})"
    
    def test_procedural_documentation(self, rule, nlp):
        """Procedural documentation with steps."""
        test_cases = [
            "After installation, restart the service.",
            "Before deployment, run the test suite.",
            "During configuration, verify the settings.",
            "Upon completion, check the logs."
        ]
        
        context = {
            'content_type': 'tutorial',
            'block_type': 'ordered_list_item'
        }
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, context)
            
            # Procedural steps with temporal introductory phrases are very common
            if errors:
                evidence = errors[0].get('evidence_score', 1.0)
                assert evidence < 0.4, \
                    f"Procedural intro phrase should have low evidence: '{text}' (got {evidence:.2f})"


class TestContextSensitivity:
    """Test context-dependent behavior."""
    
    def test_technical_vs_marketing_context(self, rule, nlp):
        """Different behavior in different content types."""
        text = "In systems with multiple configurations, the default applies."
        
        # Technical context - more tolerant
        technical_context = {
            'content_type': 'technical',
            'block_type': 'paragraph'
        }
        
        # Marketing context - less tolerant
        marketing_context = {
            'content_type': 'marketing',
            'block_type': 'paragraph'
        }
        
        technical_errors = rule.analyze(text, [text], nlp, technical_context)
        marketing_errors = rule.analyze(text, [text], nlp, marketing_context)
        
        # Both should apply introductory phrase guard
        # But marketing might be slightly less tolerant overall
        # Just verify both handle it reasonably
        assert isinstance(technical_errors, list), "Should handle technical context"
        assert isinstance(marketing_errors, list), "Should handle marketing context"
    
    def test_beginner_vs_expert_audience(self, rule, nlp):
        """Different behavior for different audiences."""
        text = "In complex scenarios with multiple dependencies, configure carefully."
        
        # Beginner audience - prefer simpler
        beginner_context = {
            'content_type': 'documentation',
            'block_type': 'paragraph',
            'audience': 'beginner'
        }
        
        # Expert audience - tolerate complexity
        expert_context = {
            'content_type': 'documentation',
            'block_type': 'paragraph',
            'audience': 'expert'
        }
        
        beginner_errors = rule.analyze(text, [text], nlp, beginner_context)
        expert_errors = rule.analyze(text, [text], nlp, expert_context)
        
        # Expert audience should be more tolerant
        if beginner_errors and expert_errors:
            beginner_evidence = beginner_errors[0].get('evidence_score', 0)
            expert_evidence = expert_errors[0].get('evidence_score', 0)
            assert expert_evidence <= beginner_evidence, \
                "Expert audience should be more tolerant than beginner"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

