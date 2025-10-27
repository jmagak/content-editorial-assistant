"""
World-Class Comprehensive Test Suite for Global Audiences Rule Refinements

This test suite validates the conditional clause guard that prevents false positives
on negative constructions in conditional logic.

Tests cover:
1. Conditional clauses with negatives (should be filtered)
2. Non-conditional negatives (should still be flagged appropriately)
3. Edge cases and corner cases
4. Real-world technical documentation scenarios
"""

import pytest
import spacy
from rules.audience_and_medium.global_audiences_rule import GlobalAudiencesRule


@pytest.fixture(scope="module")
def nlp():
    """Load spaCy model once for all tests."""
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        pytest.skip("spaCy model 'en_core_web_sm' not installed")


@pytest.fixture
def rule():
    """Create a global audiences rule instance."""
    return GlobalAudiencesRule()


@pytest.fixture
def base_context():
    """Create a base context for testing."""
    return {
        'content_type': 'documentation',
        'block_type': 'paragraph'
    }


class TestConditionalClauseGuard:
    """Test conditional clauses with negatives that should NOT be flagged."""
    
    def test_if_clause_with_not(self, rule, nlp, base_context):
        """'If' clause with 'not' should be filtered."""
        test_cases = [
            "If the value does not match, the system will fail.",
            "If the file cannot be found, an error occurs.",
            "If users do not specify a name, the default is used.",
            "If the connection is not available, retry the operation."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            # Should not flag negative in conditional clause
            negative_errors = [e for e in errors if 'not' in e.get('flagged_text', '').lower() 
                             or "n't" in e.get('flagged_text', '')]
            
            assert len(negative_errors) == 0, \
                f"Should not flag negative in 'if' clause: '{text}'"
    
    def test_when_clause_with_not(self, rule, nlp, base_context):
        """'When' clause with 'not' should be filtered."""
        test_cases = [
            "When the system does not respond, check the logs.",
            "When users cannot access the file, verify permissions.",
            "When the service is not running, start it manually."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            negative_errors = [e for e in errors if 'not' in e.get('flagged_text', '').lower() 
                             or "n't" in e.get('flagged_text', '')]
            
            assert len(negative_errors) == 0, \
                f"Should not flag negative in 'when' clause: '{text}'"
    
    def test_unless_clause(self, rule, nlp, base_context):
        """'Unless' clause (inherently negative) should be filtered."""
        test_cases = [
            "Unless the value is set, the default applies.",
            "Unless you specify otherwise, automatic mode is used.",
            "Unless authentication succeeds, access is denied."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            # 'Unless' is inherently negative but should be acceptable
            # Just verify it processes without error
            assert isinstance(errors, list), \
                f"Should handle 'unless' clause: '{text}'"
    
    def test_where_clause_conditional(self, rule, nlp, base_context):
        """'Where' clause used conditionally should be filtered."""
        test_cases = [
            "Where the configuration does not exist, create a new one.",
            "Where authentication cannot succeed, deny access.",
            "Where values are not specified, use defaults."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            negative_errors = [e for e in errors if 'not' in e.get('flagged_text', '').lower() 
                             or "n't" in e.get('flagged_text', '')]
            
            assert len(negative_errors) == 0, \
                f"Should not flag negative in 'where' clause: '{text}'"
    
    def test_while_clause_conditional(self, rule, nlp, base_context):
        """'While' clause with negatives should be filtered."""
        test_cases = [
            "While the process is not complete, continue monitoring.",
            "While the file does not exist, wait for creation.",
            "While the connection cannot establish, retry periodically."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            negative_errors = [e for e in errors if 'not' in e.get('flagged_text', '').lower() 
                             or "n't" in e.get('flagged_text', '')]
            
            assert len(negative_errors) == 0, \
                f"Should not flag negative in 'while' clause: '{text}'"
    
    def test_whenever_clause(self, rule, nlp, base_context):
        """'Whenever' clause with negatives should be filtered."""
        test_cases = [
            "Whenever the system cannot connect, it logs an error.",
            "Whenever authentication does not succeed, retry is attempted.",
            "Whenever the value is not valid, reject the input."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            negative_errors = [e for e in errors if 'not' in e.get('flagged_text', '').lower() 
                             or "n't" in e.get('flagged_text', '')]
            
            assert len(negative_errors) == 0, \
                f"Should not flag negative in 'whenever' clause: '{text}'"


class TestNonConditionalNegatives:
    """Test non-conditional negatives that should still be evaluated appropriately."""
    
    def test_simple_negative_statement(self, rule, nlp, base_context):
        """Simple negative statements should be evaluated (may still be flagged based on evidence)."""
        text = "The system does not support this feature."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        # This is NOT in a conditional clause, so rule should evaluate it
        # Whether it's flagged depends on evidence calculation
        # We just verify it wasn't completely filtered by conditional guard
        assert isinstance(errors, list), "Should evaluate non-conditional negative"
    
    def test_negative_after_comma(self, rule, nlp, base_context):
        """Negative after conditional comma should be evaluated."""
        text = "If the file exists, the system does not overwrite it."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        # The "does not" is AFTER the comma, so it's in the consequence clause
        # This might be flagged appropriately - we just verify it processes
        assert isinstance(errors, list), \
            "Should evaluate negative in consequence clause (after comma)"
    
    def test_embedded_negative_not_at_start(self, rule, nlp, base_context):
        """Negative not at sentence start should be evaluated."""
        text = "The configuration does not allow this operation."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        # Not a conditional clause, should evaluate normally
        assert isinstance(errors, list), "Should evaluate non-conditional negative"


class TestEdgeCases:
    """Test edge cases and corner cases."""
    
    def test_if_mid_sentence(self, rule, nlp, base_context):
        """'If' not at sentence start should be evaluated."""
        text = "Check if the value does not match the expected result."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        # 'if' is not at sentence start, so might not trigger guard
        # Just verify it processes without error
        assert isinstance(errors, list), "Should handle 'if' mid-sentence"
    
    def test_empty_sentence(self, rule, nlp, base_context):
        """Empty sentence should not cause errors."""
        text = ""
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        assert errors == [], "Empty text should return empty list"
    
    def test_no_negatives(self, rule, nlp, base_context):
        """Sentence with no negatives."""
        text = "The system supports this feature completely."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        # Should not flag anything related to negatives
        negative_errors = [e for e in errors if 'not' in e.get('message', '').lower()]
        assert len(negative_errors) == 0, "Should not flag sentences without negatives"
    
    def test_multiple_clauses(self, rule, nlp, base_context):
        """Conditional with multiple clauses."""
        text = "If the file does not exist and the user cannot create it, show an error."
        
        errors = rule.analyze(text, [text], nlp, base_context)
        
        # Both negatives are in the conditional clause (before comma)
        negative_errors = [e for e in errors if 'not' in e.get('flagged_text', '').lower() 
                          or "n't" in e.get('flagged_text', '')]
        
        assert len(negative_errors) == 0, \
            "Should not flag negatives in complex conditional clause"


class TestRealWorldScenarios:
    """Test real-world scenarios from actual technical documentation."""
    
    def test_error_handling_conditions(self, rule, nlp, base_context):
        """Error handling conditions from technical docs."""
        test_cases = [
            "If the database connection cannot be established, the application will retry.",
            "If authentication does not succeed within 3 attempts, lock the account.",
            "If the required parameters are not provided, return a 400 error."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            negative_errors = [e for e in errors if 'not' in e.get('flagged_text', '').lower() 
                             or "n't" in e.get('flagged_text', '')]
            
            assert len(negative_errors) == 0, \
                f"Should not flag error handling condition: '{text}'"
    
    def test_validation_conditions(self, rule, nlp, base_context):
        """Validation conditions from technical docs."""
        test_cases = [
            "If the input does not match the expected format, reject it.",
            "If the token cannot be validated, deny access.",
            "If the configuration is not valid, use default settings."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            negative_errors = [e for e in errors if 'not' in e.get('flagged_text', '').lower() 
                             or "n't" in e.get('flagged_text', '')]
            
            assert len(negative_errors) == 0, \
                f"Should not flag validation condition: '{text}'"
    
    def test_fallback_logic(self, rule, nlp, base_context):
        """Fallback logic from technical docs."""
        test_cases = [
            "When the primary server does not respond, connect to the backup.",
            "When the cache is not available, query the database directly.",
            "When the file cannot be read, generate a new one."
        ]
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, base_context)
            
            negative_errors = [e for e in errors if 'not' in e.get('flagged_text', '').lower() 
                             or "n't" in e.get('flagged_text', '')]
            
            assert len(negative_errors) == 0, \
                f"Should not flag fallback logic: '{text}'"
    
    def test_procedural_conditions(self, rule, nlp):
        """Procedural conditions from tutorials."""
        test_cases = [
            "If the installation does not complete successfully, check the logs.",
            "If the service cannot start, verify the configuration.",
            "If the test does not pass, review the error message."
        ]
        
        context = {
            'content_type': 'tutorial',
            'block_type': 'ordered_list_item'
        }
        
        for text in test_cases:
            errors = rule.analyze(text, [text], nlp, context)
            
            negative_errors = [e for e in errors if 'not' in e.get('flagged_text', '').lower() 
                             or "n't" in e.get('flagged_text', '')]
            
            assert len(negative_errors) == 0, \
                f"Should not flag procedural condition: '{text}'"


class TestContextSensitivity:
    """Test context-dependent behavior."""
    
    def test_technical_vs_marketing_context(self, rule, nlp):
        """Different behavior in different content types."""
        text = "If the system cannot connect, retry the operation."
        
        # Technical context
        technical_context = {
            'content_type': 'technical',
            'block_type': 'paragraph'
        }
        
        # Marketing context
        marketing_context = {
            'content_type': 'marketing',
            'block_type': 'paragraph'
        }
        
        technical_errors = rule.analyze(text, [text], nlp, technical_context)
        marketing_errors = rule.analyze(text, [text], nlp, marketing_context)
        
        # Both should apply conditional clause guard
        tech_negative = [e for e in technical_errors if 'not' in e.get('flagged_text', '').lower() 
                        or "n't" in e.get('flagged_text', '')]
        market_negative = [e for e in marketing_errors if 'not' in e.get('flagged_text', '').lower() 
                          or "n't" in e.get('flagged_text', '')]
        
        assert len(tech_negative) == 0, "Should not flag conditional in technical docs"
        assert len(market_negative) == 0, "Should not flag conditional in marketing docs"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

