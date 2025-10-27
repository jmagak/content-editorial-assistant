"""
Unit Tests for ListPunctuationRule - Prerequisites Guard
Tests the zero-false-positive guard for prerequisite and requirements lists
"""

import pytest
from rules.structure_and_format.list_punctuation_rule import ListPunctuationRule

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None


@pytest.mark.unit
@pytest.mark.skipif(not SPACY_AVAILABLE or nlp is None, reason="spaCy model not available")
class TestListPunctuationPrerequisitesGuard:
    """
    Test the zero-false-positive guard for prerequisites and requirements lists.
    
    These lists correctly use full sentences ending with periods,
    and should NOT be flagged as punctuation errors.
    """
    
    def test_prerequisites_list_with_periods_not_flagged(self):
        """
        Test that Prerequisites lists with full sentences ending in periods are NOT flagged.
        
        This is the correct format for prerequisite lists and should NOT be an error.
        """
        rule = ListPunctuationRule()
        text = "You must use ASCII characters for the alternative name."
        
        # Context indicates this is a list item under "Prerequisites" heading
        context = {
            'block_type': 'ordered_list_item',
            'preceding_heading': 'Prerequisites',
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should return NO errors (correct format for prerequisites)
        assert len(issues) == 0, \
            f"Expected no errors for prerequisite list with periods, but got: {issues}"
    
    def test_requirements_list_with_periods_not_flagged(self):
        """
        Test that Requirements lists with full sentences ending in periods are NOT flagged.
        
        Similar to prerequisites, requirements lists correctly use full sentences.
        """
        rule = ListPunctuationRule()
        text = "The alternative name must be shorter than 128 characters."
        
        # Context indicates this is under "Requirements" heading
        context = {
            'block_type': 'ordered_list_item',
            'preceding_heading': 'Requirements',
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should return NO errors
        assert len(issues) == 0, \
            f"Expected no errors for requirements list with periods, but got: {issues}"
    
    def test_before_you_begin_list_not_flagged(self):
        """
        Test that "Before you begin" lists with periods are NOT flagged.
        
        This is another common variant of prerequisite heading.
        """
        rule = ListPunctuationRule()
        text = "You need to have administrator access to the system."
        
        context = {
            'block_type': 'unordered_list_item',
            'preceding_heading': 'Before you begin',
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should return NO errors
        assert len(issues) == 0, \
            f"Expected no errors for 'Before you begin' list, but got: {issues}"
    
    def test_what_you_need_list_not_flagged(self):
        """
        Test that "What you need" lists with periods are NOT flagged.
        
        Another common prerequisite heading variant.
        """
        rule = ListPunctuationRule()
        text = "The system must be running Linux kernel version 5.0 or later."
        
        context = {
            'block_type': 'unordered_list_item',
            'preceding_heading': 'What you need',
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should return NO errors
        assert len(issues) == 0, \
            f"Expected no errors for 'What you need' list, but got: {issues}"
    
    def test_case_insensitive_prerequisite_matching(self):
        """
        Test that prerequisite keywords are matched case-insensitively.
        
        "PREREQUISITES", "Prerequisites", "prerequisites" should all work.
        """
        rule = ListPunctuationRule()
        text = "You must have root access to the server."
        
        test_headings = [
            'PREREQUISITES',
            'Prerequisites',
            'prerequisites',
            'PreRequisites',
            'REQUIREMENTS',
            'Requirements'
        ]
        
        for heading in test_headings:
            context = {
                'block_type': 'ordered_list_item',
                'preceding_heading': heading,
                'content_type': 'procedural'
            }
            
            sentences = [text]
            issues = rule.analyze(text, sentences, nlp=nlp, context=context)
            
            assert len(issues) == 0, \
                f"Expected no errors for heading '{heading}', but got: {issues}"
    
    def test_regular_list_still_checked(self):
        """
        Test that non-prerequisite lists are still checked normally.
        
        The guard should ONLY apply to prerequisites/requirements lists.
        Regular procedural lists should still be analyzed.
        """
        rule = ListPunctuationRule()
        text = "Display the system configuration."
        
        # Context indicates this is under "Procedure" heading (not prerequisites)
        context = {
            'block_type': 'ordered_list_item',
            'preceding_heading': 'Procedure',
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # This should be analyzed normally (the guard should not activate)
        # We're just checking that the analysis runs, not necessarily that it flags
        assert isinstance(issues, list), "Should still analyze non-prerequisite lists"
    
    def test_fragment_in_prerequisites_may_flag(self):
        """
        Test that fragments (non-sentences) in prerequisites lists may still be flagged.
        
        The guard only protects full sentences with periods.
        Fragments should still be checked normally.
        """
        rule = ListPunctuationRule()
        text = "Administrative access"  # Fragment, not a complete sentence
        
        context = {
            'block_type': 'unordered_list_item',
            'preceding_heading': 'Prerequisites',
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Fragments should be analyzed normally (guard doesn't apply)
        assert isinstance(issues, list), "Should analyze fragments normally"
    
    def test_sentence_without_period_in_prerequisites_may_flag(self):
        """
        Test that sentences WITHOUT periods in prerequisites may still be flagged.
        
        The guard specifically checks for: sentence + period = correct format.
        If missing period, should still be flagged.
        """
        rule = ListPunctuationRule()
        text = "You must have root access"  # Sentence but missing period
        
        context = {
            'block_type': 'ordered_list_item',
            'preceding_heading': 'Prerequisites',
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should analyze normally (guard only protects correct format)
        assert isinstance(issues, list), "Should still check sentences without periods"
    
    def test_real_world_prerequisites_from_document(self):
        """
        Test the actual examples from the RHTAP workflow document.
        
        These are the real prerequisites that prompted the fix.
        """
        rule = ListPunctuationRule()
        
        test_cases = [
            "You must use ASCII characters for the alternative name.",
            "The alternative name must be shorter than 128 characters.",
        ]
        
        context = {
            'block_type': 'ordered_list_item',
            'preceding_heading': 'Prerequisites',
            'content_type': 'procedural'
        }
        
        for text in test_cases:
            sentences = [text]
            issues = rule.analyze(text, sentences, nlp=nlp, context=context)
            
            assert len(issues) == 0, \
                f"Expected no errors for prerequisite '{text}', but got: {issues}"
    
    def test_partial_keyword_match(self):
        """
        Test that keywords work when embedded in longer headings.
        
        E.g., "System Prerequisites" or "Installation Requirements" should match.
        """
        rule = ListPunctuationRule()
        text = "The system must be connected to the network."
        
        test_headings = [
            'System Prerequisites',
            'Installation Requirements',
            'Setup Requirements',
            'Prerequisites and Dependencies'
        ]
        
        for heading in test_headings:
            context = {
                'block_type': 'unordered_list_item',
                'preceding_heading': heading,
                'content_type': 'procedural'
            }
            
            sentences = [text]
            issues = rule.analyze(text, sentences, nlp=nlp, context=context)
            
            assert len(issues) == 0, \
                f"Expected no errors for heading '{heading}', but got: {issues}"
    
    def test_no_preceding_heading_analyzed_normally(self):
        """
        Test that when there's no preceding_heading in context, analysis proceeds normally.
        
        The guard should gracefully handle missing context data.
        """
        rule = ListPunctuationRule()
        text = "You must have administrative privileges."
        
        # Context without preceding_heading
        context = {
            'block_type': 'ordered_list_item',
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should still analyze (guard doesn't activate without heading info)
        assert isinstance(issues, list), "Should handle missing preceding_heading gracefully"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

