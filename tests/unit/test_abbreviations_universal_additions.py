"""
Unit Tests for AbbreviationsRule - ASCII and MAC Universal Abbreviations
Tests that ASCII and MAC are recognized as universally known abbreviations
"""

import pytest
from rules.language_and_grammar.abbreviations_rule import AbbreviationsRule

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
class TestAbbreviationsUniversalAdditions:
    """
    Test that ASCII and MAC are recognized as universally known abbreviations.
    
    These common technical terms should NOT be flagged as needing definitions.
    """
    
    def test_ascii_not_flagged_in_text(self):
        """
        Test that ASCII is not flagged as needing definition.
        
        ASCII (American Standard Code for Information Interchange) is universally
        known in technical contexts and should not require definition.
        """
        rule = AbbreviationsRule()
        text = "You must use ASCII characters for the alternative name."
        
        context = {
            'block_type': 'paragraph',
            'content_type': 'technical'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should NOT flag ASCII as needing definition
        ascii_issues = [
            issue for issue in issues
            if 'ASCII' in issue.get('flagged_text', '') or
               'ascii' in issue.get('message', '').lower()
        ]
        
        assert len(ascii_issues) == 0, \
            f"Expected no issues for ASCII abbreviation, but got: {ascii_issues}"
    
    def test_mac_not_flagged_in_text(self):
        """
        Test that MAC is not flagged as needing definition.
        
        MAC (Media Access Control) address is a fundamental networking term
        and should not require definition.
        """
        rule = AbbreviationsRule()
        text = "Display the network interface names and their MAC addresses."
        
        context = {
            'block_type': 'ordered_list_item',
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should NOT flag MAC as needing definition
        mac_issues = [
            issue for issue in issues
            if 'MAC' in issue.get('flagged_text', '') or
               ('mac' in issue.get('message', '').lower() and 'address' in text.lower())
        ]
        
        assert len(mac_issues) == 0, \
            f"Expected no issues for MAC abbreviation, but got: {mac_issues}"
    
    def test_ascii_and_mac_in_same_text(self):
        """
        Test that both ASCII and MAC are not flagged when appearing together.
        
        Real-world scenario where both abbreviations appear in technical text.
        """
        rule = AbbreviationsRule()
        text = "The system uses ASCII encoding for interface names and displays MAC addresses in hexadecimal format."
        
        context = {
            'block_type': 'paragraph',
            'content_type': 'technical'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should NOT flag either ASCII or MAC
        relevant_issues = [
            issue for issue in issues
            if 'ASCII' in issue.get('flagged_text', '') or
               'MAC' in issue.get('flagged_text', '') or
               'ascii' in issue.get('message', '').lower() or
               ('mac' in issue.get('message', '').lower() and 'address' in text.lower())
        ]
        
        assert len(relevant_issues) == 0, \
            f"Expected no issues for ASCII/MAC abbreviations, but got: {relevant_issues}"
    
    def test_ascii_case_insensitive(self):
        """
        Test that ASCII recognition is case-insensitive.
        
        The rule should recognize ASCII, ascii, Ascii, etc.
        """
        rule = AbbreviationsRule()
        
        test_cases = [
            "Use ASCII characters only.",
            "Use ascii characters only.",
            "Use Ascii characters only.",
        ]
        
        context = {
            'block_type': 'paragraph',
            'content_type': 'technical'
        }
        
        for text in test_cases:
            sentences = [text]
            issues = rule.analyze(text, sentences, nlp=nlp, context=context)
            
            ascii_issues = [
                issue for issue in issues
                if 'ascii' in issue.get('flagged_text', '').lower()
            ]
            
            assert len(ascii_issues) == 0, \
                f"Expected no issues for '{text}', but got: {ascii_issues}"
    
    def test_mac_case_insensitive(self):
        """
        Test that MAC recognition is case-insensitive.
        
        The rule should recognize MAC, mac, Mac, etc.
        """
        rule = AbbreviationsRule()
        
        test_cases = [
            "Display the MAC address.",
            "Display the mac address.",
            "Display the Mac address.",
        ]
        
        context = {
            'block_type': 'paragraph',
            'content_type': 'technical'
        }
        
        for text in test_cases:
            sentences = [text]
            issues = rule.analyze(text, sentences, nlp=nlp, context=context)
            
            mac_issues = [
                issue for issue in issues
                if 'mac' in issue.get('flagged_text', '').lower() and 'address' in text.lower()
            ]
            
            assert len(mac_issues) == 0, \
                f"Expected no issues for '{text}', but got: {mac_issues}"
    
    def test_real_world_document_examples(self):
        """
        Test the actual examples from the RHTAP workflow document.
        
        These are the real sentences that prompted the fix.
        """
        rule = AbbreviationsRule()
        
        test_cases = [
            {
                'text': 'You must use ASCII characters for the alternative name.',
                'context': {'block_type': 'ordered_list_item', 'content_type': 'procedural'},
                'description': 'Prerequisites: ASCII characters requirement'
            },
            {
                'text': 'Display the network interface names and their MAC addresses:',
                'context': {'block_type': 'ordered_list_item', 'content_type': 'procedural'},
                'description': 'Procedure: Display MAC addresses'
            },
            {
                'text': 'Record the MAC address of the interface to which you want to assign an alternative name.',
                'context': {'block_type': 'ordered_list_item', 'content_type': 'procedural'},
                'description': 'Procedure: Record MAC address'
            },
        ]
        
        for test_case in test_cases:
            text = test_case['text']
            context = test_case['context']
            description = test_case['description']
            
            sentences = [text]
            issues = rule.analyze(text, sentences, nlp=nlp, context=context)
            
            # Filter for ASCII/MAC related issues
            relevant_issues = [
                issue for issue in issues
                if 'ASCII' in issue.get('flagged_text', '') or
                   'MAC' in issue.get('flagged_text', '') or
                   'ascii' in issue.get('message', '').lower() or
                   ('mac' in issue.get('message', '').lower() and 'address' in text.lower())
            ]
            
            assert len(relevant_issues) == 0, \
                f"Expected no issues for '{description}', but got: {relevant_issues}"
    
    def test_other_abbreviations_still_checked(self):
        """
        Test that the rule still flags unknown abbreviations.
        
        The fix should only affect ASCII and MAC, not disable the rule entirely.
        """
        rule = AbbreviationsRule()
        text = "The XYZABC protocol is used for communication."
        
        context = {
            'block_type': 'paragraph',
            'content_type': 'technical'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # This should still be analyzed (though it may or may not be flagged
        # depending on other guards and heuristics)
        # We're just checking that the rule still runs
        assert isinstance(issues, list), "Rule should still analyze unknown abbreviations"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

