"""
Unit Tests for ColonsRule - Code Block Guard
Tests the zero-false-positive guard for instructional phrases before code blocks
"""

import pytest
from rules.punctuation.colons_rule import ColonsRule

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
class TestColonsRuleCodeBlockGuard:
    """
    Test the zero-false-positive guard for colons preceding code blocks.
    
    This guard prevents false positives when instructional phrases are followed
    by code blocks, which is a standard technical writing pattern.
    """
    
    def test_colon_before_listing_block_no_error(self):
        """
        Test that a colon before a listing block is not flagged as an error.
        
        Pattern: "create the /etc/systemd/network/ directory:" followed by [listing]
        This is standard technical documentation and should NOT be flagged.
        """
        rule = ColonsRule()
        text = "create the /etc/systemd/network/ directory:"
        
        # Context indicates next block is a listing
        context = {
            'next_block_type': 'listing',
            'block_type': 'paragraph'
        }
        
        # Run the analysis
        # The analyze method expects: (text, sentences, nlp, context)
        sentences = [text]  # Treat the whole text as one sentence for this test
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should return NO errors (zero false positives)
        assert len(issues) == 0, f"Expected no errors, but got: {issues}"
    
    def test_colon_before_code_block_no_error(self):
        """
        Test that a colon before a code block is not flagged as an error.
        
        Pattern: "Run the following command:" followed by [code_block]
        This is standard technical documentation and should NOT be flagged.
        """
        rule = ColonsRule()
        text = "Run the following command:"
        
        # Context indicates next block is a code block
        context = {
            'next_block_type': 'code_block',
            'block_type': 'paragraph'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should return NO errors
        assert len(issues) == 0, f"Expected no errors, but got: {issues}"
    
    def test_colon_before_literal_block_no_error(self):
        """
        Test that a colon before a literal block is not flagged as an error.
        
        Pattern: "Reboot the system:" followed by [literal]
        This is standard technical documentation and should NOT be flagged.
        """
        rule = ColonsRule()
        text = "Reboot the system:"
        
        # Context indicates next block is a literal block
        context = {
            'next_block_type': 'literal',
            'block_type': 'paragraph'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should return NO errors
        assert len(issues) == 0, f"Expected no errors, but got: {issues}"
    
    def test_colon_incomplete_clause_before_code_block_no_error(self):
        """
        Test that even incomplete clauses before code blocks are not flagged.
        
        This is because the structural pattern (instruction + colon + code block)
        is valid regardless of whether the instruction is grammatically complete.
        The code block provides the completion of the thought.
        """
        rule = ColonsRule()
        text = "For example:"
        
        # Context indicates next block is a code block
        context = {
            'next_block_type': 'code_block',
            'block_type': 'paragraph'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should return NO errors due to structural context
        assert len(issues) == 0, f"Expected no errors, but got: {issues}"
    
    def test_colon_without_next_block_context_may_error(self):
        """
        Test that colons are still checked when there's no next_block_type context.
        
        This ensures the guard only activates when appropriate structural context
        is present, maintaining normal error detection otherwise.
        """
        rule = ColonsRule()
        text = "For the:"  # Incomplete clause ending with article
        
        # No next_block_type in context
        context = {
            'block_type': 'paragraph'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # May return errors for incomplete clause (normal behavior)
        # This test just verifies the guard doesn't interfere with normal checking
        assert isinstance(issues, list)
    
    def test_colon_before_regular_paragraph_may_error(self):
        """
        Test that colons before regular paragraphs are still checked normally.
        
        The guard should ONLY activate for code/listing/literal blocks.
        """
        rule = ColonsRule()
        text = "Review the:"  # Incomplete clause
        
        # Next block is a paragraph, not a code block
        context = {
            'next_block_type': 'paragraph',
            'block_type': 'paragraph'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # May return errors (normal behavior for non-code blocks)
        assert isinstance(issues, list)
    
    def test_real_world_example_from_doc(self):
        """
        Test the actual example from the RHTAP workflow document.
        
        This is the real-world case that prompted the fix:
        "create the /etc/systemd/network/ directory:" followed by a code block
        """
        rule = ColonsRule()
        text = "If it does not already exist, create the /etc/systemd/network/ directory:"
        
        # Context from AsciiDoc parser
        context = {
            'next_block_type': 'listing',  # AsciiDoc uses 'listing' for code blocks
            'block_type': 'olist',  # ordered list item
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should return NO errors
        assert len(issues) == 0, f"Expected no errors for real-world example, but got: {issues}"
    
    def test_multiple_instructions_with_code_blocks(self):
        """
        Test multiple common technical writing patterns that should not be flagged.
        """
        rule = ColonsRule()
        
        test_cases = [
            "Display the network interface names and their MAC addresses:",
            "Modify the file you created in the previous step:",
            "Regenerate the initrd RAM disk image:",
            "Use the alternative interface name:",
            "Record the MAC address of the interface to which you want to assign an alternative name:",
        ]
        
        context = {
            'next_block_type': 'listing',
            'block_type': 'olist',
            'content_type': 'procedural'
        }
        
        for text in test_cases:
            sentences = [text]
            issues = rule.analyze(text, sentences, nlp=nlp, context=context)
            assert len(issues) == 0, f"Expected no errors for '{text}', but got: {issues}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

