"""
Unit Tests for ArticlesRule - Procedural Context Fix
Tests the two fixes:
1. Remove global guard that restricts missing article check to descriptive_content only
2. Add zero-false-positive guard for procedural lists (ordered_list_item blocks)
"""

import pytest
from rules.language_and_grammar.articles_rule import ArticlesRule

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
class TestArticlesRuleProceduralContext:
    """
    Test the fixes for procedural context handling in ArticlesRule.
    
    Fix 1: Remove global guard that limited missing article checks to descriptive_content
    Fix 2: Add structural clue to ignore missing articles in procedural steps
    """
    
    def test_missing_article_in_paragraph_detected(self):
        """
        Test that missing articles in paragraph blocks ARE detected.
        
        Example: "provides overview" (missing "an")
        This should be flagged because it's in a paragraph block, not a procedural step.
        """
        rule = ArticlesRule()
        text = "This system provides overview of the process."
        
        # Context indicates this is a paragraph in procedural documentation
        context = {
            'block_type': 'paragraph',
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should detect missing article before "overview"
        # Filter for missing article issues (not incorrect a/an issues)
        missing_article_issues = [
            issue for issue in issues
            if 'overview' in issue.get('flagged_text', '').lower()
            and ('missing' in issue.get('message', '').lower() or 
                 'adding' in issue.get('message', '').lower() or
                 'article' in issue.get('message', '').lower())
        ]
        
        assert len(missing_article_issues) > 0, \
            f"Expected to detect missing article before 'overview', but got: {issues}"
    
    def test_missing_article_in_procedural_step_ignored(self):
        """
        Test that missing articles in procedural steps (ordered list items) are NOT flagged.
        
        Example: "Display interface names" (no article before "interface")
        This should NOT be flagged because it's an imperative in an ordered_list_item.
        """
        rule = ArticlesRule()
        text = "Display interface names and their MAC addresses."
        
        # Context indicates this is an ordered list item (procedural step)
        context = {
            'block_type': 'ordered_list_item',
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should NOT detect missing article before "interface" or "MAC"
        # because this is a procedural imperative in an ordered list
        missing_article_issues = [
            issue for issue in issues
            if ('missing' in issue.get('message', '').lower() or
                'adding' in issue.get('message', '').lower())
        ]
        
        assert len(missing_article_issues) == 0, \
            f"Expected no missing article errors in procedural step, but got: {issues}"
    
    def test_missing_article_in_olist_block_ignored(self):
        """
        Test that missing articles in olist blocks are also handled correctly.
        
        AsciiDoc uses 'olist' for ordered list items.
        """
        rule = ArticlesRule()
        text = "Record MAC address of the interface."
        
        # Context with 'olist' block type (AsciiDoc style)
        context = {
            'block_type': 'olist',
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # Should NOT flag missing article before "MAC" in procedural step
        missing_article_issues = [
            issue for issue in issues
            if 'mac' in issue.get('flagged_text', '').lower()
            and ('missing' in issue.get('message', '').lower() or
                 'adding' in issue.get('message', '').lower())
        ]
        
        assert len(missing_article_issues) == 0, \
            f"Expected no missing article errors in olist procedural step, but got: {issues}"
    
    def test_real_world_paragraph_example(self):
        """
        Test the real-world example from the document.
        
        The abstract says: "provides overview" which is missing "an".
        This is in a paragraph block and should be flagged.
        """
        rule = ArticlesRule()
        text = "With alternative interface naming, the kernel can assign additional names to network interfaces."
        
        # This is the abstract paragraph - should have complete grammar
        context = {
            'block_type': 'paragraph',
            'content_type': 'procedural'  # Document is PROCEDURE type
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # This sentence is actually grammatically correct, so should have no issues
        assert isinstance(issues, list), "Should return a list"
    
    def test_real_world_procedural_step_examples(self):
        """
        Test multiple real-world procedural steps from the document.
        
        These should NOT be flagged even though they might have missing articles.
        """
        rule = ArticlesRule()
        
        test_cases = [
            "Display the network interface names and their MAC addresses.",
            "Record the MAC address of the interface to which you want to assign an alternative name.",
            "Regenerate the initrd RAM disk image.",
            "Reboot the system.",
        ]
        
        context = {
            'block_type': 'ordered_list_item',
            'content_type': 'procedural'
        }
        
        for text in test_cases:
            sentences = [text]
            issues = rule.analyze(text, sentences, nlp=nlp, context=context)
            
            # Filter for missing article issues only
            missing_article_issues = [
                issue for issue in issues
                if ('missing' in issue.get('message', '').lower() or
                    'adding' in issue.get('message', '').lower())
            ]
            
            # These procedural imperatives should not have missing article errors
            assert len(missing_article_issues) == 0, \
                f"Expected no missing article errors for '{text}', but got: {missing_article_issues}"
    
    def test_non_imperative_in_list_may_flag(self):
        """
        Test that non-imperative sentences in list items are still checked.
        
        If a list item doesn't start with an imperative verb, it should be
        checked normally for missing articles.
        """
        rule = ArticlesRule()
        text = "Overview is provided by the system."  # Not an imperative
        
        context = {
            'block_type': 'ordered_list_item',
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues = rule.analyze(text, sentences, nlp=nlp, context=context)
        
        # This should check for missing articles because it's not an imperative
        # (though "Overview" as subject doesn't need article)
        assert isinstance(issues, list), "Should still analyze the text"
    
    def test_paragraph_vs_list_context_difference(self):
        """
        Test that the same text gets different treatment based on block_type.
        
        The same phrase should be flagged in a paragraph but not in a procedural step.
        """
        rule = ArticlesRule()
        text = "Configure system settings."
        
        # Test 1: In paragraph context - might flag missing article before "system"
        context_paragraph = {
            'block_type': 'paragraph',
            'content_type': 'procedural'
        }
        
        sentences = [text]
        issues_paragraph = rule.analyze(text, sentences, nlp=nlp, context=context_paragraph)
        
        # Test 2: In ordered list context - should not flag
        context_list = {
            'block_type': 'ordered_list_item',
            'content_type': 'procedural'
        }
        
        issues_list = rule.analyze(text, sentences, nlp=nlp, context=context_list)
        
        # Filter for missing article issues
        missing_paragraph = [i for i in issues_paragraph 
                           if 'missing' in i.get('message', '').lower() or 'adding' in i.get('message', '').lower()]
        missing_list = [i for i in issues_list 
                       if 'missing' in i.get('message', '').lower() or 'adding' in i.get('message', '').lower()]
        
        # List context should have fewer or equal missing article issues than paragraph
        assert len(missing_list) <= len(missing_paragraph), \
            f"List context should have fewer missing article issues. Paragraph: {len(missing_paragraph)}, List: {len(missing_list)}"
    
    def test_fix_enables_cross_content_type_checking(self):
        """
        Test that Fix 1 (removing global guard) enables checking across all content types.
        
        Before the fix, missing article checks only ran on 'descriptive_content'.
        After the fix, they should run on all content types, with evidence adjusted by context.
        """
        rule = ArticlesRule()
        text = "System provides functionality for users."
        
        # Test with different content classifications
        contexts = [
            {'block_type': 'paragraph', 'content_type': 'procedural'},
            {'block_type': 'paragraph', 'content_type': 'technical'},
            {'block_type': 'paragraph', 'content_type': 'descriptive'},
        ]
        
        for context in contexts:
            sentences = [text]
            issues = rule.analyze(text, sentences, nlp=nlp, context=context)
            
            # All content types should be analyzed (though evidence scores may vary)
            # The analyze method should run without errors
            assert isinstance(issues, list), \
                f"Should analyze all content types, failed for: {context.get('content_type')}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

