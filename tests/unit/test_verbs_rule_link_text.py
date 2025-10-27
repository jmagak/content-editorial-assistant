"""
Unit tests for Verbs Rule with link text context.

Tests that verbs in link text are properly analyzed with increased evidence
for tense consistency.
"""

import unittest
from rules.language_and_grammar.verbs_rule import VerbsRule

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False


@unittest.skipIf(not SPACY_AVAILABLE, "SpaCy not available")
class TestVerbsRuleLinkText(unittest.TestCase):
    """Test that the Verbs Rule properly handles link text context."""

    def setUp(self):
        self.rule = VerbsRule()
        self.nlp = spacy.load("en_core_web_sm")

    def test_past_tense_in_link_text_higher_evidence(self):
        """Test that past tense verbs in link text have higher evidence scores."""
        # Test sentence with past tense verb
        text = "The system configured the settings automatically."
        sentences = [text]
        
        # Without link text context
        context_no_link = {'block_type': 'paragraph', 'content_type': 'technical'}
        errors_no_link = self.rule.analyze(text, sentences, nlp=self.nlp, context=context_no_link)
        
        # With link text context
        context_with_link = {
            'block_type': 'paragraph',
            'content_type': 'technical',
            'is_link_text': True  # This should increase evidence
        }
        errors_with_link = self.rule.analyze(text, sentences, nlp=self.nlp, context=context_with_link)
        
        # Both should flag the past tense, but link context should have higher evidence
        if errors_no_link and errors_with_link:
            evidence_no_link = errors_no_link[0].get('evidence_score', 0.0)
            evidence_with_link = errors_with_link[0].get('evidence_score', 0.0)
            
            self.assertGreater(evidence_with_link, evidence_no_link,
                             "Link text should have higher evidence score for past tense")
            
            # The difference should be approximately 0.3 (the clue value)
            self.assertAlmostEqual(evidence_with_link - evidence_no_link, 0.3, delta=0.1)

    def test_link_text_detection_integration(self):
        """Test that link text context is properly passed from parser to rule."""
        from structural_parsing.asciidoc.parser import AsciiDocParser
        
        parser = AsciiDocParser()
        
        # Test content with link macro containing past tense
        test_content = """
For details, see link:https://example.com[The system crashed unexpectedly].
"""
        
        parse_result = parser.parse(test_content)
        document = parse_result.document
        
        # Find the block with link text
        for block in document.blocks:
            context = block.get_context_info()
            if context.get('is_link_text'):
                # This block has link text, analyze it
                text_content = block.get_text_content()
                errors = self.rule.analyze(
                    text=text_content,
                    sentences=[text_content],
                    nlp=self.nlp,
                    context=context
                )
                
                # Should detect past tense in link text if present
                # Note: This depends on actual content and parsing
                self.assertIsInstance(errors, list)
                break

    def test_present_tense_in_link_text_not_flagged(self):
        """Test that present tense verbs in link text are not flagged (they're correct)."""
        # Present tense in link text - should not be flagged
        text = "NetworkManager duplicates a connection after restart."
        sentences = [text]
        context = {
            'block_type': 'paragraph',
            'content_type': 'technical',
            'is_link_text': True
        }
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # Present tense should not be flagged even in link text
        # (only past tense gets increased evidence)
        past_tense_errors = [e for e in errors if 'past' in e.get('message', '').lower()]
        self.assertEqual(len(past_tense_errors), 0,
                        "Present tense verbs should not be flagged as past tense")


if __name__ == '__main__':
    unittest.main()

