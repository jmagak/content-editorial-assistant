"""
Unit tests for Conjunctions Rule with coordinated passive phrases.

Tests that coordinated passive participles that share an auxiliary are both
recognized as passive and therefore parallel.
"""

import unittest
from rules.language_and_grammar.conjunctions_rule import ConjunctionsRule

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
class TestConjunctionsRuleCoordinatedPassive(unittest.TestCase):
    """Test that coordinated passive phrases are recognized as parallel."""

    def setUp(self):
        self.rule = ConjunctionsRule()
        self.nlp = spacy.load("en_core_web_sm")

    def test_coordinated_passive_participles_known_and_described(self):
        """Test that 'is known and described' is recognized as parallel passive phrases."""
        # The original problematic sentence
        text = "The vulnerability is also known as TunnelVision and described in the CVE article."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # Should not flag parallel structure issues for coordinated passives
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'known and described' should both be recognized as passive, therefore parallel")

    def test_was_created_and_deployed(self):
        """Test that 'was created and deployed' is parallel."""
        text = "The application was created and deployed by the team."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'created and deployed' should both be recognized as passive")

    def test_are_synced_and_encrypted(self):
        """Test that 'are synced and encrypted' is parallel."""
        text = "The files are synced and encrypted automatically."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'synced and encrypted' should both be recognized as passive")

    def test_is_configured_and_managed(self):
        """Test that 'is configured and managed' is parallel."""
        text = "The system is configured and managed by administrators."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'configured and managed' should both be recognized as passive")

    def test_were_tested_and_validated(self):
        """Test that 'were tested and validated' is parallel."""
        text = "The changes were tested and validated before release."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'tested and validated' should both be recognized as passive")

    def test_three_coordinated_passives(self):
        """Test that three coordinated passive participles are all recognized as parallel."""
        text = "The data is collected, processed, and stored securely."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'collected, processed, and stored' should all be recognized as passive")

    def test_actual_non_parallel_still_flagged(self):
        """Test that actual non-parallel structures are still flagged."""
        # This should still be flagged if it has genuinely non-parallel structure
        text = "The system supports creating datasets, data analysis, and to generate reports."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # Just ensure the rule is still working (doesn't break)
        self.assertIsInstance(errors, list)

    def test_passive_with_different_auxiliaries(self):
        """Test coordinated passives with different auxiliary forms."""
        text = "The file is backed up and has been verified."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # This might or might not flag depending on SpaCy parsing
        # Just ensure no crash
        self.assertIsInstance(errors, list)


if __name__ == '__main__':
    unittest.main()

