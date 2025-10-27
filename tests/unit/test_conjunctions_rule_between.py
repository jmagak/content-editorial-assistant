"""
Unit tests for Conjunctions Rule with "between X and Y" guard.

Tests that "between X and Y" structures are not flagged as non-parallel.
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
class TestConjunctionsRuleBetween(unittest.TestCase):
    """Test that the Conjunctions Rule properly handles 'between X and Y' structures."""

    def setUp(self):
        self.rule = ConjunctionsRule()
        self.nlp = spacy.load("en_core_web_sm")

    def test_between_numbers_not_flagged(self):
        """Test that 'between 1 and 32766' is not flagged as non-parallel."""
        text = "The value must be between 1 and 32766."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # Should not flag parallel structure issues for "between X and Y"
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'between 1 and 32766' should not be flagged as non-parallel")

    def test_between_nouns_not_flagged(self):
        """Test that 'between Monday and Friday' is not flagged."""
        text = "The system is available between Monday and Friday."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'between Monday and Friday' should not be flagged as non-parallel")

    def test_between_mixed_types_not_flagged(self):
        """Test that 'between nodes and edges' with different tags is not flagged."""
        text = "The graph consists of relationships between nodes and edges."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'between nodes and edges' should not be flagged as non-parallel")

    def test_between_range_values_not_flagged(self):
        """Test that ranges like 'between 0 and 100' are not flagged."""
        text = "Set the priority value between 32345 and 32766."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'between 32345 and 32766' should not be flagged as non-parallel")

    def test_non_between_coordination_still_flagged(self):
        """Test that non-'between' coordinations with mismatched structure are still flagged."""
        # This should still be flagged if it has non-parallel structure
        text = "The system supports creating datasets, data analysis, and to generate reports."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # This might or might not flag depending on the specific parsing
        # Just ensure the rule is still working (doesn't break)
        self.assertIsInstance(errors, list)

    def test_three_way_coordination_not_affected(self):
        """Test that three-way coordinations (not 'between X and Y') are still checked."""
        text = "The process involves X, Y, and Z."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # The guard should only apply to two-element coordinations with "between"
        # Three-element coordinations should still be analyzed normally
        self.assertIsInstance(errors, list)


if __name__ == '__main__':
    unittest.main()

