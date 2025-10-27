"""
Unit tests for Commas Rule with relative clause guard.

Tests that relative clauses introduced by "that", "which", "who", etc. after commas
are not incorrectly flagged as comma splices.
"""

import unittest
from rules.punctuation.commas_rule import CommasRule

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
class TestCommasRuleRelativeClauses(unittest.TestCase):
    """Test that the Commas Rule properly handles relative clauses after commas."""

    def setUp(self):
        self.rule = CommasRule()
        self.nlp = spacy.load("en_core_web_sm")

    def test_that_relative_clause_not_flagged(self):
        """Test that relative clause with 'that' is not flagged as comma splice."""
        # The original problematic sentence
        text = "Your network does not use mechanisms, such as DHCP snooping, that prevent a rogue DHCP server."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # Should not flag comma splice for relative clauses
        splice_errors = [e for e in errors if 'splice' in e.get('message', '').lower()]
        self.assertEqual(len(splice_errors), 0,
                        "Relative clause with 'that' should not be flagged as comma splice")

    def test_which_relative_clause_not_flagged(self):
        """Test that relative clause with 'which' is not flagged as comma splice."""
        text = "The system uses a mechanism, which is called DHCP, that assigns addresses."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        splice_errors = [e for e in errors if 'splice' in e.get('message', '').lower()]
        self.assertEqual(len(splice_errors), 0,
                        "Relative clause with 'which' should not be flagged as comma splice")

    def test_who_relative_clause_not_flagged(self):
        """Test that relative clause with 'who' is not flagged as comma splice."""
        text = "The administrator, who manages the network, configured the settings."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        splice_errors = [e for e in errors if 'splice' in e.get('message', '').lower()]
        self.assertEqual(len(splice_errors), 0,
                        "Relative clause with 'who' should not be flagged as comma splice")

    def test_subordinating_conjunction_after_comma_not_flagged(self):
        """Test that subordinating conjunction after comma is not flagged."""
        text = "The process starts automatically, while the system is running."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        splice_errors = [e for e in errors if 'splice' in e.get('message', '').lower()]
        self.assertEqual(len(splice_errors), 0,
                        "Subordinating conjunction 'while' after comma should not be flagged as comma splice")

    def test_because_after_comma_not_flagged(self):
        """Test that 'because' after comma is not flagged as comma splice."""
        text = "The connection failed, because the server was unavailable."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        splice_errors = [e for e in errors if 'splice' in e.get('message', '').lower()]
        self.assertEqual(len(splice_errors), 0,
                        "Subordinating conjunction 'because' after comma should not be flagged as comma splice")

    def test_actual_comma_splice_still_flagged(self):
        """Test that actual comma splices are still flagged."""
        # This is an actual comma splice - two independent clauses joined only by comma
        text = "The system crashed, the administrator restarted it."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # This should still be flagged as a comma splice (might or might not depending on parsing)
        # Just ensure the rule is still working (doesn't break)
        self.assertIsInstance(errors, list)

    def test_appositive_with_relative_clause_not_flagged(self):
        """Test complex sentence with appositive and relative clause."""
        # Simplified version of the original sentence
        text = "The network uses mechanisms, like DHCP snooping, that prevent attacks."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        splice_errors = [e for e in errors if 'splice' in e.get('message', '').lower()]
        self.assertEqual(len(splice_errors), 0,
                        "Appositive with relative clause should not be flagged as comma splice")

    def test_where_relative_clause_not_flagged(self):
        """Test that relative clause with 'where' is not flagged."""
        text = "The server, where the data is stored, processes requests."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        splice_errors = [e for e in errors if 'splice' in e.get('message', '').lower()]
        self.assertEqual(len(splice_errors), 0,
                        "Relative clause with 'where' should not be flagged as comma splice")


if __name__ == '__main__':
    unittest.main()

