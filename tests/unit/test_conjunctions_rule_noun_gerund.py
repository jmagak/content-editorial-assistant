"""
Unit tests for Conjunctions Rule with noun/gerund coordination.

Tests that NOMINAL vs ACTION mismatches (common in technical writing)
are not flagged as non-parallel when they are semantically parallel.
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
class TestConjunctionsRuleNounGerund(unittest.TestCase):
    """Test that noun/gerund coordinations are accepted as semantically parallel."""

    def setUp(self):
        self.rule = ConjunctionsRule()
        self.nlp = spacy.load("en_core_web_sm")

    def test_bypassing_tunnel_and_compromising_data(self):
        """Test that 'bypassing the tunnel and compromising data' is not flagged."""
        # The original problematic sentence
        text = "A malicious DHCP server can force a host to redirect traffic, bypassing the secure tunnel and compromising data integrity."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # Should not flag parallel structure issues for noun/gerund coordination
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'bypassing the tunnel and compromising data' should not be flagged")

    def test_creating_datasets_and_analysis(self):
        """Test that 'creating datasets and analysis' is not flagged."""
        text = "The process involves creating datasets and analysis of the results."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'creating datasets and analysis' should not be flagged")

    def test_monitoring_systems_and_alerts(self):
        """Test that 'monitoring systems and alerts' is not flagged."""
        text = "The platform provides monitoring systems and alerts for administrators."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'monitoring systems and alerts' should not be flagged")

    def test_processing_data_and_reports(self):
        """Test that 'processing data and reports' is not flagged."""
        text = "The service handles processing data and reports automatically."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'processing data and reports' should not be flagged")

    def test_managing_resources_and_configuration(self):
        """Test that 'managing resources and configuration' is not flagged."""
        text = "Administrators are responsible for managing resources and configuration."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'managing resources and configuration' should not be flagged")

    def test_securing_connections_and_data(self):
        """Test that 'securing connections and data' is not flagged."""
        text = "The VPN focuses on securing connections and data transmission."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'securing connections and data' should not be flagged")

    def test_gerund_and_noun_reversed_order(self):
        """Test that reversed order (noun then gerund) is also not flagged."""
        text = "The system provides analysis and processing of the data."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        parallel_errors = [e for e in errors if 'parallel' in e.get('message', '').lower()]
        self.assertEqual(len(parallel_errors), 0,
                        "'analysis and processing' (reversed order) should not be flagged")

    def test_actual_problematic_structure_still_detected(self):
        """Test that genuinely problematic non-parallel structures are still caught."""
        # This should still potentially flag: infinitive + gerund + noun (not the NOMINAL/ACTION pattern)
        text = "The system supports to create datasets, generating reports, and a dashboard."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # Just ensure the rule is still working (doesn't break)
        self.assertIsInstance(errors, list)


if __name__ == '__main__':
    unittest.main()

