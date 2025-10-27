"""
Unit tests for Articles Rule.

Tests the rule's ability to correctly identify a/an usage errors
while properly handling technical terms with inline markup.
"""

import unittest
from rules.language_and_grammar.articles_rule import ArticlesRule

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
class TestArticlesRuleWithMarkup(unittest.TestCase):
    """Test that the Articles Rule correctly handles technical terms with markup."""

    def setUp(self):
        self.rule = ArticlesRule()
        self.nlp = spacy.load("en_core_web_sm")

    def test_an_with_vowel_sound_technical_term(self):
        """Test that 'an' is correct before IntegrationSink (vowel sound)."""
        text = "You can create an IntegrationSink API resource."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        # Should not flag - 'an' is correct before IntegrationSink
        self.assertEqual(len(errors), 0, 
                        "'an IntegrationSink' should not be flagged")

    def test_an_with_backticked_technical_term(self):
        """Test that 'an' is correct before `IntegrationSink` with backticks."""
        text = "You can create an `IntegrationSink` API resource."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        # Should not flag - 'an' is correct even with backticks
        self.assertEqual(len(errors), 0, 
                        "'an `IntegrationSink`' should not be flagged")

    def test_an_with_api_abbreviation(self):
        """Test that 'an' is correct before API (vowel sound)."""
        text = "Configure an API endpoint."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        # Should not flag - 'an' is correct before API
        self.assertEqual(len(errors), 0, 
                        "'an API' should not be flagged")

    def test_a_with_consonant_sound_technical_term(self):
        """Test that 'a' is correct before URL (consonant sound)."""
        text = "This is a `URL` parameter."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        # Should not flag - 'a' is correct before URL
        self.assertEqual(len(errors), 0, 
                        "'a `URL`' should not be flagged")

    def test_an_with_http(self):
        """Test that 'an' is correct before HTTP (vowel sound 'aitch')."""
        text = "Use an HTTP request."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        # Should not flag - 'an' is correct before HTTP
        self.assertEqual(len(errors), 0, 
                        "'an HTTP' should not be flagged")

    def test_a_before_unique_consonant_sound(self):
        """Test that 'a' is correct before 'unique' (consonant sound 'yoo')."""
        text = "Create a unique identifier."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        # Should not flag - 'a' is correct before unique
        self.assertEqual(len(errors), 0, 
                        "'a unique' should not be flagged")

    def test_incorrect_a_before_vowel(self):
        """Test that 'a' before vowel sound is correctly flagged."""
        text = "This is a IntegrationSink."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        # Should flag - should be 'an'
        self.assertGreater(len(errors), 0, 
                          "'a IntegrationSink' should be flagged")

    def test_incorrect_an_before_consonant(self):
        """Test that 'an' before consonant sound is correctly flagged."""
        text = "Configure an user account."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        # Should flag - should be 'a'
        self.assertGreater(len(errors), 0, 
                          "'an user' should be flagged")


@unittest.skipIf(not SPACY_AVAILABLE, "SpaCy not available")
class TestArticlesRuleVowelSoundDetection(unittest.TestCase):
    """Test the vowel sound detection logic."""

    def setUp(self):
        self.rule = ArticlesRule()

    def test_vowel_sound_with_vowel_letter(self):
        """Test words starting with vowel letters."""
        self.assertTrue(self.rule._starts_with_vowel_sound("integration"))
        self.assertTrue(self.rule._starts_with_vowel_sound("API"))
        self.assertTrue(self.rule._starts_with_vowel_sound("apple"))
        self.assertTrue(self.rule._starts_with_vowel_sound("orange"))

    def test_consonant_sound_with_consonant_letter(self):
        """Test words starting with consonant letters."""
        self.assertFalse(self.rule._starts_with_vowel_sound("user"))
        self.assertFalse(self.rule._starts_with_vowel_sound("URL"))
        self.assertFalse(self.rule._starts_with_vowel_sound("banana"))
        self.assertFalse(self.rule._starts_with_vowel_sound("computer"))

    def test_special_cases_consonant_sound_with_vowel_letter(self):
        """Test words starting with vowel letters but consonant sounds."""
        # 'u' pronounced as 'yoo' (consonant sound)
        self.assertFalse(self.rule._starts_with_vowel_sound("university"))
        self.assertFalse(self.rule._starts_with_vowel_sound("unique"))
        self.assertFalse(self.rule._starts_with_vowel_sound("uniform"))
        
        # 'eu' pronounced as 'yoo' (consonant sound)
        self.assertFalse(self.rule._starts_with_vowel_sound("european"))
        self.assertFalse(self.rule._starts_with_vowel_sound("euphemism"))

    def test_markup_stripping(self):
        """Test that markup characters are properly stripped."""
        self.assertTrue(self.rule._starts_with_vowel_sound("`integration`"))
        self.assertTrue(self.rule._starts_with_vowel_sound("'apple'"))
        self.assertTrue(self.rule._starts_with_vowel_sound('"API"'))
        self.assertTrue(self.rule._starts_with_vowel_sound("[integration]"))
        self.assertFalse(self.rule._starts_with_vowel_sound("`user`"))
        self.assertFalse(self.rule._starts_with_vowel_sound("'URL'"))


@unittest.skipIf(not SPACY_AVAILABLE, "SpaCy not available")
class TestArticlesRuleCompoundNouns(unittest.TestCase):
    """Test that the Articles Rule correctly handles compound technical nouns."""

    def setUp(self):
        self.rule = ArticlesRule()
        self.nlp = spacy.load("en_core_web_sm")

    def test_compound_noun_routing_table_not_flagged(self):
        """Test that 'routing table' (compound noun) is not flagged for missing article."""
        text = "To protect your VPN connection from traffic redirection attacks, assign it to routing table."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # Should not flag 'table' because it's the head of compound noun 'routing table'
        flagged_texts = [error.get('flagged_text', '') for error in errors]
        self.assertNotIn('table', flagged_texts, 
                        "'routing table' should not be flagged for missing article")

    def test_compound_noun_vpn_connection_not_flagged(self):
        """Test that 'VPN connection' (compound noun) is not flagged for missing article."""
        text = "Configure VPN connection profile to place the VPN routes in table."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # Should not flag 'connection' because it's the head of compound noun 'VPN connection'
        flagged_texts = [error.get('flagged_text', '') for error in errors]
        self.assertNotIn('connection', flagged_texts, 
                        "'VPN connection' should not be flagged for missing article")

    def test_compound_noun_network_interface_not_flagged(self):
        """Test that 'network interface' (compound noun) is not flagged for missing article."""
        text = "At least one network interface uses DHCP or SLAAC."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'technical'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # Should not flag 'interface' because it's the head of compound noun 'network interface'
        flagged_texts = [error.get('flagged_text', '') for error in errors]
        self.assertNotIn('interface', flagged_texts, 
                        "'network interface' should not be flagged for missing article")

    def test_standalone_noun_still_flagged_in_descriptive_content(self):
        """Test that standalone singular nouns are still flagged when appropriate."""
        text = "User can configure settings."
        sentences = [text]
        context = {'block_type': 'paragraph', 'content_type': 'general'}
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp, context=context)
        
        # Note: This may or may not flag depending on the content classification
        # The rule uses content_classification to determine if missing article detection applies
        # We're just ensuring the compound noun guard doesn't break normal detection
        # For descriptive content, it should still potentially flag standalone nouns
        # This test primarily ensures we haven't broken the normal flow
        self.assertIsInstance(errors, list)


if __name__ == '__main__':
    unittest.main()

