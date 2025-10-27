"""
Unit tests for Product Names Rule.

Tests the rule's ability to detect IBM products that need IBM prefix
while correctly ignoring competitor products from AWS, Microsoft, Google, etc.
"""

import unittest
from rules.references.product_names_rule import ProductNamesRule

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
class TestProductNamesRuleCompetitorProducts(unittest.TestCase):
    """Test that competitor products are not flagged."""

    def setUp(self):
        self.rule = ProductNamesRule()
        self.nlp = spacy.load("en_core_web_sm")

    def test_aws_simple_notification_service_not_flagged(self):
        """Test that Amazon Simple Notification Service is not flagged."""
        text = "You can publish CloudEvents to an Amazon Simple Notification Service (SNS) topic."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        # Should not flag AWS products
        self.assertEqual(len(errors), 0, 
                        "Amazon Simple Notification Service should not be flagged")

    def test_aws_abbreviations_not_flagged(self):
        """Test that AWS service abbreviations are not flagged."""
        text = "Configure the SNS topic for notifications."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        self.assertEqual(len(errors), 0, 
                        "SNS (AWS abbreviation) should not be flagged")

    def test_aws_lambda_not_flagged(self):
        """Test that AWS Lambda is not flagged."""
        text = "The AWS Lambda function processes events from S3."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        self.assertEqual(len(errors), 0, 
                        "AWS Lambda and S3 should not be flagged")

    def test_azure_not_flagged(self):
        """Test that Microsoft Azure products are not flagged."""
        text = "You can use Azure DevOps for CI/CD."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        self.assertEqual(len(errors), 0, 
                        "Azure DevOps should not be flagged")

    def test_google_cloud_not_flagged(self):
        """Test that Google Cloud products are not flagged."""
        text = "The Google Cloud Platform provides many services."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        self.assertEqual(len(errors), 0, 
                        "Google Cloud Platform should not be flagged")

    def test_ibm_watson_without_prefix_is_flagged(self):
        """Test that IBM products without IBM prefix ARE flagged."""
        text = "Watson can help you analyze the data."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        # Should flag Watson without IBM prefix
        self.assertGreater(len(errors), 0, 
                          "Watson without IBM prefix should be flagged")
        
        # Check that the error is about Watson
        watson_errors = [e for e in errors if 'watson' in e.get('flagged_text', '').lower()]
        self.assertGreater(len(watson_errors), 0, 
                          "Error should specifically mention Watson")

    def test_ibm_watson_with_prefix_not_flagged(self):
        """Test that IBM products with IBM prefix are not flagged."""
        text = "IBM Watson can help you analyze the data."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        # Should not flag Watson with IBM prefix
        watson_errors = [e for e in errors if 'watson' in e.get('flagged_text', '').lower()]
        self.assertEqual(len(watson_errors), 0, 
                        "IBM Watson with prefix should not be flagged")


@unittest.skipIf(not SPACY_AVAILABLE, "SpaCy not available")
class TestProductNamesRuleContextAwareness(unittest.TestCase):
    """Test that the rule is context-aware and understands third-party references."""

    def setUp(self):
        self.rule = ProductNamesRule()
        self.nlp = spacy.load("en_core_web_sm")

    def test_amazon_in_context(self):
        """Test that products are recognized as Amazon products in context."""
        text = "Use the Amazon EC2 service to launch instances."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        self.assertEqual(len(errors), 0, 
                        "Products mentioned with Amazon should not be flagged")

    def test_microsoft_in_context(self):
        """Test that products are recognized as Microsoft products in context."""
        text = "Microsoft Office 365 provides cloud-based productivity tools."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        self.assertEqual(len(errors), 0, 
                        "Products mentioned with Microsoft should not be flagged")

    def test_oracle_in_context(self):
        """Test that products are recognized as Oracle products in context."""
        text = "The application connects to Oracle Database for data storage."
        sentences = [text]
        
        errors = self.rule.analyze(text, sentences, nlp=self.nlp)
        
        # Filter out errors that are not about Oracle products
        oracle_errors = [e for e in errors if 'oracle' in e.get('sentence', '').lower()]
        self.assertEqual(len(oracle_errors), 0, 
                        "Products mentioned with Oracle should not be flagged")


@unittest.skipIf(not SPACY_AVAILABLE, "SpaCy not available")
class TestProductNamesRuleConfigurationBased(unittest.TestCase):
    """Test that the rule uses configuration properly."""

    def setUp(self):
        self.rule = ProductNamesRule()
        self.nlp = spacy.load("en_core_web_sm")

    def test_config_service_loaded(self):
        """Test that configuration service is properly loaded."""
        self.assertIsNotNone(self.rule.config_service)
        self.assertEqual(self.rule.config_service.config_name, 'product_patterns')

    def test_competitor_product_check(self):
        """Test that competitor product check works."""
        # These should be identified as competitor products
        self.assertTrue(self.rule.config_service.is_competitor_product("SNS"))
        self.assertTrue(self.rule.config_service.is_competitor_product("Simple Notification Service"))
        self.assertTrue(self.rule.config_service.is_competitor_product("AWS"))
        self.assertTrue(self.rule.config_service.is_competitor_product("Azure"))
        
    def test_competitor_company_check(self):
        """Test that competitor company check works."""
        # These should be identified as competitor companies
        self.assertTrue(self.rule.config_service.is_competitor_company("Amazon"))
        self.assertTrue(self.rule.config_service.is_competitor_company("Microsoft"))
        self.assertTrue(self.rule.config_service.is_competitor_company("Google"))

    def test_ibm_product_check(self):
        """Test that IBM product check works."""
        # These should be identified as IBM products
        watson = self.rule.config_service.get_ibm_product("watson")
        self.assertIsNotNone(watson)
        self.assertTrue(watson.requires_ibm_prefix)


if __name__ == '__main__':
    unittest.main()

