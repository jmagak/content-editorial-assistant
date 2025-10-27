#!/usr/bin/env python
"""
Pre-compute taxonomy category embeddings at Docker build time.

This script generates and caches all category embeddings so they're available
immediately when the app starts on OpenShift, reducing first-request latency.
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def precompute_embeddings():
    """Pre-compute all taxonomy category embeddings."""
    try:
        # Import required modules
        from metadata_assistant.taxonomy_classifier import NextGenTaxonomyClassifier
        from metadata_assistant.config import MetadataConfig
        
        logger.info("Starting taxonomy embedding pre-computation...")
        
        # Initialize config to load taxonomy
        config = MetadataConfig()
        
        # Initialize classifier (will load sentence transformer)
        classifier = NextGenTaxonomyClassifier()
        
        if not classifier.sentence_transformer:
            logger.warning("Sentence transformer not available - skipping pre-computation")
            return False
        
        # Get taxonomy configuration
        taxonomy_config = config.taxonomy_config
        
        if not taxonomy_config:
            logger.warning("No taxonomy configuration found - skipping pre-computation")
            return False
        
        logger.info(f"Pre-computing embeddings for {len(taxonomy_config)} categories...")
        
        # Pre-compute embeddings for all categories
        for category, category_config in taxonomy_config.items():
            try:
                embedding = classifier._create_category_embedding(category_config)
                classifier.category_embeddings[category] = embedding
                logger.info(f"  ✓ Computed embedding for: {category}")
            except Exception as e:
                logger.error(f"  ✗ Failed to compute embedding for {category}: {e}")
        
        # Save all embeddings to disk
        if classifier.category_embeddings:
            classifier._save_all_embeddings_to_cache()
            logger.info(f"✅ Successfully pre-computed and cached {len(classifier.category_embeddings)} category embeddings")
            return True
        else:
            logger.warning("No embeddings were computed")
            return False
            
    except Exception as e:
        logger.error(f"Pre-computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = precompute_embeddings()
    sys.exit(0 if success else 1)

