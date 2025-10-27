"""
Enterprise NLP Correction Layer
Addresses SpaCy parsing limitations with fallback strategies and corrections.
"""

from typing import Dict, Any, Optional
import re


class NLPCorrectionLayer:
    """
    Enterprise-grade correction layer for SpaCy parsing errors.
    Uses domain knowledge and linguistic patterns to correct common NLP mistakes.
    """
    
    def __init__(self):
        # Known nouns that SpaCy often mislabels as adjectives
        self.noun_corrections = {
            # Time-related compounds
            'uptime', 'downtime', 'runtime', 'timestamp', 'lifetime', 'timeout',
            # System/network compounds  
            'username', 'hostname', 'filename', 'pathname', 'database', 'namespace',
            'filesystem', 'workflow', 'endpoint', 'payload', 'throughput', 'bandwidth',
            'firewall', 'gateway', 'subnet', 'localhost', 'wildcard', 'passphrase',
            # Process/operation compounds
            'checkout', 'setup', 'startup', 'shutdown', 'rollback', 'failover',
            'handshake', 'handoff', 'lookup', 'callback', 'runbook', 'playbook'
        }
        
        # Technical proper nouns that should be treated as common nouns for agreement
        # These end in 's' and are plural despite being tagged as singular proper nouns
        self.proper_noun_plural_markers = {
            # TCP/Network
            'timestamps', 'packets', 'segments', 'frames', 'routes',
            # Web/API
            'urls', 'apis', 'endpoints', 'webhooks', 'cookies',
            # Computing
            'ids', 'cpus', 'gpus', 'nics', 'cores', 'threads', 'processes',
            # Storage
            'bytes', 'bits', 'blocks', 'sectors', 'partitions',
            # Cloud/Container
            'pods', 'nodes', 'clusters', 'instances', 'containers'
        }
        
        # Known two-letter technical abbreviations
        self.technical_two_letter_abbrevs = {
            # Networking
            'TX', 'RX', 'IP', 'IO', 'OS', 'VM', 'UI', 'ID', 'CD', 'CI',
            'QA', 'HA', 'DR', 'VR', 'AR', 'AI', 'ML', 'DB', 'FS', 'NW'
        }
    
    def correct_pos_tag(self, token_text: str, pos: str, tag: str) -> tuple:
        """
        Correct POS tag based on domain knowledge.
        
        Args:
            token_text: The token text
            pos: Current POS tag
            tag: Current fine-grained tag
            
        Returns:
            Tuple of (corrected_pos, corrected_tag)
        """
        text_lower = token_text.lower()
        
        # Correction 1: Known nouns mislabeled as adjectives
        if pos == 'ADJ' and text_lower in self.noun_corrections:
            return ('NOUN', 'NN')
        
        return (pos, tag)
    
    def correct_number_agreement(self, token_text: str, morphology: str) -> str:
        """
        Correct number agreement for technical terms.
        
        Args:
            token_text: The token text
            morphology: Current morphological features
            
        Returns:
            Corrected morphology string or None if no correction needed
        """
        text_lower = token_text.lower()
        
        # Correction: Proper nouns ending in 's' are plural
        if text_lower in self.proper_noun_plural_markers:
            if 'Number=Sing' in morphology:
                # Correct to plural
                return morphology.replace('Number=Sing', 'Number=Plur')
        
        return morphology
    
    def is_technical_abbreviation(self, token_text: str, entity_type: str) -> bool:
        """
        Determine if a token is a technical abbreviation despite entity labeling.
        
        Args:
            token_text: The token text
            entity_type: SpaCy entity type
            
        Returns:
            True if this should be treated as abbreviation
        """
        # If SpaCy labeled as PROPN but it's a known technical abbreviation
        if token_text in self.technical_two_letter_abbrevs:
            return True
        
        # Pattern: 2-4 uppercase letters that are likely abbreviations
        # Enhanced to catch MAC, EAPOL, etc.
        if (2 <= len(token_text) <= 4 and 
            token_text.isupper() and 
            token_text.isalpha()):
            # Only exclude if it's clearly a person/place name
            if entity_type in ['PERSON', 'GPE']:
                return False
            # ORG entities might be abbreviations (e.g., IBM, NASA), so include them
            return True
        
        return False
    
    def apply_corrections_to_doc(self, doc):
        """
        Apply corrections to a SpaCy doc in-place where possible.
        Returns a correction map for tokens that can't be modified in-place.
        
        Args:
            doc: SpaCy doc
            
        Returns:
            Dict mapping token index to correction data
        """
        corrections = {}
        
        for i, token in enumerate(doc):
            correction_data = {}
            
            # Check POS corrections
            corrected_pos, corrected_tag = self.correct_pos_tag(
                token.text, token.pos_, token.tag_
            )
            if corrected_pos != token.pos_ or corrected_tag != token.tag_:
                correction_data['pos'] = corrected_pos
                correction_data['tag'] = corrected_tag
            
            # Check morphology corrections
            corrected_morph = self.correct_number_agreement(
                token.text, str(token.morph)
            )
            if corrected_morph != str(token.morph):
                correction_data['morph'] = corrected_morph
            
            # Check abbreviation status
            if self.is_technical_abbreviation(token.text, token.ent_type_):
                correction_data['is_technical_abbrev'] = True
            
            if correction_data:
                corrections[i] = correction_data
        
        return corrections


# Global singleton instance
_correction_layer = None

def get_correction_layer() -> NLPCorrectionLayer:
    """Get global correction layer instance."""
    global _correction_layer
    if _correction_layer is None:
        _correction_layer = NLPCorrectionLayer()
    return _correction_layer

