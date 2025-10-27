"""
Enterprise Integration Layer
Integrates NLP corrections and evidence orchestration into existing rules.
"""

from typing import Any, Dict, List, Optional
from .nlp_correction_layer import get_correction_layer
from .evidence_orchestrator import get_orchestrator, EvidencePriority


class EnterpriseRuleAdapter:
    """
    Adapter that wraps existing rules with enterprise capabilities.
    Drop-in enhancement that doesn't require rule rewrites.
    """
    
    def __init__(self):
        self.correction_layer = get_correction_layer()
        self.orchestrator = get_orchestrator()
        self.stats = {
            'corrections_applied': 0,
            'conflicts_resolved': 0,
            'thresholds_adjusted': 0
        }
    
    def enhance_doc_analysis(self, doc, rule_type: str):
        """
        Enhance SpaCy doc with corrections before rule analysis.
        
        Args:
            doc: SpaCy doc
            rule_type: Type of rule analyzing this doc
            
        Returns:
            Corrections map
        """
        corrections = self.correction_layer.apply_corrections_to_doc(doc)
        
        if corrections:
            self.stats['corrections_applied'] += len(corrections)
        
        return corrections
    
    def get_corrected_token_info(self, token, corrections: Dict) -> Dict[str, Any]:
        """
        Get corrected token information.
        
        Args:
            token: SpaCy token
            corrections: Corrections map from enhance_doc_analysis
            
        Returns:
            Dict with corrected attributes
        """
        info = {
            'text': token.text,
            'pos_': token.pos_,
            'tag_': token.tag_,
            'morph': str(token.morph),
            'is_technical_abbrev': False
        }
        
        if token.i in corrections:
            correction = corrections[token.i]
            info['pos_'] = correction.get('pos', token.pos_)
            info['tag_'] = correction.get('tag', token.tag_)
            info['morph'] = correction.get('morph', str(token.morph))
            info['is_technical_abbrev'] = correction.get('is_technical_abbrev', False)
            info['was_corrected'] = True
        
        return info
    
    def is_plural_corrected(self, token, corrections: Dict) -> bool:
        """
        Check if token is plural after corrections.
        
        Args:
            token: SpaCy token
            corrections: Corrections map
            
        Returns:
            True if plural (after corrections)
        """
        info = self.get_corrected_token_info(token, corrections)
        
        # Check corrected morphology
        if 'Number=Plur' in info['morph']:
            return True
        
        # Check corrected tag
        if info['tag_'] in ['NNS', 'NNPS']:
            return True
        
        return False
    
    def should_treat_as_abbreviation(self, token, corrections: Dict) -> bool:
        """
        Determine if token should be treated as abbreviation after corrections.
        
        Args:
            token: SpaCy token  
            corrections: Corrections map
            
        Returns:
            True if should be treated as abbreviation
        """
        info = self.get_corrected_token_info(token, corrections)
        return info.get('is_technical_abbrev', False)
    
    def calculate_final_evidence(
        self,
        base_evidence: float,
        additional_factors: List[tuple],  # [(value, priority, reason)]
        rule_type: str,
        context: Dict[str, Any]
    ) -> tuple[float, bool, List[str]]:
        """
        Calculate final evidence score with conflict resolution.
        
        Args:
            base_evidence: Initial evidence score
            additional_factors: List of (value, priority_name, reason) tuples
            rule_type: Rule type
            context: Additional context
            
        Returns:
            Tuple of (final_score, should_flag, reasoning_chain)
        """
        from .evidence_orchestrator import EvidenceFactor
        
        # Create base factor
        factors = [
            EvidenceFactor(
                name='base',
                value=base_evidence,
                priority=EvidencePriority.MEDIUM,
                reason='Initial rule evidence',
                rule_type=rule_type
            )
        ]
        
        # Add additional factors
        for value, priority_name, reason in additional_factors:
            priority = getattr(EvidencePriority, priority_name.upper())
            factors.append(
                EvidenceFactor(
                    name='additional',
                    value=value,
                    priority=priority,
                    reason=reason,
                    rule_type=rule_type
                )
            )
        
        # Resolve conflicts
        final_score, reasoning = self.orchestrator.resolve_evidence_conflicts(factors)
        self.stats['conflicts_resolved'] += 1
        
        # Get adjusted threshold
        threshold = self.orchestrator.get_adjusted_threshold(rule_type, context)
        self.stats['thresholds_adjusted'] += 1
        
        should_flag = final_score >= threshold
        reasoning.append(f"THRESHOLD: {threshold:.3f}")
        reasoning.append(f"DECISION: {'FLAG' if should_flag else 'SKIP'}")
        
        return final_score, should_flag, reasoning
    
    def get_stats(self) -> Dict[str, int]:
        """Get enterprise system statistics."""
        return self.stats.copy()


# Global singleton
_adapter = None

def get_adapter() -> EnterpriseRuleAdapter:
    """Get global adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = EnterpriseRuleAdapter()
    return _adapter


# Convenience functions for easy integration

def enhance_doc(doc, rule_type: str = 'general'):
    """Quick access to doc enhancement."""
    return get_adapter().enhance_doc_analysis(doc, rule_type)


def get_token_info(token, corrections: Dict):
    """Quick access to corrected token info."""
    return get_adapter().get_corrected_token_info(token, corrections)


def calculate_evidence(base_score: float, factors: List, rule_type: str, context: Dict = None):
    """Quick access to evidence calculation."""
    return get_adapter().calculate_final_evidence(
        base_score, 
        factors, 
        rule_type, 
        context or {}
    )

