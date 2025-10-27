"""
Enterprise Evidence Orchestration System
Manages complex interactions between guards, evidence factors, and thresholds.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class EvidencePriority(Enum):
    """Priority levels for evidence factors."""
    CRITICAL = 100  # Zero-false-positive guards
    HIGH = 75       # Strong linguistic anchors
    MEDIUM = 50     # Standard evidence factors
    LOW = 25        # Contextual hints


@dataclass
class EvidenceFactor:
    """Represents a single evidence factor with priority."""
    name: str
    value: float  # Positive = evidence for error, negative = evidence against
    priority: EvidencePriority
    reason: str
    rule_type: str  # e.g., 'conjunctions', 'verbs', 'articles'


class EvidenceOrchestrator:
    """
    Enterprise-grade evidence orchestration system.
    Handles conflicts, priorities, and threshold calibration.
    """
    
    def __init__(self):
        # Rule-specific threshold configurations
        self.threshold_config = {
            'articles_complete_sentence_list': {
                'base_threshold': 0.35,
                'complete_sentence_boost': 0.4,  # Increased from 0.3
                'fragment_penalty': -0.3
            },
            'abbreviations_two_letter': {
                'base_threshold': 0.35,
                'technical_context_boost': 0.2
            },
            'verbs_sv_agreement': {
                'base_threshold': 0.5,
                'plural_proper_noun_correction': -0.4  # Override for proper nouns
            },
            'conjunctions_parallel': {
                'base_threshold': 0.5,
                'pos_mismatch_correction': -0.3  # Reduce when SpaCy mislabels
            }
        }
    
    def resolve_evidence_conflicts(
        self, 
        factors: List[EvidenceFactor]
    ) -> tuple[float, List[str]]:
        """
        Resolve conflicts between evidence factors using priority system.
        
        Enterprise Strategy:
        1. CRITICAL factors (guards) override everything
        2. HIGH priority factors have 2x weight
        3. Conflicting factors at same priority are averaged
        4. Final score is normalized
        
        Args:
            factors: List of evidence factors
            
        Returns:
            Tuple of (final_evidence_score, reasoning_chain)
        """
        if not factors:
            return 0.0, ["No evidence factors provided"]
        
        reasoning = []
        
        # Separate by priority
        critical_factors = [f for f in factors if f.priority == EvidencePriority.CRITICAL]
        high_factors = [f for f in factors if f.priority == EvidencePriority.HIGH]
        medium_factors = [f for f in factors if f.priority == EvidencePriority.MEDIUM]
        low_factors = [f for f in factors if f.priority == EvidencePriority.LOW]
        
        # Step 1: CRITICAL factors can override everything
        critical_negative = [f for f in critical_factors if f.value < -0.5]
        if critical_negative:
            reasoning.append(f"CRITICAL GUARD: {critical_negative[0].reason}")
            return 0.0, reasoning
        
        # Step 2: Calculate weighted evidence
        evidence_score = 0.0
        total_weight = 0.0
        
        # CRITICAL: weight = 4x
        for factor in critical_factors:
            weight = 4.0
            evidence_score += factor.value * weight
            total_weight += weight
            reasoning.append(f"CRITICAL ({factor.value:+.2f}x4): {factor.reason}")
        
        # HIGH: weight = 2x
        for factor in high_factors:
            weight = 2.0
            evidence_score += factor.value * weight
            total_weight += weight
            reasoning.append(f"HIGH ({factor.value:+.2f}x2): {factor.reason}")
        
        # MEDIUM: weight = 1x
        for factor in medium_factors:
            weight = 1.0
            evidence_score += factor.value * weight
            total_weight += weight
            reasoning.append(f"MEDIUM ({factor.value:+.2f}x1): {factor.reason}")
        
        # LOW: weight = 0.5x
        for factor in low_factors:
            weight = 0.5
            evidence_score += factor.value * weight
            total_weight += weight
            reasoning.append(f"LOW ({factor.value:+.2f}x0.5): {factor.reason}")
        
        # Step 3: Normalize by total weight
        if total_weight > 0:
            normalized_score = evidence_score / total_weight
        else:
            normalized_score = 0.0
        
        reasoning.append(f"WEIGHTED AVERAGE: {normalized_score:.3f}")
        
        # Step 4: Apply calibration curve (prevent extreme scores)
        calibrated_score = self._apply_calibration_curve(normalized_score)
        if calibrated_score != normalized_score:
            reasoning.append(f"CALIBRATED: {calibrated_score:.3f}")
        
        return calibrated_score, reasoning
    
    def _apply_calibration_curve(self, score: float) -> float:
        """
        Apply sigmoid-like calibration curve to prevent extreme scores.
        Maps any score to reasonable range while preserving ordering.
        """
        import math
        
        if score >= 0:
            # Positive scores: compress high values
            # 0.5 → 0.5, 1.0 → 0.85, 2.0 → 0.95
            return 1.0 - math.exp(-score * 1.5)
        else:
            # Negative scores: compress low values
            # -0.5 → -0.4, -1.0 → -0.6, -2.0 → -0.8
            return -1.0 + math.exp(score * 1.5)
    
    def get_adjusted_threshold(
        self, 
        rule_context: str, 
        additional_context: Dict[str, Any]
    ) -> float:
        """
        Get calibrated threshold based on rule context.
        
        Args:
            rule_context: Rule type and context (e.g., 'articles_complete_sentence_list')
            additional_context: Additional context data
            
        Returns:
            Adjusted threshold value
        """
        if rule_context not in self.threshold_config:
            return 0.35  # Default universal threshold
        
        config = self.threshold_config[rule_context]
        threshold = config['base_threshold']
        
        # Apply context-specific adjustments
        if rule_context == 'articles_complete_sentence_list':
            if additional_context.get('is_complete_sentence'):
                threshold -= config['complete_sentence_boost']
            else:
                threshold += abs(config['fragment_penalty'])
        
        # Clamp to reasonable range
        return max(0.1, min(0.9, threshold))
    
    def create_evidence_factor(
        self,
        name: str,
        value: float,
        priority: EvidencePriority,
        reason: str,
        rule_type: str
    ) -> EvidenceFactor:
        """Factory method for creating evidence factors."""
        return EvidenceFactor(
            name=name,
            value=value,
            priority=priority,
            reason=reason,
            rule_type=rule_type
        )


# Global singleton
_orchestrator = None

def get_orchestrator() -> EvidenceOrchestrator:
    """Get global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = EvidenceOrchestrator()
    return _orchestrator

