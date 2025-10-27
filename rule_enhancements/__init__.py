"""
Rule Enhancement Layer

This package provides enterprise-grade enhancements to the rule system:

1. NLP Correction Layer - Fixes SpaCy parsing errors
2. Evidence Orchestrator - Manages complex evidence interactions
3. Enterprise Integration - Drop-in adapter for existing rules

Usage:
    from enterprise import enhance_doc, calculate_evidence
    
    corrections = enhance_doc(doc, 'verbs')
    final_score, should_flag, reasoning = calculate_evidence(
        base_score=0.6,
        factors=[
            (0.2, 'HIGH', 'SpaCy correction applied'),
            (-0.1, 'LOW', 'Technical context')
        ],
        rule_type='verbs_sv_agreement',
        context={}
    )
"""

from .nlp_correction_layer import get_correction_layer, NLPCorrectionLayer
from .evidence_orchestrator import get_orchestrator, EvidenceOrchestrator, EvidencePriority, EvidenceFactor
from .enterprise_integration import (
    get_adapter, 
    EnterpriseRuleAdapter,
    enhance_doc,
    get_token_info,
    calculate_evidence
)

__all__ = [
    # Main interfaces
    'enhance_doc',
    'get_token_info',
    'calculate_evidence',
    
    # Classes for advanced usage
    'NLPCorrectionLayer',
    'EvidenceOrchestrator',
    'EnterpriseRuleAdapter',
    'EvidencePriority',
    'EvidenceFactor',
    
    # Singletons
    'get_correction_layer',
    'get_orchestrator',
    'get_adapter',
]

__version__ = '1.0.0'

