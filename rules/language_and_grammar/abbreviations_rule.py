"""
Abbreviations Rule
Based on IBM Style Guide: Checks Latin abbreviations, undefined abbreviations, and verb usage.
"""
import re
from typing import List, Dict, Any, Set, Optional
from .base_language_rule import BaseLanguageRule
from .services.language_vocabulary_service import get_abbreviations_vocabulary

try:
    from spacy.tokens import Doc, Token
except ImportError:
    Doc = None
    Token = None

try:
    from rule_enhancements import get_adapter
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False

class AbbreviationsRule(BaseLanguageRule):
    """Checks Latin abbreviations, undefined abbreviations, and verb usage."""
    
    def __init__(self):
        super().__init__()
        self.defined_abbreviations: Set[str] = set()
        self.current_document_hash: Optional[str] = None
        self.adapter = get_adapter() if ENHANCEMENTS_AVAILABLE else None
        self.vocab_service = get_abbreviations_vocabulary()
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load abbreviations configuration from vocabulary service."""
        config = self.vocab_service.get_abbreviations_config()
        
        # Flatten universal abbreviations into single set for fast lookup
        universal_abbrevs = set()
        for category in config.get('universal_abbreviations', {}).values():
            universal_abbrevs.update(category)
        config['universal_abbreviations_set'] = universal_abbrevs
        
        return config

    def _get_rule_type(self) -> str:
        return 'abbreviations'

    def _reset_document_state_if_needed(self, context: dict = None) -> None:
        """Reset state for new documents while preserving within same document."""
        document_id = context.get('source_location', '') if context else ''
        if self.current_document_hash != document_id:
            self.defined_abbreviations.clear()
            self.current_document_hash = document_id

    def analyze(self, text: str, sentences: List[str], nlp=None, context=None) -> List[Dict[str, Any]]:
        """Analyze text for abbreviation violations using evidence-based scoring."""
        if context and context.get('block_type') in ['listing', 'literal', 'code_block', 'inline_code']:
            return []
        
        errors = []
        if not nlp:
            return errors

        self._reset_document_state_if_needed(context)
        
        # Pre-process parenthetical definitions: "Full Term (ABBR)"
        definition_pattern = r'\b([A-Za-z][a-z]*(?:[ -][A-Za-z][a-z]*)+)\s+\(([A-Z]{2,})\)'
        for match in re.finditer(definition_pattern, text):
            self.defined_abbreviations.add(match.group(2))
        
        doc = nlp(text)
        corrections = self.adapter.enhance_doc_analysis(doc, self._get_rule_type()) if self.adapter else {}
        
        # Check 1: Latin Abbreviations
        for i, sent in enumerate(doc.sents):
            for token in sent:
                if self._is_latin_abbreviation(token, doc):
                    evidence_score = self._calculate_latin_abbreviation_evidence(token, sent, text, context)
                    if evidence_score > self.config['evidence_thresholds']['min_threshold']:
                        replacement = self._get_latin_equivalent(token.text.lower())
                        errors.append(self._create_error(
                            sentence=sent.text,
                            sentence_index=i,
                            message=self._get_contextual_message(token, evidence_score, 'latin'),
                            suggestions=self._generate_smart_suggestions(token, context, 'latin', replacement),
                            severity='medium',
                            text=text,
                            context=context,
                            evidence_score=evidence_score,
                            span=(token.idx, token.idx + len(token.text)),
                            flagged_text=token.text
                        ))

        # Check 2 & 3: Undefined Abbreviations and Verb Usage
        for token in doc:
            is_candidate = self._is_abbreviation_candidate(token)
            if self.adapter and not is_candidate:
                is_candidate = self.adapter.should_treat_as_abbreviation(token, corrections)
            
            if is_candidate:
                if self._is_definition_pattern(token, doc):
                    self.defined_abbreviations.add(token.text)
                    continue
                
                if token.text not in self.defined_abbreviations:
                    evidence_score = self._calculate_undefined_abbreviation_evidence(token, token.sent, text, context)
                    if evidence_score > self.config['evidence_thresholds']['min_threshold']:
                        sent = token.sent
                        sent_index = list(doc.sents).index(sent)
                        errors.append(self._create_error(
                            sentence=sent.text,
                            sentence_index=sent_index,
                            message=self._get_contextual_message(token, evidence_score, 'undefined'),
                            suggestions=self._generate_smart_suggestions(token, context, 'undefined'),
                            severity='medium',
                            text=text,
                            context=context,
                            evidence_score=evidence_score,
                            span=(token.idx, token.idx + len(token.text)),
                            flagged_text=token.text
                        ))
                    self.defined_abbreviations.add(token.text)
                
                if self._is_used_as_verb(token, doc):
                    evidence_score = self._calculate_verb_usage_evidence(token, token.sent, text, context)
                    if evidence_score > self.config['evidence_thresholds']['min_threshold']:
                        sent = token.sent
                        sent_index = list(doc.sents).index(sent)
                        errors.append(self._create_error(
                            sentence=sent.text,
                            sentence_index=sent_index,
                            message=self._get_contextual_message(token, evidence_score, 'verb'),
                            suggestions=self._generate_smart_suggestions(token, context, 'verb'),
                            severity='medium',
                            text=text,
                            context=context,
                            evidence_score=evidence_score,
                            span=(token.idx, token.idx + len(token.text)),
                            flagged_text=token.text
                        ))
        
        return errors

    def _is_latin_abbreviation(self, token: 'Token', doc: 'Doc') -> bool:
        """Detect Latin abbreviations using morphological patterns."""
        if not token.text:
            return False
            
        text_lower = token.text.lower()
        
        if (len(text_lower) == 3 and text_lower.endswith('.') and 
            text_lower[0].isalpha() and text_lower[1].isalpha()):
            if self._is_in_parenthetical_context(token, doc):
                return True
        
        if (len(text_lower) == 4 and text_lower.count('.') == 2 and
            text_lower[0].isalpha() and text_lower[2].isalpha()):
            return True
            
        if (len(text_lower) >= 3 and text_lower.endswith('.') and
            self._has_latin_morphology(text_lower)):
            return True
            
        return False

    def _get_latin_equivalent(self, latin_term: str) -> str:
        """Get English equivalent for Latin abbreviation from config."""
        base_term = latin_term.rstrip('.')
        return self.config['latin_equivalents'].get(base_term, 'an English equivalent')

    def _is_abbreviation_candidate(self, token: 'Token') -> bool:
        """Identify abbreviation candidates using POS and entity analysis."""
        if (token.is_upper and len(token.text) >= 2 and 
            token.text.isalpha() and not token.is_stop):
            if token.ent_type_ in ['PERSON', 'GPE']:
                return False
            return True
        return False

    def _is_contextually_defined(self, token: 'Token', doc: 'Doc') -> bool:
        """Check if abbreviation is defined using syntactic patterns."""
        if (token.i + 1 < len(doc) and doc[token.i + 1].text == '(' and
            self._has_definition_in_parens(token, doc)):
            return True
        
        if (token.i > 1 and doc[token.i - 1].text == ')' and
            self._is_abbreviation_in_parens(token, doc)):
            return True
            
        if self._has_explicit_definition(token, doc):
            return True
            
        return False

    def _is_used_as_verb(self, token: 'Token', doc: 'Doc') -> bool:
        """Detect verb usage through POS and dependency analysis."""
        if token.pos_ == 'VERB':
            return True
            
        if (token.i > 0 and token.i < len(doc) - 1):
            prev_token = doc[token.i - 1]
            next_token = doc[token.i + 1]
            if (prev_token.tag_ == 'MD' and next_token.pos_ in ['DET', 'NOUN', 'PRON']):
                return True
        
        if token.dep_ in ['ROOT', 'ccomp', 'xcomp'] and self._has_verbal_dependents(token, doc):
            return True
            
        return False

    def _generate_verb_alternative(self, token: 'Token', doc: 'Doc') -> str:
        """Generate context-aware verb alternatives."""
        if token.i > 0:
            prev_token = doc[token.i - 1]
            if prev_token.tag_ == 'MD':
                action_verb = self._get_semantic_action(token.text.lower())
                return f"Rewrite to use a proper verb: '{prev_token.text} {action_verb} {token.text} to...'"
        
        return f"Rewrite the sentence to use a proper verb. For example, instead of '{token.text} the file', write 'Use {token.text} to transfer the file'."

    def _is_in_parenthetical_context(self, token: 'Token', doc: 'Doc') -> bool:
        """Check if token appears in parenthetical context."""
        for i in range(max(0, token.i - 5), min(len(doc), token.i + 5)):
            if doc[i].text in ['(', ')']:
                return True
        return False

    def _has_latin_morphology(self, text: str) -> bool:
        """Detect Latin morphological patterns."""
        return any(text.startswith(pattern) for pattern in self.config['latin_patterns'])

    def _has_definition_in_parens(self, token: 'Token', doc: 'Doc') -> bool:
        """Check for definition pattern after abbreviation."""
        paren_start = token.i + 1
        if paren_start < len(doc) and doc[paren_start].text == '(':
            # Look for closing paren and alphabetic content
            for i in range(paren_start + 1, min(len(doc), paren_start + 10)):
                if doc[i].text == ')':
                    return any(doc[j].is_alpha for j in range(paren_start + 1, i))
        return False

    def _is_abbreviation_in_parens(self, token: 'Token', doc: 'Doc') -> bool:
        """Check if abbreviation appears in parentheses after its definition."""
        if token.i == 0 or doc[token.i - 1].text != ')':
            return False
        
        paren_depth = 1
        open_paren_idx = None
        
        for i in range(token.i - 2, -1, -1):
            if doc[i].text == ')':
                paren_depth += 1
            elif doc[i].text == '(':
                paren_depth -= 1
                if paren_depth == 0:
                    open_paren_idx = i
                    break
        
        if open_paren_idx is None:
            return False
        
        inside_parens = doc[open_paren_idx + 1:token.i - 1]
        inside_text = ''.join([t.text_with_ws for t in inside_parens]).strip()
        
        return inside_text == token.text

    def _is_definition_pattern(self, token: 'Token', doc: 'Doc') -> bool:
        """Check if abbreviation is being defined inside parentheses."""
        if token.i == 0:
            return False
        
        in_parens = False
        open_paren_idx = None
        
        for i in range(token.i - 1, -1, -1):
            if doc[i].text == ')':
                return False
            elif doc[i].text == '(':
                in_parens = True
                open_paren_idx = i
                break
        
        if not in_parens:
            return False
        
        close_paren_idx = None
        for i in range(token.i + 1, len(doc)):
            if doc[i].text == '(':
                return False
            elif doc[i].text == ')':
                close_paren_idx = i
                break
        
        if close_paren_idx is None:
            return False
        
        if open_paren_idx > 0:
            text_before = doc[:open_paren_idx]
            word_count = len([t for t in text_before if t.is_alpha])
            return word_count >= 2
        
        return False

    def _has_explicit_definition(self, token: 'Token', doc: 'Doc') -> bool:
        """Use dependency parsing to find explicit definitions."""
        # Look for patterns like "X stands for", "X means", etc.
        for i in range(max(0, token.i - 5), min(len(doc), token.i + 5)):
            if doc[i].lemma_ in ['stand', 'mean', 'represent', 'denote']:
                return True
        return False

    def _has_verbal_dependents(self, token: 'Token', doc: 'Doc') -> bool:
        """Check if token has dependents typical of verbs."""
        for child in token.children:
            if child.dep_ in ['dobj', 'iobj', 'nsubj']:  # Direct object, indirect object, subject
                return True
        return False

    def _is_admonition_context(self, token: 'Token', context: Optional[Dict[str, Any]]) -> bool:
        """Check if we're in an admonition context."""
        if not context:
            return False
        
        if context.get('block_type') == 'admonition':
            return True
        
        if context.get('next_block_type') == 'admonition':
            return True
        
        if token.text in self.config['admonition_keywords']:
            return True
        
        return False

    def _get_semantic_action(self, abbreviation: str) -> str:
        """Get appropriate action verb based on abbreviation semantics."""
        return self.config['semantic_actions'].get(abbreviation, self.config['semantic_actions']['default'])

    def _calculate_latin_abbreviation_evidence(self, token: 'Token', sentence, text: str, context: Optional[Dict[str, Any]]) -> float:
        """Calculate evidence score for Latin abbreviation violations."""
        if not self._is_latin_abbreviation(token, token.doc):
            return 0.0
        
        evidence_score = self.config['evidence_thresholds']['latin_base']
        evidence_score = self._apply_linguistic_clues_latin(evidence_score, token, sentence)
        evidence_score = self._apply_structural_clues_latin(evidence_score, token, context)
        evidence_score = self._apply_semantic_clues_latin(evidence_score, token, text, context)
        evidence_score = self._apply_feedback_clues_latin(evidence_score, token, context)
        
        return max(0.0, min(1.0, evidence_score))

    def _calculate_undefined_abbreviation_evidence(self, token: 'Token', sentence, text: str, context: Optional[Dict[str, Any]]) -> float:
        """
        Calculate evidence score for undefined abbreviation violations.
        """
        # DOMAIN KNOWLEDGE GUARD: Check if it's a well-known acronym for the document's domain
        if self._is_domain_specific_acronym(token.text, context):
            return 0.0
        
        # Check universal abbreviations (applies across all domains)
        if self._is_universally_known_abbreviation(token.text):
            return 0.0
        
        if not self._is_abbreviation_candidate(token):
            return 0.0
        
        evidence_score = self.config['evidence_thresholds']['undefined_base']
        evidence_score = self._apply_linguistic_clues_undefined(evidence_score, token, sentence)
        evidence_score = self._apply_structural_clues_undefined(evidence_score, token, context)
        evidence_score = self._apply_semantic_clues_undefined(evidence_score, token, text, context)
        evidence_score = self._apply_feedback_clues_undefined(evidence_score, token, context)
        
        return max(0.0, min(1.0, evidence_score))

    def _calculate_verb_usage_evidence(self, token: 'Token', sentence, text: str, context: Optional[Dict[str, Any]]) -> float:
        """Calculate evidence score for abbreviation-as-verb violations."""
        if not self._is_used_as_verb(token, token.doc):
            return 0.0
        
        evidence_score = self.config['evidence_thresholds']['verb_usage_base']
        evidence_score = self._apply_linguistic_clues_verb(evidence_score, token, sentence)
        evidence_score = self._apply_structural_clues_verb(evidence_score, token, context)
        evidence_score = self._apply_semantic_clues_verb(evidence_score, token, text, context)
        evidence_score = self._apply_feedback_clues_verb(evidence_score, token, context)
        
        return max(0.0, min(1.0, evidence_score))

    def _apply_linguistic_clues_latin(self, evidence_score: float, token: 'Token', sentence) -> float:
        """Apply linguistic analysis for Latin abbreviations."""
        if self._is_in_parenthetical_context(token, token.doc):
            evidence_score += 0.2
        
        prev_token = token.nbor(-1) if token.i > 0 else None
        next_token = token.nbor(1) if token.i < len(token.doc) - 1 else None
        
        if prev_token and prev_token.text == '(':
            evidence_score -= 0.3
        
        if next_token and next_token.text in ['.', ',', ';']:
            evidence_score += 0.1
        
        if token.text.lower() in ['i.e.', 'e.g.'] and next_token and next_token.text == ',':
            evidence_score += 0.2
        
        return evidence_score

    def _apply_linguistic_clues_undefined(self, evidence_score: float, token: 'Token', sentence) -> float:
        """Apply linguistic analysis for undefined abbreviations."""
        if token.ent_type_ == 'PERSON':
            evidence_score -= 0.6
        elif token.ent_type_ == 'GPE':
            evidence_score -= 0.6
        elif token.ent_type_ == 'ORG':
            if any(c.islower() for c in token.text):
                evidence_score -= 0.6
            else:
                evidence_score -= 0.1
        elif token.ent_type_ == 'PRODUCT':
            evidence_score -= 0.4
        elif token.ent_type_ in ['MISC', 'EVENT']:
            evidence_score -= 0.2
        
        if len(token.text) <= 5 and token.text.isupper():
            if self._is_common_technical_acronym(token.text):
                evidence_score -= 0.3
        
        if self._is_contextually_defined(token, token.doc):
            evidence_score -= 0.9
        
        if self._is_admonition_context(token, None):
            evidence_score -= 0.2
        
        return evidence_score

    def _apply_linguistic_clues_verb(self, evidence_score: float, token: 'Token', sentence) -> float:
        """Apply linguistic analysis for verb usage."""
        if token.pos_ == 'VERB':
            evidence_score += 0.2
        
        if token.dep_ == 'ROOT':
            evidence_score += 0.3
        elif token.dep_ in ['ccomp', 'xcomp']:
            evidence_score += 0.2
        
        has_direct_object = any(child.dep_ == 'dobj' for child in token.children)
        if has_direct_object:
            evidence_score += 0.3
        
        prev_token = token.nbor(-1) if token.i > 0 else None
        if prev_token and prev_token.tag_ == 'MD':
            evidence_score += 0.2
        
        return evidence_score

    def _apply_structural_clues_latin(self, evidence_score: float, token: 'Token', context: Optional[Dict[str, Any]]) -> float:
        """Apply document structure clues for Latin abbreviations."""
        if not context:
            return evidence_score
        
        block_type = context.get('block_type', 'paragraph')
        
        if block_type in ['citation', 'bibliography', 'reference']:
            evidence_score -= 0.4
        elif block_type in ['ordered_list_item', 'unordered_list_item']:
            evidence_score -= 0.2
        elif block_type in ['footnote', 'aside', 'sidebar']:
            evidence_score -= 0.3
        elif block_type == 'paragraph':
            evidence_score += 0.1
        
        return evidence_score

    def _apply_structural_clues_undefined(self, evidence_score: float, token: 'Token', context: Optional[Dict[str, Any]]) -> float:
        """Apply document structure clues for undefined abbreviations."""
        if not context:
            return evidence_score
        
        block_type = context.get('block_type', 'paragraph')
        
        if block_type == 'heading':
            heading_level = context.get('block_level', 1)
            if heading_level == 1:
                evidence_score -= 0.4
            elif heading_level >= 2:
                evidence_score -= 0.2
        elif block_type in ['code_block', 'literal_block']:
            evidence_score -= 0.9
        elif block_type == 'inline_code':
            evidence_score -= 0.6
        elif block_type in ['table_cell', 'table_header']:
            evidence_score -= 0.3
        
        return evidence_score

    def _apply_structural_clues_verb(self, evidence_score: float, token: 'Token', context: Optional[Dict[str, Any]]) -> float:
        """Apply document structure clues for verb usage."""
        if not context:
            return evidence_score
        
        block_type = context.get('block_type', 'paragraph')
        
        if block_type in ['code_block', 'literal_block']:
            evidence_score -= 0.8
        elif block_type == 'inline_code':
            evidence_score -= 0.5
        elif block_type == 'heading':
            evidence_score += 0.2
        elif block_type in ['procedure', 'step']:
            evidence_score -= 0.2
        
        return evidence_score

    def _apply_semantic_clues_latin(self, evidence_score: float, token: 'Token', text: str, context: Optional[Dict[str, Any]]) -> float:
        """Apply semantic and content-type clues for Latin abbreviations."""
        if not context:
            return evidence_score
        
        content_type = context.get('content_type', 'general')
        
        if content_type == 'academic':
            evidence_score -= 0.3
        elif content_type == 'legal':
            evidence_score -= 0.2
        elif content_type == 'technical':
            evidence_score += 0.2
        elif content_type == 'marketing':
            evidence_score += 0.3
        
        audience = context.get('audience', 'general')
        if audience in ['expert', 'academic']:
            evidence_score -= 0.2
        elif audience in ['beginner', 'general']:
            evidence_score += 0.3
        
        return evidence_score

    def _apply_semantic_clues_undefined(self, evidence_score: float, token: 'Token', text: str, context: Optional[Dict[str, Any]]) -> float:
        """Apply semantic and content-type clues for undefined abbreviations."""
        if not context:
            return evidence_score
        
        content_type = context.get('content_type', 'general')
        domain = context.get('domain', 'general')
        audience = context.get('audience', 'general')

        if content_type == 'technical' and audience in ['expert', 'developer']:
            evidence_score -= 0.05

        if domain in ['software', 'devops', 'cloud', 'engineering']:
            evidence_score -= 0.5
        elif domain in ['finance', 'legal', 'medical']:
            evidence_score += 0.1

        doc_length = len(text.split())
        if doc_length < 50:
            evidence_score -= 0.05
        elif doc_length > 2000:
            evidence_score += 0.1

        if audience in ['beginner', 'general']:
            evidence_score += 0.2

        if self._is_brand_product_context(token, text, context):
            evidence_score -= 0.3
        
        return evidence_score

    def _apply_semantic_clues_verb(self, evidence_score: float, token: 'Token', text: str, context: Optional[Dict[str, Any]]) -> float:
        """Apply semantic and content-type clues for verb usage."""
        if not context:
            return evidence_score
        
        content_type = context.get('content_type', 'general')
        
        if content_type == 'technical':
            if self._is_imperative_context(token, text):
                evidence_score -= 0.3
        elif content_type == 'procedural':
            evidence_score -= 0.2
        elif content_type == 'narrative':
            evidence_score -= 0.1
        
        return evidence_score

    def _apply_feedback_clues_latin(self, evidence_score: float, token: 'Token', context: Optional[Dict[str, Any]]) -> float:
        """Apply feedback patterns for Latin abbreviations."""
        feedback_patterns = self._get_cached_feedback_patterns('abbreviations')
        
        if token.text.lower() in feedback_patterns.get('accepted_latin_terms', set()):
            evidence_score -= 0.5
        
        if token.text.lower() in feedback_patterns.get('rejected_latin_suggestions', set()):
            evidence_score += 0.3
        
        return evidence_score

    def _apply_feedback_clues_undefined(self, evidence_score: float, token: 'Token', context: Optional[Dict[str, Any]]) -> float:
        """Apply feedback patterns for undefined abbreviations."""
        feedback_patterns = self._get_cached_feedback_patterns('abbreviations')
        
        if token.text in feedback_patterns.get('accepted_undefined_terms', set()):
            evidence_score -= 0.6
        
        if context:
            industry = context.get('industry', 'general')
            industry_terms = feedback_patterns.get(f'{industry}_accepted_abbreviations', set())
            if token.text in industry_terms:
                evidence_score -= 0.4
        
        return evidence_score

    def _apply_feedback_clues_verb(self, evidence_score: float, token: 'Token', context: Optional[Dict[str, Any]]) -> float:
        """Apply feedback patterns for verb usage."""
        feedback_patterns = self._get_cached_feedback_patterns('abbreviations')
        
        if token.text.lower() in feedback_patterns.get('accepted_verb_abbreviations', set()):
            evidence_score -= 0.5
        
        return evidence_score

    def _is_common_technical_acronym(self, text: str) -> bool:
        """Check if this is a commonly known technical acronym."""
        return text.upper() in self.config['common_technical_acronyms']
    
    def _is_domain_specific_acronym(self, token_text: str, context: Optional[Dict[str, Any]]) -> bool:
        """
        Check if an acronym is well-known within the document's domain(s).
        
        This implements domain knowledge intelligence to avoid false positives for
        standard technical acronyms that don't require definition within their domain.
        
        Example usage in context:
            context = {'domain': 'networking'}  # Single domain
            context = {'domains': ['networking', 'security']}  # Multiple domains
        
        Args:
            token_text: The token text to check (e.g., 'EAP', 'TTLS')
            context: Document context that may contain 'domain' or 'domains' keys
            
        Returns:
            True if the acronym is well-known in the document's domain, False otherwise
            
        World-class features:
            - Supports both single domain ('domain') and multiple domains ('domains')
            - Case-insensitive matching
            - Comprehensive domain coverage (networking, cloud, security, etc.)
            - Zero false positives for domain-standard acronyms
        """
        if not context:
            return False
        
        well_known_acronyms = self.config.get('well_known_technical_acronyms', {})
        token_upper = token_text.upper()
        
        # Check single domain
        if 'domain' in context:
            domain = context.get('domain')
            domain_acronyms = well_known_acronyms.get(domain, [])
            if token_upper in domain_acronyms:
                return True
        
        # Check multiple domains (e.g., a document might be both 'networking' and 'security')
        if 'domains' in context:
            domains = context.get('domains', [])
            for domain in domains:
                domain_acronyms = well_known_acronyms.get(domain, [])
                if token_upper in domain_acronyms:
                    return True
        
        return False

    def _count_technical_density(self, text: str) -> int:
        """Count the density of technical terms in the text."""
        text_lower = text.lower()
        count = sum(1 for term in self.config['technical_terms'] if term in text_lower)
        return count

    def _is_universally_known_abbreviation(self, text: str) -> bool:
        """Check if this is a universally known abbreviation."""
        return text.upper() in self.config['universal_abbreviations_set']

    def _is_imperative_context(self, token: 'Token', text: str) -> bool:
        """Check if the token appears in an imperative/command context."""
        sent = token.sent
        
        if sent and len(sent) > 0:
            first_token = sent[0]
            if first_token.pos_ == 'VERB' and first_token.tag_ == 'VB':
                return True
        
        for word in self.config['command_indicators']:
            if word in sent.text.lower():
                return True
        
        return False

    def _is_brand_product_context(self, token: 'Token', text: str, context: Optional[Dict[str, Any]]) -> bool:
        """Check if abbreviation appears in brand/product naming context."""
        if not context:
            return False
        
        content_type = context.get('content_type', '')
        domain = context.get('domain', '')
        
        if content_type in ['marketing', 'branding'] or domain in ['product', 'brand']:
            return True
        
        sent = token.sent
        sent_text = sent.text.lower()
        
        if any(indicator in sent_text for indicator in self.config['brand_indicators']['explicit_markers']):
            return True
        
        token_lower = token.text.lower()
        token_pos = sent_text.find(token_lower)
        if token_pos >= 0:
            before_token = sent_text[:token_pos].strip()
            after_token = sent_text[token_pos + len(token_lower):].strip()
            
            if any(company in before_token[-20:] for company in self.config['brand_indicators']['company_names']):
                return True
            
            if any(descriptor in after_token[:20] for descriptor in self.config['brand_indicators']['product_descriptors']):
                return True
        
        next_token = token.nbor(1) if token.i < len(token.doc) - 1 else None
        if next_token:
            if (next_token.like_num or 
                next_token.text.lower() in ['pro', 'enterprise', 'premium', 'standard', 'lite'] or
                re.match(r'^v?\d+', next_token.text.lower())):
                return True
        
        nearby_tokens = []
        start_idx = max(0, token.i - 2)
        end_idx = min(len(token.doc), token.i + 3)
        
        for i in range(start_idx, end_idx):
            if i != token.i:
                nearby_tokens.append(token.doc[i])
        
        brand_like_tokens = []
        for t in nearby_tokens:
            if (t.text and t.text[0].isupper() and t.is_alpha and 
                len(t.text) > 3 and any(c.islower() for c in t.text)):
                brand_like_tokens.append(t.text)
        
        if len(brand_like_tokens) >= 2:
            return True
        
        return False

    def _get_contextual_message(self, token: 'Token', evidence_score: float, violation_type: str) -> str:
        """Generate context-aware error messages."""
        if violation_type == 'latin':
            if evidence_score > 0.8:
                return f"Avoid using the Latin abbreviation '{token.text}' in this context."
            elif evidence_score > 0.5:
                return f"Consider replacing the Latin abbreviation '{token.text}' with its English equivalent."
            else:
                return f"The Latin abbreviation '{token.text}' may not be appropriate for your audience."
        
        elif violation_type == 'undefined':
            if evidence_score > 0.8:
                return f"Abbreviation '{token.text}' appears undefined and may confuse readers."
            elif evidence_score > 0.5:
                return f"Consider defining '{token.text}' on first use if it's not widely known."
            else:
                return f"Abbreviation '{token.text}' may benefit from definition depending on your audience."
        
        elif violation_type == 'verb':
            if evidence_score > 0.8:
                return f"Avoid using the abbreviation '{token.text}' as a verb."
            elif evidence_score > 0.5:
                return f"Consider rephrasing to avoid using '{token.text}' as a verb."
            else:
                return f"The abbreviation '{token.text}' appears to be used as a verb, which may affect readability."
        
        return f"Issue detected with abbreviation '{token.text}'."

    def _generate_smart_suggestions(self, token: 'Token', context: Optional[Dict[str, Any]], 
                                  violation_type: str, replacement: str = None) -> List[str]:
        """Generate context-aware suggestions."""
        suggestions = []
        
        if violation_type == 'latin':
            if replacement:
                suggestions.append(f"Use its English equivalent: '{replacement}'.")
            suggestions.append("Consider if your audience is familiar with Latin abbreviations.")
            if context and context.get('audience') == 'general':
                suggestions.append("For general audiences, English equivalents are usually clearer.")
        
        elif violation_type == 'undefined':
            suggestions.append(f"Define it on first use: 'Full Term ({token.text})'.")
            if self._is_common_technical_acronym(token.text):
                suggestions.append(f"While '{token.text}' is common in technical contexts, defining it helps all readers.")
            suggestions.append("Consider your audience's familiarity with this abbreviation.")
        
        elif violation_type == 'verb':
            action_verb = self._get_semantic_action(token.text.lower())
            suggestions.append(f"Use a proper verb: '{action_verb} {token.text}' instead of '{token.text}'.")
            suggestions.append("Rephrase the sentence to use the abbreviation as a noun.")
        
        return suggestions
