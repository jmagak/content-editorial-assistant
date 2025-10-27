"""
Conjunctions Rule - Evidence-Based Analysis
Based on IBM Style Guide topic: "Conjunctions"
"""
from typing import List, Dict, Any, Optional, Set, Tuple
try:
    from .base_language_rule import BaseLanguageRule
    from .services.language_vocabulary_service import LanguageVocabularyService
except ImportError:
    # Fallback for direct imports or when module structure is problematic
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from language_and_grammar.base_language_rule import BaseLanguageRule
    from language_and_grammar.services.language_vocabulary_service import LanguageVocabularyService

try:
    from spacy.tokens import Doc, Token, Span
except ImportError:
    Doc = None
    Token = None
    Span = None

# Import rule enhancements for production-grade NLP corrections
try:
    from rule_enhancements import get_adapter, get_token_info
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False

class ConjunctionsRule(BaseLanguageRule):
    """
    Evidence-based analysis for conjunction-related issues using multi-level scoring:
    
    - MICRO-LEVEL: Linguistic analysis (POS tags, dependencies, morphology)
    - MESO-LEVEL: Structural analysis (block types, heading levels, document structure)
    - MACRO-LEVEL: Semantic analysis (content type, audience, domain)
    - FEEDBACK-LEVEL: Learning from user corrections and preferences
    """
    
    def _get_rule_type(self) -> str:
        return 'conjunctions'
    
    def __init__(self):
        """Initialize the conjunctions rule with proper base class setup."""
        super().__init__()
        self.rule_type = self._get_rule_type()
        self.vocabulary_service = LanguageVocabularyService()
        
        # Initialize rule enhancements adapter for NLP corrections
        if ENHANCEMENTS_AVAILABLE:
            self.adapter = get_adapter()
        else:
            self.adapter = None

    def analyze(self, text: str, sentences: List[str], nlp=None, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evidence-based analysis for conjunction usage violations using multi-level scoring.
        """
        # === UNIVERSAL CODE CONTEXT GUARD ===
        # Skip analysis for code blocks, listings, and literal blocks (technical syntax, not prose)
        if context and context.get('block_type') in ['listing', 'literal', 'code_block', 'inline_code']:
            return []
        errors: List[Dict[str, Any]] = []
        context = context or {}
        
        if not nlp:
            return errors
        
        try:
            doc = nlp(text)
            
            # Apply NLP corrections for production-grade accuracy
            corrections = {}
            if self.adapter:
                corrections = self.adapter.enhance_doc_analysis(doc, self._get_rule_type())
            
            for i, sent in enumerate(doc.sents):
                
                # Check coordinating conjunctions with evidence-based scoring
                coord_errors = self._analyze_coordinating_conjunctions_evidence(sent, i, text, context)
                errors.extend(coord_errors)
                
                # Check subordinating conjunctions with multi-level analysis
                subord_errors = self._analyze_subordinating_conjunctions_evidence(sent, i, text, context)
                errors.extend(subord_errors)
                
                # Check parallel structure with contextual awareness (using corrections)
                parallel_errors = self._analyze_parallel_structure_evidence(sent, i, text, context, corrections)
                errors.extend(parallel_errors)
                
        except Exception as e:
            # Graceful degradation with low evidence score
            errors.append(self._create_error(
                sentence=text,
                sentence_index=0,
                message=f"Conjunction analysis failed: {str(e)}",
                suggestions=["Please review conjunction usage manually."],
                severity='low',
                text=text,
                context=context,
                evidence_score=0.0  # No evidence when analysis fails
            ))
        
        return errors
    
    # === EVIDENCE-BASED ANALYSIS METHODS ===
    
    def _analyze_coordinating_conjunctions_evidence(self, sent: 'Span', sentence_index: int, text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evidence-based analysis for coordinating conjunction issues.
        Uses multi-level evidence scoring for sentence-initial conjunctions.
        """
        errors = []
        coordinating_conjunctions = {'and', 'but', 'or', 'nor', 'for', 'yet', 'so'}
        
        for token in sent:
            if (token.lemma_.lower() in coordinating_conjunctions and 
                token.pos_ == 'CCONJ' and token.i == sent.start):
                
                # Multi-level evidence calculation
                evidence_score = self._calculate_sentence_initial_conjunction_evidence(token, sent, text, context)
                
                if evidence_score > 0.1:  # Threshold for reporting
                    errors.append(self._create_error(
                        sentence=sent.text,
                        sentence_index=sentence_index,
                        message=self._get_contextual_conjunction_message('sentence_initial', evidence_score, context, conjunction=token.text),
                        suggestions=self._generate_smart_conjunction_suggestions('sentence_initial', token.text, context),
                        severity=self._determine_severity_from_evidence(evidence_score),
                        text=text,
                        context=context,
                        evidence_score=evidence_score,
                        span=(token.idx, token.idx + len(token.text)),
                        flagged_text=token.text,
                        subtype='sentence_initial_conjunction'
                    ))
        
        return errors
    
    def _analyze_subordinating_conjunctions_evidence(self, sent: 'Span', sentence_index: int, text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evidence-based analysis for subordinating conjunction placement issues.
        NOTE: Comma-related checks are excluded to prevent duplicates with CommasRule.
        """
        errors = []
        subordinating_conjunctions = {
            'although', 'because', 'since', 'while', 'when', 'where', 'if', 'unless',
            'until', 'after', 'before', 'as', 'though', 'whereas', 'provided'
        }
        
        for token in sent:
            if (token.lemma_.lower() in subordinating_conjunctions and 
                token.pos_ == 'SCONJ' and token.i == sent.start):
                
                # Focus on conjunction placement and clause structure (not commas)
                evidence_score = self._calculate_subordinate_conjunction_structure_evidence(token, sent, text, context)
                
                if evidence_score > 0.1:
                    errors.append(self._create_error(
                        sentence=sent.text,
                        sentence_index=sentence_index,
                        message=self._get_contextual_conjunction_message('subordinate_structure', evidence_score, context, conjunction=token.text),
                        suggestions=self._generate_smart_conjunction_suggestions('subordinate_structure', token.text, context),
                        severity=self._determine_severity_from_evidence(evidence_score),
                        text=text,
                        context=context,
                        evidence_score=evidence_score,
                        span=(token.idx, token.idx + len(token.text)),
                        flagged_text=token.text,
                        subtype='subordinate_conjunction_structure'
                    ))
        
        return errors
    
    def _analyze_parallel_structure_evidence(self, sent: 'Span', sentence_index: int, text: str, context: Dict[str, Any], corrections: Dict = None) -> List[Dict[str, Any]]:
        """
        Parallel structure analysis using direct grammatical tag comparison.

        """
        errors = []
        
        if corrections is None:
            corrections = {}
        
        # Find all coordination groups in the sentence
        coordinated_groups = self._find_coordinated_elements_simple(sent)
        
        for group in coordinated_groups:
            if len(group) >= 2:
                parallel_error = self._check_parallel_tags(group, sent, sentence_index, text, context, corrections)
                if parallel_error:
                    errors.append(parallel_error)
        
        return errors

    def _find_coordinated_elements_simple(self, sent: 'Span') -> List[List['Token']]:
        """
        Robust coordination detection using direct conj relationships.
        Filters out punctuation and handles SpaCy parser quirks with prepositional phrases.
        """
        groups = []
        processed_tokens = set()
        
        def is_formatting_token(tok):
            """Check if token is punctuation/formatting that should be skipped."""
            if tok.pos_ in ['PUNCT', 'SYM', 'SPACE']:
                return True
            if len(tok.text.strip()) == 1 and not tok.text.isalnum():
                return True
            if tok.text.strip() in ['-', '*', '+', '•', '·', '◦', '▪', '▸', '→', '⇒']:
                return True
            return False
        
        def is_linking_verb(token):
            """Check if token is a linking/copular verb (be, become, seem, etc.)"""
            linking_verbs = {'be', 'become', 'seem', 'appear', 'feel', 'look', 'sound', 
                           'taste', 'smell', 'remain', 'stay', 'grow', 'turn', 'prove'}
            return (token.pos_ in ['AUX', 'VERB'] and 
                   token.lemma_.lower() in linking_verbs)
        
        def find_predicate_complement(verb_token):
            """
            Find the predicate complement of a linking verb.
            Returns the main predicate element (advmod, acomp, amod, attr, etc.)
            
            Example: "are up to date" -> returns "up" (the head of the predicate phrase)
            """
            # Look for predicate complements: acomp, attr, amod, advmod
            predicate_deps = ['acomp', 'attr', 'amod', 'advmod', 'oprd']
            for child in verb_token.children:
                if child.dep_ in predicate_deps and not is_formatting_token(child):
                    return child
            return None
        
        def find_true_parallel_element(token, head):
            """
            Handle SpaCy quirks and find the actual parallel elements.
            
            1. Linking verbs: Find predicate complements, not the verb itself
            2. Prepositions: Find prepositional objects
            3. Adverbs: Handle prepositional phrase quirks
            """
            # === CASE 1: Linking Verb with Predicate Complements ===
            # "The system is fast and reliable" -> compare "fast" vs "reliable", not "is" vs "reliable"
            if is_linking_verb(head):
                predicate = find_predicate_complement(head)
                if predicate:
                    # Head should be the predicate complement, not the linking verb
                    return predicate
            
            # === CASE 2: Prepositional Phrase Quirk ===
            # "on X or also on Y" -> find the true parallel preposition objects
            if head.pos_ == 'ADP' and token.pos_ == 'ADV':
                for i in range(token.i + 1, min(token.i + 3, len(sent))):
                    next_tok = sent[i] if hasattr(sent, '__getitem__') else sent.doc[i]
                    if next_tok.pos_ == 'ADP':
                        for child in next_tok.children:
                            if child.dep_ == 'pobj':
                                return child
                        return next_tok
            
            return token
        
        for token in sent:
            if is_formatting_token(token):
                continue
                
            if token.dep_ == 'conj' and token not in processed_tokens:
                if is_formatting_token(token.head):
                    continue
                
                # Skip adverbs marked as conjuncts (SpaCy tagging quirk)
                if token.pos_ == 'ADV':
                    if (token.head.pos_ == 'ADP' or 
                        token.head.dep_ in ['pobj', 'nsubj', 'dobj'] or 
                        (hasattr(token.head, 'head') and token.head.head.pos_ == 'ADP')):
                        continue
                
                # Get the true parallel elements
                head_element = token.head
                
                # === SPECIAL CASE: Linking Verb ===
                # If head is a linking verb, get its predicate complement
                if is_linking_verb(head_element):
                    predicate = find_predicate_complement(head_element)
                    if predicate:
                        head_element = predicate
                
                # Get the coordinated element (may also transform linking verb cases)
                conj_element = find_true_parallel_element(token, token.head)
                
                # For prepositional phrases, compare objects (pobj) not prepositions
                if head_element.pos_ == 'ADP':
                    for child in head_element.children:
                        if child.dep_ == 'pobj':
                            head_element = child
                            break
                
                if conj_element.pos_ == 'ADP':
                    for child in conj_element.children:
                        if child.dep_ == 'pobj':
                            conj_element = child
                            break
                
                group = [head_element, conj_element]
                processed_tokens.add(token)
                processed_tokens.add(token.head)
                
                # Find all other tokens coordinated with the same head
                for other_token in sent:
                    if is_formatting_token(other_token):
                        continue
                        
                    if (other_token.dep_ == 'conj' and 
                        other_token.head == token.head and 
                        other_token not in processed_tokens):
                        group.append(other_token)
                        processed_tokens.add(other_token)
                
                if len(group) >= 2:
                    groups.append(group)
        
        return groups

    
    # === MULTI-LEVEL EVIDENCE CALCULATION ===
    
    def _calculate_sentence_initial_conjunction_evidence(self, conjunction: 'Token', sent: 'Span', text: str, context: Dict[str, Any]) -> float:
        """
        Multi-level evidence calculation for sentence-initial conjunction issues.
        
        Uses MICRO/MESO/MACRO/FEEDBACK level analysis for comprehensive scoring.
        """
        
        evidence_score = self._get_base_sentence_initial_evidence(conjunction, sent)
        evidence_score = self._apply_linguistic_clues_sentence_initial(evidence_score, conjunction, sent, text)
        evidence_score = self._apply_structural_clues_sentence_initial(evidence_score, context)
        evidence_score = self._apply_semantic_clues_sentence_initial(evidence_score, conjunction, text, context)
        evidence_score = self._apply_feedback_clues_sentence_initial(evidence_score, conjunction, context)
        
        return max(0.0, min(1.0, evidence_score))  # Clamp to valid range
    
    def _calculate_subordinate_conjunction_structure_evidence(self, conjunction: 'Token', sent: 'Span', text: str, context: Dict[str, Any]) -> float:
        """
        Multi-level evidence calculation for subordinate conjunction structure issues.
        NOTE: Comma-related analysis is excluded to prevent duplicates.
        """
        
        evidence_score = self._get_base_subordinate_structure_evidence(conjunction, sent)
        evidence_score = self._apply_linguistic_clues_subordinate_structure(evidence_score, conjunction, sent, text)
        evidence_score = self._apply_structural_clues_subordinate_structure(evidence_score, context)
        evidence_score = self._apply_semantic_clues_subordinate_structure(evidence_score, conjunction, text, context)
        evidence_score = self._apply_feedback_clues_subordinate_structure(evidence_score, conjunction, context)
        
        return max(0.0, min(1.0, evidence_score))
    
    # === BASE EVIDENCE METHODS ===
    
    def _get_base_sentence_initial_evidence(self, conjunction: 'Token', sent: 'Span') -> float:
        """Get base evidence score for sentence-initial conjunction concerns."""
        
        # Different conjunctions have different base evidence
        conjunction_evidence = {
            'and': 0.3,  # Common and sometimes acceptable
            'but': 0.2,  # Often acceptable for contrast
            'or': 0.4,   # Less common, more problematic
            'nor': 0.6,  # Formal, often awkward at sentence start
            'for': 0.5,  # Can be confusing (preposition vs conjunction)
            'yet': 0.3,  # Similar to 'but', sometimes acceptable
            'so': 0.4    # Informal, better to restructure
        }
        
        return conjunction_evidence.get(conjunction.lemma_.lower(), 0.4)
    
    def _get_base_subordinate_structure_evidence(self, conjunction: 'Token', sent: 'Span') -> float:
        """Get base evidence score for subordinate conjunction structure concerns."""
        
        # Focus on clause complexity and structure, not commas
        main_clause_start = self._find_main_clause_start(conjunction, sent)
        if main_clause_start:
            clause_length = main_clause_start.i - conjunction.i
            
            # Very long subordinate clauses may be problematic for clarity
            if clause_length >= 10:  # Very long subordinate clause
                return 0.4
            elif clause_length >= 7:  # Moderately long subordinate clause
                return 0.2
            else:  # Short subordinate clause
                return 0.1
        
        return 0.2  # Default evidence
    
    # === LINGUISTIC CLUES (MICRO-LEVEL) ===
    
    def _apply_linguistic_clues_sentence_initial(self, evidence_score: float, conjunction: 'Token', sent: 'Span', text: str) -> float:
        """Apply linguistic analysis clues for sentence-initial conjunctions."""
        
        # Check if previous sentence exists and ends with specific patterns
        if conjunction.i > 0:
            # Check for sentence connectivity patterns
            prev_tokens = [token for token in sent.doc[:sent.start] if not token.is_space]
            if prev_tokens:
                last_prev_token = prev_tokens[-1]
                
                # Strong connection suggests conjunction is appropriate
                if last_prev_token.text in ['.', '!', '?']:
                    # Look for semantic connection indicators
                    if any(word in text.lower() for word in ['however', 'therefore', 'thus', 'consequently']):
                        evidence_score -= 0.2  # Reduce evidence if strong connective context
        
        # Check conjunction's syntactic role
        if conjunction.dep_ in ['cc', 'mark']:  # Coordinating or subordinating marker
            evidence_score += 0.1  # Slight increase for clear syntactic role
        
        return evidence_score
    
    def _apply_linguistic_clues_subordinate_structure(self, evidence_score: float, conjunction: 'Token', sent: 'Span', text: str) -> float:
        """Apply linguistic analysis clues for subordinate conjunction structure."""
        
        # Analyze the complexity of the subordinate clause
        subordinate_tokens = []
        for token in sent[conjunction.i - sent.start:]:
            if token.dep_ == 'nsubj' and token.head != conjunction:
                break  # Found main clause subject
            subordinate_tokens.append(token)
        
        # Complex subordinate clauses may need restructuring
        if len(subordinate_tokens) > 8:
            evidence_score += 0.2
        
        # Check for nested clauses within the subordinate clause
        nested_conjunctions = sum(1 for token in subordinate_tokens 
                                if token.pos_ in ['SCONJ', 'CCONJ'])
        if nested_conjunctions > 1:
            evidence_score += 0.2  # Nested complexity may impact clarity
        
        return evidence_score
    
    # === STRUCTURAL CLUES (MESO-LEVEL) ===
    
    def _apply_structural_clues_sentence_initial(self, evidence_score: float, context: Dict[str, Any]) -> float:
        """Apply document structure clues for sentence-initial conjunction detection."""
        
        block_type = context.get('block_type', 'paragraph')
        
        if block_type in ['code_block', 'literal_block']:
            evidence_score -= 0.8
        elif block_type == 'inline_code':
            evidence_score -= 0.6
        
        if block_type == 'heading':
            heading_level = context.get('block_level', 1)
            if heading_level == 1:
                evidence_score += 0.3
            elif heading_level <= 3:
                evidence_score += 0.2
        
        if block_type in ['list_item', 'unordered_list', 'ordered_list']:
            evidence_score -= 0.2
        
        if block_type in ['note', 'tip', 'warning', 'caution', 'important']:
            evidence_score += 0.1
        
        return evidence_score
    
    def _apply_structural_clues_subordinate_structure(self, evidence_score: float, context: Dict[str, Any]) -> float:
        """Apply document structure clues for subordinate conjunction structure."""
        
        block_type = context.get('block_type', 'paragraph')
        
        if block_type in ['procedure', 'step', 'instruction']:
            evidence_score += 0.2
        
        if block_type in ['heading', 'title']:
            evidence_score += 0.1
        
        if block_type in ['specification', 'requirement']:
            evidence_score += 0.3
        
        return evidence_score
    
    # === SEMANTIC CLUES (MACRO-LEVEL) ===
    
    def _apply_semantic_clues_sentence_initial(self, evidence_score: float, conjunction: 'Token', text: str, context: Dict[str, Any]) -> float:
        """Apply semantic analysis clues for sentence-initial conjunctions."""
        
        content_type = context.get('content_type', 'general')
        audience = context.get('audience', 'general')
        
        if content_type == 'academic':
            evidence_score += 0.4
        elif content_type == 'formal':
            evidence_score += 0.3
        elif content_type == 'legal':
            evidence_score += 0.5
        elif content_type == 'technical':
            evidence_score += 0.2
        elif content_type in ['conversational', 'marketing']:
            evidence_score -= 0.2
        
        if audience in ['expert', 'developer']:
            evidence_score -= 0.1
        elif audience in ['beginner', 'general']:
            evidence_score += 0.2
        
        if self._is_procedural_documentation(text):
            evidence_score += 0.2
        elif self._is_api_documentation(text):
            evidence_score += 0.1
        
        return evidence_score
    
    def _apply_semantic_clues_subordinate_structure(self, evidence_score: float, conjunction: 'Token', text: str, context: Dict[str, Any]) -> float:
        """Apply semantic analysis clues for subordinate conjunction structure."""
        
        content_type = context.get('content_type', 'general')
        
        if content_type in ['academic', 'formal', 'legal']:
            evidence_score += 0.2
        elif content_type == 'technical':
            evidence_score += 0.1
        elif content_type == 'procedural':
            evidence_score += 0.3
        
        if self._has_high_technical_density(text):
            evidence_score += 0.2
        
        return evidence_score
    
    # === FEEDBACK PATTERNS (LEARNING CLUES) ===
    
    def _apply_feedback_clues_sentence_initial(self, evidence_score: float, conjunction: 'Token', context: Dict[str, Any]) -> float:
        """Apply user feedback patterns for sentence-initial conjunctions."""
        
        feedback_patterns = self._get_cached_feedback_patterns('conjunctions')
        conjunction_text = conjunction.text.lower()
        
        if conjunction_text in feedback_patterns.get('accepted_terms', set()):
            evidence_score -= 0.3
        
        if conjunction_text in feedback_patterns.get('rejected_terms', set()):
            evidence_score += 0.3
        
        content_type = context.get('content_type', 'general')
        context_patterns = feedback_patterns.get('context_patterns', {})
        if content_type in context_patterns:
            context_feedback = context_patterns[content_type]
            if conjunction_text in context_feedback.get('accepted', set()):
                evidence_score -= 0.2
            elif conjunction_text in context_feedback.get('rejected', set()):
                evidence_score += 0.2
        
        return evidence_score
    
    def _apply_feedback_clues_subordinate_structure(self, evidence_score: float, conjunction: 'Token', context: Dict[str, Any]) -> float:
        """Apply user feedback patterns for subordinate conjunction structure."""
        
        feedback_patterns = self._get_cached_feedback_patterns('conjunctions')
        correction_success = feedback_patterns.get('correction_success', {})
        structure_corrections = correction_success.get('subordinate_structure', {})
        
        conjunction_text = conjunction.text.lower()
        if conjunction_text in structure_corrections:
            success_rate = structure_corrections[conjunction_text]
            if success_rate > 0.8:
                evidence_score += 0.2
            elif success_rate < 0.3:
                evidence_score -= 0.3
        
        return evidence_score
    
    # === UTILITY METHODS ===
    
    def _find_main_clause_start(self, subordinating_conj: 'Token', sent: 'Span') -> Optional['Token']:
        """Find the start of the main clause after a subordinating conjunction."""
        
        for token in sent[subordinating_conj.i - sent.start + 1:]:
            # Look for main clause subject that's not dependent on the subordinate clause
            if (token.dep_ == 'nsubj' and 
                token.head != subordinating_conj and
                not any(child.head == subordinating_conj for child in token.subtree)):
                return token
        
        return None
    
    
    def _check_parallel_tags(self, elements: List['Token'], sent: 'Span', sentence_index: int, text: str, context: Dict[str, Any], corrections: Dict = None) -> Optional[Dict[str, Any]]:
        """
        Simple, robust parallel structure check using grammatical tags with NLP corrections.
        
        Compares token.tag_ for each coordinated element:
        - VBG (creating) vs NN (analysis) vs TO (to) + VB (generate) = violation
        - VBG vs VBG vs VBG = parallel ✓
        """
        
        if len(elements) < 2:
            return None
        
        if corrections is None:
            corrections = {}
        
        # === LINGUISTIC ANCHOR: Basic POS Tag Sanity Check with NLP Corrections ===
        # If all elements share the same basic Part-of-Speech (after corrections), they are parallel.
        # This acts as a robust guard against complex parsing errors on simple lists.
        
        # Get corrected POS tags for all elements
        element_pos_tags = []
        for elem in elements:
            if self.adapter:
                token_info = get_token_info(elem, corrections)
                elem_pos = token_info.get('pos_', elem.pos_)
            else:
                elem_pos = elem.pos_
            element_pos_tags.append(elem_pos)
        
        # Check if all corrected POS tags match
        first_pos = element_pos_tags[0]
        if all(pos == first_pos for pos in element_pos_tags):
            return None  # EXIT EARLY: The structure is fundamentally parallel (after corrections).
        
        # === ZERO FALSE POSITIVE GUARD: Passive and Active Verb Coordination ===
        if len(elements) == 2:
            # Strategy 1: Check POS tags directly (including passive constructions)
            both_verbal = all(
                elem.pos_ in ['VERB', 'AUX'] or 
                elem.tag_ in ['VBN', 'VBG', 'VBZ', 'VBP', 'VBD', 'VB']
                for elem in elements
            )
            if both_verbal:
                return None  # EXIT EARLY: Both are verbal predicates - parallel
            
            # Strategy 2: Use corrected POS tags from NLP correction layer
            if self.adapter and corrections:
                corrected_tags = []
                for elem in elements:
                    token_info = get_token_info(elem, corrections)
                    corrected_tags.append(token_info.get('pos_', elem.pos_))
                
                # If both corrected to VERB, they're parallel
                if all(tag == 'VERB' for tag in corrected_tags):
                    return None  # EXIT EARLY: Both are verbs after NLP corrections
            
            # Strategy 3: Syntactic evidence - both ROOT or ROOT+conj relationship
            # This catches cases like "RHEL blocks... and uses..." where SpaCy mislabels "blocks" as NOUN
            # but the dependency structure shows it's actually functioning as the main verb
            is_verb_by_syntax = []
            for elem in elements:
                # Check if token is ROOT (main verb) or conj to ROOT (coordinated verb)
                is_root = elem.dep_ == 'ROOT'
                is_conj_to_verb = (elem.dep_ == 'conj' and 
                                  elem.head.pos_ in ['VERB', 'AUX', 'NOUN'])  # NOUN catches mislabeled verbs
                
                # Additional check: if it has verb-like dependents (subjects, objects)
                has_verb_dependents = any(child.dep_ in ['nsubj', 'nsubjpass', 'dobj', 'iobj', 'xcomp', 'ccomp'] 
                                         for child in elem.children)
                
                # If the word could be both noun and verb (ambiguous), check morphology
                can_be_verb = (elem.tag_ in ['VBZ', 'VBP', 'VBD', 'VB', 'VBG', 'VBN'] or
                              # Words that look like plural nouns but are actually verbs in context
                              (elem.tag_ == 'NNS' and (is_root or is_conj_to_verb or has_verb_dependents)))
                
                is_verb_by_syntax.append(is_root or is_conj_to_verb or has_verb_dependents or can_be_verb)
            
            # If both elements show verb-like syntactic behavior, treat as parallel
            if all(is_verb_by_syntax):
                return None  # EXIT EARLY: Both function as verbs syntactically
        
        # === ZERO FALSE POSITIVE GUARD: Adjective/Noun list preceding a head noun ===
        if self._is_adjective_noun_list_pattern(elements, sent):
            return None  # EXIT EARLY: This is a valid adjective/noun coordination
        
        # === ZERO FALSE POSITIVE GUARD: "between X and Y" is always parallel ===
        if len(elements) == 2:
            # Strategy 1: Direct head is "between" (most common)
            head_of_first = elements[0].head
            if head_of_first and head_of_first.pos_ == 'ADP' and head_of_first.lemma_.lower() == 'between':
                return None  # EXIT EARLY: This is a valid "between X and Y" structure
            
            # Strategy 2: Check children of first element for "between" as a modifier
            # Handles: "The range is between 0 and infinity" where "between" modifies "0"
            for child in elements[0].children:
                if child.pos_ == 'ADP' and child.lemma_.lower() == 'between':
                    return None  # EXIT EARLY: Found "between" as child modifier
            
            # Strategy 3: Check if "between" appears in the dependency path to the verb
            # Look at nearby tokens in the sentence for "between" preposition
            start_idx = min(elem.i for elem in elements)
            end_idx = max(elem.i for elem in elements)
            for i in range(max(0, start_idx - 3), min(len(sent), end_idx + 1)):
                token = sent[i]
                if token.pos_ == 'ADP' and token.lemma_.lower() == 'between':
                    # Check if this "between" is related to our coordination
                    # by verifying it appears before or near the coordinated elements
                    if token.i < elements[0].i:
                        return None  # EXIT EARLY: Found "between" governing this coordination
        
        # === STYLE CHECK: Gerund + Noun Mixing ===
        # Check for common style issue: mixing gerunds (VBG) with plain nouns (NN)
        # Example: "creating and analysis" should be "creation and analysis" or "creating and analyzing"
        tags = [elem.tag_ for elem in elements]
        if len(tags) == 2:
            if (tags[0] == 'VBG' and tags[1] == 'NN') or (tags[0] == 'NN' and tags[1] == 'VBG'):
                # This is a gerund mixed with a plain noun - style issue
                evidence_score = 0.75  # High confidence this should be fixed
                
                gerund_idx = 0 if tags[0] == 'VBG' else 1
                noun_idx = 1 if gerund_idx == 0 else 0
                
                message = f"Mixed forms: '{elements[gerund_idx].text}' (gerund) and '{elements[noun_idx].text}' (noun). Use consistent grammatical forms for parallel structure."
                suggestions = [
                    f"Use both gerunds: '{elements[gerund_idx].text}' and analyzing/processing",
                    f"Use both nouns: 'creation' and '{elements[noun_idx].text}'"
                ]
                return self._create_error(
                    sentence=sent.text,
                    sentence_index=sentence_index,
                    message=message,
                    suggestions=suggestions,
                    severity='medium',
                    text=text,
                    context=context,
                    evidence_score=evidence_score,
                    span=(elements[0].idx, elements[-1].idx + len(elements[-1].text)),
                    flagged_text=f'{elements[0].text}, {elements[1].text}',
                    subtype='parallel_structure_gerund_noun_mix'
                )
        
        # Get grammatical patterns for each element
        element_patterns = []
        element_texts = []
        
        for element in elements:
            # Get the grammatical pattern for this element
            pattern = self._get_grammatical_pattern(element, sent)
            element_patterns.append(pattern)
            
            # Get clean text representation
            element_text = self._get_element_text(element, sent)
            element_texts.append(element_text)
        
        # === FUNCTIONAL CATEGORY NORMALIZATION ===
        # Map specific patterns to broad functional categories
        # This allows the rule to understand that gerunds and infinitives are both
        # actions, and that nouns and prepositional phrases are both nominal structures.
        
        def get_functional_category(pattern: str) -> str:
            """
            Map specific grammatical patterns to broad functional categories.
            
            - ACTION: Gerunds (ACTION_GERUND), Infinitives (ACTION_INFINITIVE)
            - NOMINAL: Nouns (NOMINAL_PHRASE), Proper Nouns (NOMINAL_PROPER_NOUN),
                       Prepositional Phrases (NOMINAL_PREP_PHRASE)
            - Other categories retain their specific pattern for strict matching
            
            This provides granular control: within ACTION, we accept gerunds and
            infinitives as parallel. But ACTION vs NOMINAL is a real error.
            """
            if pattern.startswith('ACTION_'):
                return 'ACTION'
            if pattern.startswith('NOMINAL_'):
                return 'NOMINAL'
            return pattern  # Return the pattern itself if not in a functional group
        
        # Normalize to functional categories
        normalized_categories = [get_functional_category(p) for p in element_patterns]
        
        # Check for consistency in the *normalized functional categories*
        unique_categories = set(normalized_categories)
        if len(unique_categories) > 1:
            # Pass the ORIGINAL patterns to the evidence calculation for better messaging
            evidence_score = self._calculate_tag_mismatch_evidence(element_patterns, context)
            
            if evidence_score > 0.5:  # Simple threshold
                message = self._create_parallel_structure_message(element_texts, element_patterns)
                suggestions = self._create_parallel_structure_suggestions(element_patterns)
                
                return self._create_error(
                    sentence=sent.text,
                    sentence_index=sentence_index,
                    message=message,
                    suggestions=suggestions,
                    severity=self._determine_severity_from_evidence(evidence_score),
                    text=text,
                    context=context,
                    evidence_score=evidence_score,
                    span=(elements[0].idx, elements[-1].idx + len(elements[-1].text)),
                    flagged_text=', '.join(element_texts),
                    subtype='parallel_structure_tag_mismatch'
                )
        
        return None
    
    def _determine_severity_from_evidence(self, evidence_score: float) -> str:
        """Determine error severity based on evidence score."""
        
        if evidence_score >= 0.7:
            return 'high'
        elif evidence_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _get_contextual_conjunction_message(self, violation_type: str, evidence_score: float, context: Dict[str, Any], **kwargs) -> str:
        """Generate contextual error message for conjunction violations."""
        
        content_type = context.get('content_type', 'general')
        
        if violation_type == 'sentence_initial':
            conjunction = kwargs.get('conjunction', 'conjunction')
            if evidence_score > 0.6:
                if content_type in ['academic', 'formal']:
                    return f"Avoid starting sentences with '{conjunction}' in formal writing. Consider restructuring the sentence."
                else:
                    return f"Consider avoiding '{conjunction}' at the beginning of sentences for improved clarity."
            else:
                return f"Starting sentences with '{conjunction}' may impact readability. Consider restructuring if appropriate."
                
        elif violation_type == 'subordinate_structure':
            conjunction = kwargs.get('conjunction', 'subordinate conjunction')
            return f"Consider restructuring this '{conjunction}' clause for improved clarity and readability."
            
        elif violation_type == 'parallel_structure':
            elements = kwargs.get('elements', [])
            return f"These coordinated elements may not have parallel structure: {', '.join(elements[:2])}. Ensure consistent grammatical forms."
        
        return f"Consider reviewing this {violation_type} for context appropriateness."
    
    def _generate_smart_conjunction_suggestions(self, violation_type: str, original_text: str, context: Dict[str, Any]) -> List[str]:
        """Generate smart suggestions for conjunction violations."""
        
        content_type = context.get('content_type', 'general')
        suggestions = []
        
        if violation_type == 'sentence_initial':
            suggestions.extend([
                "Combine with the previous sentence using a comma",
                "Start the sentence with a different word or phrase",
                "Restructure to put the main clause first"
            ])
            
        elif violation_type == 'subordinate_structure':
            suggestions.extend([
                "Break into shorter, clearer sentences",
                "Restructure to put the main clause first",
                "Simplify the subordinate clause structure"
            ])
            
        elif violation_type == 'parallel_structure':
            suggestions.extend([
                "Use consistent grammatical forms for coordinated elements",
                "Ensure all items in the series follow the same pattern",
                "Consider breaking into separate sentences if too complex"
            ])
        
        # Add context-specific suggestions
        if content_type == 'technical':
            suggestions.append("Ensure clarity for technical accuracy")
        elif content_type == 'procedural':
            suggestions.append("Maintain clear step-by-step flow")
        
        return suggestions[:3]  # Return top 3 suggestions

    def _is_adjective_noun_list_pattern(self, elements: List['Token'], sent: 'Span') -> bool:
        """
        ZERO FALSE POSITIVE GUARD: Detect valid adjective/noun coordination patterns.
        """
        if len(elements) < 2:
            return False
        
        # Check if all elements are adjectives or nouns
        all_adj_or_noun = all(elem.pos_ in ['ADJ', 'NOUN'] for elem in elements)
        if not all_adj_or_noun:
            return False  # Mixed with other parts of speech - not this pattern
        # Pattern: The coordinated elements should all relate to the same noun
        
        # Strategy 1: Check if the last element has a noun child (common head)
        last_element = elements[-1]
        for child in last_element.children:
            if child.pos_ == 'NOUN' and child.i > last_element.i:
                # Found a noun following the last coordinated element
                # This is the head noun that all modifiers are describing
                return True
        
        # Strategy 2: Check if there's a noun immediately after the coordination
        last_idx = last_element.i
        for i in range(last_idx + 1, min(last_idx + 3, len(sent))):
            token = sent[i]
            if token.pos_ == 'NOUN':
                # Check if this noun is syntactically related to the coordinated elements
                # It should be the head or object that the adjectives modify
                for elem in elements:
                    if token == elem.head or elem == token.head:
                        return True
                # Also accept if the noun appears immediately after with no intervening content words
                if i == last_idx + 1 or (i == last_idx + 2 and sent[last_idx + 1].pos_ == 'PUNCT'):
                    return True
        
        # Strategy 3: Check if all elements share a common noun head
        heads = [elem.head for elem in elements]
        if heads and all(head.pos_ == 'NOUN' for head in heads):
            # Check if they all modify the same noun
            unique_heads = set(head.i for head in heads)
            if len(unique_heads) == 1:
                return True
        
        return False

    def _get_grammatical_pattern(self, token: 'Token', sent) -> str:
        """
        Get the FUNCTIONAL grammatical pattern for a coordinated element.
        
        Returns functional categories that group grammatically equivalent forms:
        - ACTION_GERUND/ACTION_INFINITIVE for action phrases  
        - NOMINAL_PHRASE/NOMINAL_PROPER_NOUN/NOMINAL_PREP_PHRASE for nominal structures
        - PASSIVE_PHRASE for passive voice constructions (are synced, is encrypted)
        - PAST_VERB_FORM for past tense/participles
        
        This functional equivalence treats gerunds and infinitives as parallel actions.
        """
        
        # === LINGUISTIC GUARD: Correct mis-tagged imperative verbs at sentence start ===
        # Common imperative verbs that might be mis-tagged when capitalized
        common_imperatives = {
            'review', 'commit', 'push', 'run', 'open', 'copy', 'update', 'access', 
            'navigate', 'click', 'select', 'enter', 'configure', 'install', 'download',
            'upload', 'export', 'import', 'delete', 'remove', 'add', 'create', 'modify',
            'edit', 'save', 'load', 'restart', 'start', 'stop', 'pause', 'resume',
            'execute', 'launch', 'close', 'exit', 'verify', 'check', 'test', 'validate',
            'confirm', 'approve', 'reject', 'submit', 'cancel', 'reset', 'refresh',
            'build', 'deploy', 'merge', 'fork', 'clone', 'pull', 'fetch', 'branch',
            'tag', 'release', 'publish', 'sync', 'backup', 'restore', 'migrate'
        }
        
        # Check if at sentence start and mis-tagged as noun/adjective/proper noun
        if token.is_sent_start and token.tag_ in ['NNP', 'NN', 'JJ'] and token.lower_ in common_imperatives:
            # Override: treat as base verb (VB) for parallel structure analysis
            return 'VB'
        
        # Also check coordinated verbs that might be mis-tagged (not at sentence start)
        # Example: "Review, configure, and deploy" -> "configure" might be tagged as NN
        if (token.dep_ == 'conj' and token.tag_ in ['NN', 'JJ', 'NNP'] and 
            token.lower_ in common_imperatives):
            # Check if the head is a verb or another imperative
            if token.head.tag_ in ['VB', 'VBP', 'VBZ'] or token.head.lower_ in common_imperatives:
                return 'VB'
        
        # === LINGUISTIC ANCHOR: Detect Passive Voice Constructions ===
        if token.tag_ in ['VBN', 'VBD']:
            # Check if this participle has an auxiliary "be" (auxpass dependency)
            for child in token.children:
                if child.dep_ == 'auxpass' and child.lemma_.lower() == 'be':
                    return 'PASSIVE_PHRASE'
            
            # Also check if token's head is an auxiliary "be" (for ROOT dependencies)
            if token.dep_ == 'ROOT' or token.dep_ == 'conj':
                for child in token.children:
                    if child.dep_ in ['aux', 'auxpass'] and child.lemma_.lower() == 'be':
                        return 'PASSIVE_PHRASE'
        
            # === ENHANCED LINGUISTIC ANCHOR: Inherit passive voice in coordinated phrases ===
            if token.dep_ == 'conj' and hasattr(token, 'head'):
                current = token.head
                max_depth = 5  # Prevent infinite loops
                depth = 0
                while current and depth < max_depth:
                    # Check if current verb has a passive auxiliary
                    is_passive = any(
                        child.dep_ in ['aux', 'auxpass'] and child.lemma_.lower() == 'be' 
                        for child in current.children
                    )
                    if is_passive:
                        return 'PASSIVE_PHRASE'  # Inherit passive from coordinated head
                    
                    # If current is also a conj, check its head (for multi-way coordination)
                    if current.dep_ == 'conj' and hasattr(current, 'head'):
                        current = current.head
                        depth += 1
                    else:
                        break 
        
        if token.pos_ in ['ADJ', 'ADV'] and token.dep_ in ['acomp', 'amod', 'advmod']:
            return 'PREDICATE_ADJECTIVE'
        
        # Prepositional phrase objects: use dependency parsing to identify
        if token.dep_ == 'pobj' and hasattr(token, 'head') and token.head.pos_ == 'ADP':
            return 'NOMINAL_PREP_PHRASE'
        
        # Normalize inconsistent -ing tagging in coordinated lists
        if token.text.lower().endswith('ing') and self._is_in_ing_coordination(token, sent):
            return 'ACTION_GERUND'
        
        # Gerunds: "creating", "updating"
        if token.tag_ == 'VBG':
            return 'ACTION_GERUND'
        
        # Infinitives: "to create", "to update"
        if token.tag_ == 'VB' and token.i > 0:
            if hasattr(sent, 'start'):
                start_offset = sent.start
            else:
                start_offset = 0
            
            prev_idx = token.i - start_offset - 1
            if prev_idx >= 0:
                prev_token = sent[prev_idx]
                if prev_token and prev_token.text.lower() == 'to' and prev_token.pos_ == 'PART':
                    return 'ACTION_INFINITIVE'
        
        # Normalize past tense/participle forms (SpaCy often misstags VBD/VBN)
        if token.tag_ in ['VBN', 'VBD']:
            return 'PAST_VERB_FORM'
        
        # Regular and plural nouns
        if token.tag_ in ['NN', 'NNS']:
            return 'NOMINAL_PHRASE'
        
        # Proper nouns
        if token.tag_ in ['NNP', 'NNPS']:
            return 'NOMINAL_PROPER_NOUN'
        
        # Prepositions
        if token.pos_ == 'ADP':
            return 'NOMINAL_PREP_PHRASE'
        
        # Other grammatical categories
        tag_mappings = {
            'VB': 'VB',
            'VBP': 'VBP',
            'VBZ': 'VBZ',
            'JJ': 'JJ',
            'RB': 'RB',
        }
        
        return tag_mappings.get(token.tag_, token.tag_)

    def _is_in_ing_coordination(self, token: 'Token', sent) -> bool:
        """Check if this -ing word is part of a coordinated list with other -ing words."""
        head = token.head if token.dep_ == 'conj' else token
        
        ing_words = []
        if head.text.lower().endswith('ing'):
            ing_words.append(head)
        
        for child in head.children:
            if child.dep_ == 'conj' and child.text.lower().endswith('ing'):
                ing_words.append(child)
        
        return len(ing_words) >= 2
    
    def _get_element_text(self, token: 'Token', sent) -> str:
        """Get clean text representation of a coordinated element."""
        
        # Handle infinitives: include "to"
        if token.tag_ == 'VB' and token.i > 0:
            if hasattr(sent, 'start'):
                start_offset = sent.start
            else:
                start_offset = 0
            
            prev_idx = token.i - start_offset - 1
            if prev_idx >= 0:
                prev_token = sent[prev_idx]
                if prev_token and prev_token.text.lower() == 'to' and prev_token.pos_ == 'PART':
                    return f"to {token.text}"
        
        # For compound nouns, include the modifier
        if token.dep_ in ['compound', 'amod'] and token.head:
            return f"{token.text} {token.head.text}"
        
        return token.text

    def _calculate_tag_mismatch_evidence(self, patterns: List[str], context: Dict[str, Any]) -> float:
        """
        Calculate evidence score for tag mismatches in parallel structure.
        
        Enhanced with linguistic awareness of common acceptable patterns in technical writing.
        """
        
        unique_patterns = set(patterns)
        
        # All proper nouns are parallel (e.g., product names)
        if unique_patterns == {'PROPN'} or (len(unique_patterns) == 1 and 'PROPN' in unique_patterns):
            return 0.0
        
        base_evidence = 0.7
        
        # === NEW LINGUISTIC CLUE: Detect functional category mismatch ===
        def get_functional_category(pattern: str) -> str:
            """Map patterns to functional categories (ACTION, NOMINAL, or specific)."""
            if pattern.startswith('ACTION_'):
                return 'ACTION'
            if pattern.startswith('NOMINAL_'):
                return 'NOMINAL'
            return pattern 
        
        functional_categories = {get_functional_category(p) for p in patterns}
        
        # === ENHANCED LINGUISTIC CLUE: Common acceptable Noun/Action mismatch pattern ===
        # This pattern is semantically parallel and very common in technical writing.
        # Examples: "bypassing the tunnel and compromising data", "creating datasets and analysis"
        if 'NOMINAL' in functional_categories and 'ACTION' in functional_categories:
            # Strong reduction for this common, acceptable technical writing pattern
            # Increased from 0.4 to 0.6 to be more decisive in preventing false positives
            base_evidence -= 0.6  # Reduces from 0.7 to 0.1 (well below reporting threshold)
        
        # Assess severity of specific tag mismatch patterns
        has_verbs = any(p in ['VBG', 'VB', 'VBD', 'VBN', 'VBP', 'VBZ', 'TO+VB'] for p in unique_patterns)
        has_nouns = any(p in ['NN', 'NNS', 'NOUN', 'PREP_PHRASE'] for p in unique_patterns)
        has_prep = any(p == 'PREP' for p in unique_patterns)
        
        # Only increase evidence if NOT the NOMINAL+ACTION pattern (already handled above)
        if has_verbs and has_nouns and not ('NOMINAL' in functional_categories and 'ACTION' in functional_categories):
            base_evidence = 0.9
        elif has_prep and (has_verbs or has_nouns):
            base_evidence = 0.8
        elif 'TO+VB' in unique_patterns and 'VBG' in unique_patterns:
            base_evidence = 0.85
        
        # Context adjustments
        if context.get('content_type') in ['technical', 'academic', 'formal']:
            # In technical/formal writing, NOMINAL+ACTION patterns are even more acceptable
            if 'NOMINAL' in functional_categories and 'ACTION' in functional_categories:
                base_evidence -= 0.1  # Additional reduction in technical contexts
            else:
                base_evidence += 0.1  # Increase for other mismatches
        
        return max(0.0, min(base_evidence, 1.0))  # Clamp to [0.0, 1.0]

    def _create_parallel_structure_message(self, element_texts: List[str], patterns: List[str]) -> str:
        """Create a clear, helpful error message."""
        
        pattern_descriptions = {
            'VBG': 'gerund (-ing verb)',
            'NN': 'noun',
            'NNS': 'plural noun',
            'NOUN': 'noun',
            'PROPN': 'proper noun',
            'TO+VB': 'infinitive (to + verb)',
            'PREP': 'prepositional phrase',
            'PREP_PHRASE': 'prepositional phrase',
            'JJ': 'adjective',
            'VB': 'base verb'
        }
        
        # Create element descriptions
        elements_desc = []
        for text, pattern in zip(element_texts, patterns):
            desc = pattern_descriptions.get(pattern, pattern)
            elements_desc.append(f"'{text}' ({desc})")
        
        return f"Non-parallel structure: {', '.join(elements_desc)}. Use consistent grammatical forms."

    def _create_parallel_structure_suggestions(self, patterns: List[str]) -> List[str]:
        """Create helpful suggestions for fixing parallel structure."""
        
        unique_patterns = set(patterns)
        suggestions = []
        
        if 'VBG' in unique_patterns and 'TO+VB' in unique_patterns:
            suggestions.extend([
                "Convert all to gerunds: 'creating datasets, analyzing data, generating reports'",
                "Convert all to infinitives: 'to create datasets, to analyze data, to generate reports'"
            ])
        elif 'VBG' in unique_patterns and 'NN' in unique_patterns:
            suggestions.extend([
                "Convert all to gerunds: 'creating datasets, analyzing data, monitoring systems'",
                "Convert all to nouns: 'dataset creation, data analysis, system monitoring'"
            ])
        elif 'PREP' in unique_patterns:
            suggestions.append("Use consistent prepositional structure or remove prepositions")
        else:
            suggestions.append("Ensure all coordinated elements use the same grammatical form")
        
        return suggestions[:3]  # Return top 3

    def _get_conjunctions_patterns(self) -> Dict[str, Any]:
        """Get conjunctions patterns from YAML vocabulary service."""
        return self.vocabulary_service._load_yaml_file("conjunctions_patterns.yaml")