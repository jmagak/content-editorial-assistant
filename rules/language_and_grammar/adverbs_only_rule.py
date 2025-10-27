"""
Adverbs - only Rule
Based on IBM Style Guide: Checks ambiguous placement of "only".
"""
from typing import List, Dict, Any
from .base_language_rule import BaseLanguageRule
from .services.language_vocabulary_service import get_adverbs_only_vocabulary

try:
    from spacy.tokens import Doc
except ImportError:
    Doc = None

class AdverbsOnlyRule(BaseLanguageRule):
    """Checks for ambiguous placement of "only" using evidence-based scoring."""
    
    def __init__(self):
        super().__init__()
        self.vocab_service = get_adverbs_only_vocabulary()
        self.config = self.vocab_service.get_adverbs_only_config()
    
    def _get_rule_type(self) -> str:
        return 'adverbs_only'

    def analyze(self, text: str, sentences: List[str], nlp=None, context=None) -> List[Dict[str, Any]]:
        """Analyze sentences for ambiguous placement of "only"."""
        if context and context.get('block_type') in ['listing', 'literal', 'code_block', 'inline_code']:
            return []
        
        errors = []
        if not nlp:
            return errors

        doc = nlp(text)
        for i, sent in enumerate(doc.sents):
            for token in sent:
                if token.lemma_ == 'only':
                    evidence_score = self._calculate_only_ambiguity_evidence(token, sent, text, context)
                    
                    if evidence_score > self.config['evidence_thresholds']['min_threshold']:
                        errors.append(self._create_error(
                            sentence=sent.text,
                            sentence_index=i,
                            message=self._get_contextual_message(token, evidence_score),
                            suggestions=self._generate_smart_suggestions(token, evidence_score, context),
                            severity='low',
                            text=text,
                            context=context,
                            evidence_score=evidence_score,
                            span=(token.idx, token.idx + len(token.text)),
                            flagged_text=token.text
                        ))
        return errors

    def _calculate_only_ambiguity_evidence(self, token, sentence, text: str, context: dict) -> float:
        """Calculate evidence score for "only" placement ambiguity."""
        if token.lemma_ != 'only':
            return 0.0
        
        # === SURGICAL GUARD: Clear restrictive patterns ===
        # Patterns like "only if", "only when" are UNAMBIGUOUS and should not be flagged
        next_token = token.nbor(1) if token.i < len(token.doc) - 1 else None
        if next_token and next_token.lemma_ in self.config['conditional_words']:
            return 0.0  # EXIT EARLY: This is a clear, unambiguous pattern
        
        evidence_score = self.config['evidence_thresholds']['base_score']
        evidence_score = self._apply_linguistic_clues_only(evidence_score, token, sentence)
        evidence_score = self._apply_structural_clues_only(evidence_score, token, context)
        evidence_score = self._apply_semantic_clues_only(evidence_score, token, text, context)
        evidence_score = self._apply_feedback_clues_only(evidence_score, token, context)
        
        return max(0.0, min(1.0, evidence_score))

    def _apply_linguistic_clues_only(self, evidence_score: float, token, sentence) -> float:
        """Apply linguistic analysis for "only" placement."""
        w = self.config['linguistic_weights']
        
        if self._detect_classic_noun_only_verb_pattern(token, sentence):
            evidence_score += w['classic_ambiguity_boost']
        
        if self._detect_clear_only_modification_pattern(token, sentence):
            evidence_score -= w['clear_modification_reduction']
        
        sent_tokens = list(sentence)
        token_position = next((i for i, t in enumerate(sent_tokens) if t.i == token.i), None)
        
        if token_position is not None:
            sentence_length = len(sent_tokens)
            relative_position = token_position / sentence_length if sentence_length > 0 else 0
            
            if relative_position < self.config['position_thresholds']['beginning']:
                evidence_score -= w['position_beginning_reduction']
            elif (self.config['position_thresholds']['middle_start'] < relative_position < 
                  self.config['position_thresholds']['middle_end']):
                evidence_score += w['position_middle_boost']
            elif relative_position > self.config['position_thresholds']['end']:
                next_token = token.nbor(1) if token.i < len(token.doc) - 1 else None
                if next_token and next_token.pos_ in self.config['pos_tags']['nouns'] + self.config['pos_tags']['determiners']:
                    evidence_score -= w['position_end_det_noun_reduction']
                else:
                    evidence_score += w['position_end_other_boost']
        
        only_head = token.head
        if token.dep_ == 'advmod' and only_head.pos_ in self.config['pos_tags']['nouns']:
            evidence_score -= w['advmod_noun_reduction']
        elif token.dep_ == 'advmod' and only_head.pos_ in self.config['pos_tags']['verbs']:
            evidence_score += w['advmod_verb_boost']
        elif token.dep_ in ['ROOT', 'nsubj']:
            evidence_score += w['unclear_dependency_boost']
        
        prev_token = token.nbor(-1) if token.i > 0 else None
        next_token = token.nbor(1) if token.i < len(token.doc) - 1 else None
        
        if prev_token:
            if prev_token.lemma_ in self.config['articles']:
                evidence_score -= w['article_prev_reduction']
            elif prev_token.pos_ in self.config['pos_tags']['verbs']:
                evidence_score += w['verb_prev_boost']
            
            if hasattr(prev_token, 'morph') and prev_token.morph:
                morph_str = str(prev_token.morph)
                if self.config['verb_forms']['finite'] in morph_str:
                    evidence_score += w.get('finite_verb_prev_boost', 0.05)
                elif self.config['verb_forms']['infinitive'] in morph_str:
                    evidence_score -= w.get('infinitive_prev_reduction', 0.05)
        
        if next_token:
            if next_token.pos_ in self.config['pos_tags']['nouns']:
                evidence_score -= w['noun_next_reduction']
            elif next_token.pos_ in self.config['pos_tags']['determiners']:
                evidence_score -= w['det_next_reduction']
            elif next_token.pos_ in self.config['pos_tags']['prepositions'] + self.config['pos_tags']['particles']:
                evidence_score -= w['prep_next_reduction']
            elif next_token.lemma_ in self.config['conditional_words']:
                evidence_score -= w['conditional_next_reduction']
            elif next_token.pos_ in self.config['pos_tags']['verbs']:
                evidence_score += w['verb_next_boost']
            
            if hasattr(next_token, 'morph') and next_token.morph:
                morph_str = str(next_token.morph)
                if self.config['number_morphology']['singular'] in morph_str and next_token.pos_ == 'NOUN':
                    evidence_score -= w.get('singular_noun_next_reduction', 0.05)
                elif self.config['number_morphology']['plural'] in morph_str and next_token.pos_ == 'NOUN':
                    evidence_score -= w.get('plural_noun_next_reduction', 0.1)
                if self.config['definiteness_morphology']['definite'] in morph_str:
                    evidence_score -= w.get('definite_reduction', 0.05)
            
            if hasattr(next_token, 'ent_type_') and next_token.ent_type_:
                if next_token.ent_type_ in self.config['clear_entity_types']:
                    evidence_score -= w['entity_reduction']
        
        if hasattr(token, 'tag_'):
            if token.tag_ == 'RB':
                evidence_score -= w.get('adverb_tag_reduction', 0.1)
            elif token.tag_ == 'JJ':
                evidence_score += w.get('adjective_tag_boost', 0.2)
        
        sentence_complexity = len([t for t in sentence if t.pos_ in self.config['pos_tags']['verbs']])
        if sentence_complexity == self.config['complexity_thresholds']['simple']:
            evidence_score -= w['single_verb_reduction']
        elif sentence_complexity > self.config['complexity_thresholds']['complex']:
            evidence_score += w['multi_verb_boost']
        
        if any(child.dep_ in self.config['complexity_dependencies'] for child in sentence):
            evidence_score += w['conjunction_boost']
        
        return evidence_score

    def _apply_structural_clues_only(self, evidence_score: float, token, context: dict) -> float:
        """Apply document structure clues for "only" placement."""
        if not context:
            return evidence_score
        
        w = self.config['structural_weights']
        block_type = context.get('block_type', 'paragraph')
        
        if block_type in ['code_block', 'literal_block']:
            evidence_score -= w['code_block_reduction']
        elif block_type == 'inline_code':
            evidence_score -= w['inline_code_reduction']
        elif block_type in ['table_cell', 'table_header']:
            evidence_score -= w['table_reduction']
        elif block_type == 'heading':
            heading_level = context.get('block_level', 1)
            if heading_level <= 2:
                evidence_score -= w['h1_h2_reduction']
            else:
                evidence_score -= w['h3_plus_reduction']
        elif block_type in ['ordered_list_item', 'unordered_list_item']:
            evidence_score -= w['list_reduction']
            if context.get('list_depth', 1) > 1:
                evidence_score -= w['nested_list_reduction']
        elif block_type == 'admonition':
            admonition_type = context.get('admonition_type', '').upper()
            if admonition_type in self.config['admonition_types']:
                evidence_score -= w['admonition_reduction']
        
        return evidence_score

    def _apply_semantic_clues_only(self, evidence_score: float, token, text: str, context: dict) -> float:
        """Apply semantic and content-type clues for "only" usage."""
        if not context:
            return evidence_score
        
        w = self.config['semantic_weights']
        content_type = context.get('content_type', 'general')
        
        if self._is_api_documentation(text):
            evidence_score -= w['api_docs_reduction']
        elif self._is_procedural_documentation(text):
            evidence_score -= w['procedural_docs_reduction']
        elif self._is_reference_documentation(text):
            evidence_score -= w['reference_docs_reduction']
        elif self._is_troubleshooting_documentation(text):
            evidence_score -= w['troubleshooting_reduction']
        elif self._is_installation_documentation(text):
            evidence_score -= w['installation_reduction']
        elif self._is_configuration_documentation(text):
            evidence_score -= w['configuration_reduction']
        elif content_type == 'technical':
            if not self._detect_classic_noun_only_verb_pattern(token, token.sent):
                evidence_score -= w['technical_content_reduction']
                if self._has_technical_context_words(token.sent.text, distance=5):
                    evidence_score -= w['technical_content_reduction']
        elif content_type == 'procedural':
            evidence_score -= w['procedural_content_reduction']
        elif content_type == 'marketing':
            evidence_score += w['marketing_boost']
        elif content_type == 'legal':
            evidence_score -= w['legal_reduction']
        elif content_type == 'academic':
            evidence_score -= w['academic_reduction']
        
        domain = context.get('domain', 'general')
        if domain in ['software', 'engineering', 'devops']:
            evidence_score -= w['technical_domain_reduction']
        elif domain in ['api', 'documentation']:
            evidence_score -= w['api_domain_reduction']
        
        audience = context.get('audience', 'general')
        if audience in ['developer', 'expert']:
            evidence_score -= w['expert_audience_reduction']
        elif audience in ['beginner', 'general']:
            evidence_score += w['general_audience_boost']
        
        sentence_text = token.sent.text.lower()
        
        if any(pattern in sentence_text for pattern in self.config['clear_patterns']):
            evidence_score -= w['clear_pattern_reduction']
        
        if any(pattern in sentence_text for pattern in self.config['ambiguous_patterns']):
            evidence_score += w['ambiguous_pattern_boost']
        
        return evidence_score

    def _apply_feedback_clues_only(self, evidence_score: float, token, context: dict) -> float:
        """Apply feedback patterns for "only" placement."""
        w = self.config['feedback_weights']
        feedback_patterns = self._get_cached_feedback_patterns('adverbs_only')
        
        sentence_text = token.sent.text.lower()
        
        accepted_patterns = feedback_patterns.get('accepted_only_patterns', set())
        for pattern in accepted_patterns:
            if pattern in sentence_text:
                evidence_score -= w['accepted_pattern_reduction']
                break
        
        flagged_patterns = feedback_patterns.get('flagged_only_patterns', set())
        for pattern in flagged_patterns:
            if pattern in sentence_text:
                evidence_score += w['flagged_pattern_boost']
                break
        
        sent_tokens = list(token.sent)
        token_position = next((i for i, t in enumerate(sent_tokens) if t.i == token.i), None)
        
        if token_position is not None:
            position_feedback = feedback_patterns.get('only_position_feedback', {})
            if token_position < 2:
                acceptance_rate = position_feedback.get('beginning', w['position_beginning_acceptance'])
            elif token_position > len(sent_tokens) - 3:
                acceptance_rate = position_feedback.get('end', w['position_end_acceptance'])
            else:
                acceptance_rate = position_feedback.get('middle', w['position_middle_acceptance'])
            
            if acceptance_rate > 0.7:
                evidence_score -= w['high_acceptance_reduction']
            elif acceptance_rate < 0.4:
                evidence_score += w['low_acceptance_boost']
        
        return evidence_score

    def _get_contextual_message(self, token, evidence_score: float) -> str:
        """Generate context-aware error messages."""
        if self._detect_classic_noun_only_verb_pattern(token, token.sent):
            return "Classic ambiguous placement: 'only' could modify either the subject or the verb, creating confusion."
        
        thresholds = self.config['evidence_thresholds']
        if evidence_score > thresholds['high_ambiguity']:
            return "The placement of 'only' in this sentence is ambiguous and may confuse readers."
        elif evidence_score > thresholds['moderate_ambiguity']:
            return "Consider reviewing the placement of 'only' to ensure clarity."
        else:
            return "The word 'only' could potentially be clearer depending on its intended meaning."

    def _generate_smart_suggestions(self, token, evidence_score: float, context: dict) -> List[str]:
        """Generate context-aware suggestions."""
        suggestions = []
        
        if self._detect_classic_noun_only_verb_pattern(token, token.sent):
            sent_tokens = list(token.sent)
            token_idx = next((i for i, t in enumerate(sent_tokens) if t.i == token.i), None)
            
            if token_idx and token_idx > 0:
                prev_token = sent_tokens[token_idx - 1]
                next_token = sent_tokens[token_idx + 1] if token_idx < len(sent_tokens) - 1 else None
                
                suggestions.append(f"AMBIGUOUS PLACEMENT: Does 'only' modify '{prev_token.text}' or the verb?")
                suggestions.append(f"If you mean 'only {prev_token.text}' (not others): Move to 'Only {prev_token.text}...'")
                if next_token:
                    suggestions.append(f"If you mean 'only {next_token.text}' (limited action): Move to '{prev_token.text} can only {next_token.text}...'")
                suggestions.append("Consider rephrasing entirely to eliminate ambiguity.")
                return suggestions
        
        sent_tokens = list(token.sent)
        token_position = next((i for i, t in enumerate(sent_tokens) if t.i == token.i), None)
        
        suggestions.append("Place 'only' immediately before the word or phrase it modifies.")
        
        if token_position is not None:
            if token_position > len(sent_tokens) // 2:
                suggestions.append("Consider moving 'only' earlier in the sentence for clarity.")
            
            next_token = token.nbor(1) if token.i < len(token.doc) - 1 else None
            if next_token and next_token.pos_ in self.config['pos_tags']['nouns'] + self.config['pos_tags']['verbs']:
                suggestions.append(f"If 'only' modifies '{next_token.text}', consider: 'only {next_token.text}'.")
        
        if context:
            content_type = context.get('content_type', 'general')
            if content_type == 'technical':
                suggestions.append("In technical writing, be precise about what 'only' restricts or limits.")
            elif content_type == 'procedural':
                suggestions.append("In instructions, clarify exactly what limitation 'only' expresses.")
        
        thresholds = self.config['evidence_thresholds']
        if evidence_score > 0.7:
            suggestions.append("Consider rephrasing the sentence entirely to avoid ambiguity.")
        elif evidence_score < thresholds['low_ambiguity']:
            suggestions.append("The current placement may be acceptable, but review for your intended meaning.")
        
        return suggestions
    
    def _detect_classic_noun_only_verb_pattern(self, token, sentence) -> bool:
        """Detect the classic ambiguous [Noun] [only] [Verb] pattern."""
        sent_tokens = list(sentence)
        token_idx = next((i for i, t in enumerate(sent_tokens) if t.i == token.i), None)
        
        if token_idx is None or token_idx == 0:
            return False
        
        prev_token = sent_tokens[token_idx - 1] if token_idx > 0 else None
        next_token = sent_tokens[token_idx + 1] if token_idx < len(sent_tokens) - 1 else None
        
        if (prev_token and next_token and 
            prev_token.pos_ in self.config['pos_tags']['nouns'] and
            next_token.pos_ in self.config['pos_tags']['verbs']):
            
            if not (prev_token.dep_ in self.config['compound_dependencies'] and 
                    next_token.dep_ in self.config['compound_dependencies']):
                if prev_token.dep_ in self.config['subject_dependencies'] or self._is_likely_subject(prev_token):
                    return True
        
        if (prev_token and token_idx + 2 < len(sent_tokens) and
            prev_token.pos_ in self.config['pos_tags']['nouns']):
            
            if (next_token and (next_token.pos_ == 'AUX' or next_token.lemma_ in self.config['modal_verbs'])):
                verb_token = sent_tokens[token_idx + 2]
                if verb_token.pos_ in self.config['pos_tags']['verbs']:
                    if prev_token.dep_ in self.config['subject_dependencies'] or self._is_likely_subject(prev_token):
                        return True
        
        if (prev_token and next_token and
            prev_token.pos_ in self.config['pos_tags']['nouns'] and
            next_token.lemma_ == 'have'):
            
            if prev_token.dep_ in self.config['subject_dependencies'] or self._is_likely_subject(prev_token):
                return True
        
        return False
    
    def _is_likely_subject(self, token) -> bool:
        """Check if token is likely functioning as sentence subject."""
        if token.dep_ in self.config['subject_dependencies']:
            return True
        
        sent_tokens = list(token.sent)
        for sent_token in sent_tokens:
            if sent_token.pos_ in self.config['pos_tags']['nouns']:
                return sent_token.i == token.i
        
        if token.head.pos_ in self.config['pos_tags']['verbs']:
            for child in token.head.children:
                if child.i == token.i and child.dep_ in self.config['subject_dependencies']:
                    return True
        
        return False
    
    def _detect_clear_only_modification_pattern(self, token, sentence) -> bool:
        """Detect clear "only [determiner/noun phrase]" patterns."""
        next_token = token.nbor(1) if token.i < len(token.doc) - 1 else None
        if not next_token:
            return False
        
        if next_token.pos_ in self.config['pos_tags']['determiners']:
            next_next_token = next_token.nbor(1) if next_token.i < len(next_token.doc) - 1 else None
            if next_next_token and next_next_token.pos_ in self.config['pos_tags']['nouns'] + self.config['pos_tags']['adjectives']:
                return True
        
        if next_token.pos_ in self.config['pos_tags']['numbers'] or next_token.like_num:
            next_next_token = next_token.nbor(1) if next_token.i < len(next_token.doc) - 1 else None
            if next_next_token and next_next_token.pos_ in self.config['pos_tags']['nouns']:
                return True
        
        if next_token.pos_ in self.config['pos_tags']['adjectives']:
            next_next_token = next_token.nbor(1) if next_token.i < len(next_token.doc) - 1 else None
            if next_next_token and next_next_token.pos_ in self.config['pos_tags']['nouns']:
                return True
        
        if next_token.pos_ in self.config['pos_tags']['nouns']:
            next_next_token = next_token.nbor(1) if next_token.i < len(next_token.doc) - 1 else None
            if not (next_next_token and next_next_token.pos_ in self.config['pos_tags']['verbs']):
                return True
        
        return False
