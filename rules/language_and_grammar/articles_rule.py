"""
Articles Rule
Based on IBM Style Guide: Checks incorrect and missing articles using linguistic analysis.
"""
from typing import List, Dict, Any, Optional
import pyinflect
from .base_language_rule import BaseLanguageRule
from .services.language_vocabulary_service import get_articles_vocabulary

try:
    from spacy.tokens import Doc, Token
except ImportError:
    Doc = None
    Token = None

try:
    from rule_enhancements import get_adapter, calculate_evidence, EvidencePriority
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False

class ArticlesRule(BaseLanguageRule):
    """Checks incorrect and missing articles using YAML-based configuration."""
    
    def __init__(self):
        super().__init__()
        self.vocabulary_service = get_articles_vocabulary()
        self.config = self.vocabulary_service.get_articles_config()
        self.adapter = get_adapter() if ENHANCEMENTS_AVAILABLE else None
    
    def _get_rule_type(self) -> str:
        return 'articles'

    def analyze(self, text: str, sentences: List[str], nlp=None, context=None) -> List[Dict[str, Any]]:
        """Analyze sentences for article errors using evidence-based scoring."""
        if context and context.get('block_type') in ['listing', 'literal', 'code_block', 'inline_code']:
            return []
        
        errors = []
        if not nlp:
            return errors

        content_classification = self._get_content_classification(text, context, nlp)
        if not self._should_apply_rule(self._get_rule_category(), content_classification):
            return errors

        doc = nlp(text)
        for i, sent in enumerate(doc.sents):
            for token in sent:
                if token.lower_ in ['a', 'an'] and token.i + 1 < len(doc):
                    next_token = self._get_next_non_punct_token(token, doc)
                    if next_token and self._is_incorrect_article_usage(token, next_token):
                        evidence_score = self._calculate_incorrect_article_evidence(token, next_token, sent, text, context)
                        
                        if evidence_score > self.config['evidence_thresholds']['min_threshold']:
                            errors.append(self._create_error(
                                sentence=sent.text, sentence_index=i,
                                message=self._get_contextual_message_incorrect(token, next_token, evidence_score),
                                suggestions=self._generate_smart_suggestions_incorrect(token, next_token, evidence_score, context),
                                severity='medium',
                                text=text,
                                context=context,
                                evidence_score=evidence_score,  # Your nuanced assessment
                                span=(token.idx, next_token.idx + len(next_token.text)),
                                flagged_text=f"{token.text} {next_token.text}"
                            ))
                
                is_candidate = self._is_missing_article_candidate(token, doc)
                
                if not is_candidate and self.adapter and ENHANCEMENTS_AVAILABLE:
                    block_type = context.get('block_type', '') if context else ''
                    if block_type in self.config['block_types']['lists']:
                        is_complete = self._is_complete_sentence_in_list(sent)
                        if (is_complete and token.pos_ == self.config['pos_tags']['noun'] and 
                            token.dep_ in [self.config['dependency_tags']['subject'], self.config['dependency_tags']['passive_subject']]):
                            has_article = any(child.dep_ == self.config['dependency_tags']['determiner'] or 
                                            child.pos_ == 'DET' for child in token.children)
                            if not has_article and token.i < 3:
                                is_candidate = True
                
                if is_candidate and not self._is_admonition_context(token, context):
                    evidence_score = self._calculate_missing_article_evidence(token, sent, text, context, content_classification)
                    
                    if self.adapter and ENHANCEMENTS_AVAILABLE:
                        block_type = context.get('block_type', '') if context else ''
                        if block_type in self.config['block_types']['lists']:
                            is_complete = self._is_complete_sentence_in_list(sent)
                            if is_complete:
                                final_score, should_flag, reasoning = calculate_evidence(
                                    base_score=evidence_score,
                                    factors=[
                                        (0.4, 'HIGH', 'Complete sentence in list requires articles'),
                                        (0.1, 'MEDIUM', 'Missing article detected')
                                    ],
                                    rule_type='articles_complete_sentence_list',
                                    context={'is_complete_sentence': True}
                                )
                                evidence_score = final_score
                    
                    if evidence_score > self.config['evidence_thresholds']['min_threshold']:
                        errors.append(self._create_error(
                            sentence=sent.text, sentence_index=i,
                            message=self._get_contextual_message_missing(token, evidence_score),
                            suggestions=self._generate_smart_suggestions_missing(token, evidence_score, context),
                            severity='low',
                            text=text,
                            context=context,
                            evidence_score=evidence_score,  # Your nuanced assessment
                            span=(token.idx, token.idx + len(token.text)),
                            flagged_text=token.text
                        ))
        return errors

    def _get_next_non_punct_token(self, token: Token, doc: Doc) -> Optional[Token]:
        """Get the next non-punctuation token, skipping markup characters."""
        for i in range(token.i + 1, min(len(doc), token.i + self.config['inline_list_thresholds']['max_lookahead'])):
            candidate = doc[i]
            if candidate.pos_ != self.config['pos_tags']['punctuation']:
                return candidate
        return None
    
    def _starts_with_vowel_sound(self, word: str) -> bool:
        """YAML-based phonetic analysis for article selection."""
        word_lower = word.lower().strip('`\'\"[](){}')
        
        if not word_lower:
            return False
        
        phonetics = self.vocabulary_service.get_articles_phonetics()
        
        consonant_sound_words = set()
        consonant_data = phonetics.get('consonant_sound_words', {})
        for category in consonant_data.values():
            if isinstance(category, list):
                consonant_sound_words.update(category)
        
        vowel_sound_words = set()
        vowel_data = phonetics.get('vowel_sound_words', {})
        for category in vowel_data.values():
            if isinstance(category, list):
                vowel_sound_words.update(category)
        
        if word_lower in consonant_sound_words:
            return False
        if word_lower in vowel_sound_words:
            return True
            
        if word_lower.startswith('uni') and len(word_lower) > 3:
            return False
        
        if word_lower.startswith('eu') and len(word_lower) > 2:
            return False
        
        return word_lower[0] in 'aeiou'

    def _is_incorrect_article_usage(self, article_token: Token, next_token: Token) -> bool:
        if 'attributeplaceholder' in next_token.text or 'asciidoclinkplaceholder' in next_token.text:
            return False
        starts_with_vowel = self._starts_with_vowel_sound(next_token.text)
        if article_token.lower_ == 'a' and starts_with_vowel: return True
        if article_token.lower_ == 'an' and not starts_with_vowel: return True
        return False

    def _is_uncountable(self, token: Token) -> bool:
        """Enhanced mass noun detection using YAML and pyinflect."""
        lemma = token.lemma_.lower()
        
        if self._is_technical_mass_noun(lemma):
            return True
        
        if self._is_mass_noun_context(token):
            return True
        
        plural_form = pyinflect.getInflection(token.lemma_, 'NNS')
        return plural_form is None

    def _is_technical_mass_noun(self, lemma: str) -> bool:
        """Check if word is a technical mass noun using YAML vocabulary."""
        phonetics_data = self.vocabulary_service.get_articles_phonetics()
        technical_mass_nouns = phonetics_data.get('technical_mass_nouns', {})
        
        for category, nouns_list in technical_mass_nouns.items():
            if isinstance(nouns_list, list) and lemma in nouns_list:
                return True
        
        return False

    def _is_mass_noun_context(self, token: Token) -> bool:
        """Context-aware mass noun detection using conservative patterns."""
        if token.i > 0:
            prev_token = token.doc[token.i - 1]
            
            if (prev_token.pos_ in (self.config['pos_tags']['noun'], 'ADJ') and 
                prev_token.lemma_.lower() in self.config['mass_noun_modifiers'] and
                token.dep_ in (self.config['dependency_tags']['compound'], 
                               self.config['dependency_tags']['direct_object'], 
                               self.config['dependency_tags']['prep_object'])):
                return True
        
        return False

    def _is_missing_article_candidate(self, token: Token, doc: Doc) -> bool:
        if not (token.pos_ == self.config['pos_tags']['noun'] and token.tag_ == 'NN' and not self._is_uncountable(token)):
            return False
            
        if any(child.dep_ in (self.config['dependency_tags']['determiner'], self.config['dependency_tags']['possessive']) 
               for child in token.children) or token.dep_ == self.config['dependency_tags']['possessive']:
            return False

        if self._is_hyphenated_compound_element(token, doc):
            return False

        if any(child.dep_ == self.config['dependency_tags']['compound'] for child in token.children):
            return False

        if token.i > 0:
            prev_token = doc[token.i - 1]
            if (prev_token.pos_ in (self.config['pos_tags']['noun'], self.config['pos_tags']['proper_noun'], 'ADJ') or 
                (prev_token.tag_ == self.config['pos_tags']['gerund'] and prev_token.dep_ in ('pcomp', self.config['dependency_tags']['adjectival_modifier'], self.config['dependency_tags']['compound']))):
                has_det = any(child.dep_ == self.config['dependency_tags']['determiner'] for child in prev_token.children)
                if not has_det:
                    return False

        if token.dep_ == self.config['dependency_tags']['prep_object'] and token.head.lemma_ == 'by':
            if token.lemma_ in self.config['adverbial_nouns']:
                return False

        if token.dep_ == self.config['dependency_tags']['compound']:
            return False
        if token.lemma_.lower() == 'step' and token.i + 1 < len(doc) and doc[token.i + 1].like_num:
            return False
            
        if token.dep_ == self.config['dependency_tags']['prep_object']:
            prep = token.head
            if prep.lemma_ in ['in', 'on', 'at', 'by', 'before', 'after'] and prep.head.pos_ == self.config['pos_tags']['verb']:
                if token.lemma_ in ['bold', 'italic', 'text']:
                    return False
        
        if self._is_technical_compound_phrase(token, doc):
            return False
            
        if self._is_technical_coordination(token, doc):
            return False
        
        if token.dep_ in (self.config['dependency_tags']['subject'], self.config['dependency_tags']['direct_object'], 
                          self.config['dependency_tags']['prep_object'], self.config['dependency_tags']['attribute'], 
                          self.config['dependency_tags']['noun_modifier']):
            if token.dep_ == self.config['dependency_tags']['noun_modifier']:
                has_coordination = any(sib.dep_ in (self.config['dependency_tags']['coordinating_conjunction'], 
                                                     self.config['dependency_tags']['coordination']) 
                                      for sib in token.head.children)
                if not has_coordination:
                    return False
            
            if token.i > 0 and ('attributeplaceholder' in doc[token.i - 1].text or 'asciidoclinkplaceholder' in doc[token.i - 1].text):
                return False
            return True
            
        return False

    def _is_technical_compound_phrase(self, token: Token, doc: Doc) -> bool:
        """Detect technical compound phrases where articles are typically omitted."""
        if token.lemma_.lower() not in self.config['truly_uncountable_technical_terms']:
            return False
        
        plural_form = pyinflect.getInflection(token.lemma_, 'NNS')
        if plural_form and len(plural_form) > 0:
            return False
        
        if token.dep_ == self.config['dependency_tags']['direct_object'] and token.head.pos_ == self.config['pos_tags']['verb']:
            if token.head.lemma_.lower() in self.config['countable_context_verbs']:
                return False
        
        has_technical_modifier = False
        for child in token.children:
            if (child.pos_ in ('ADJ', self.config['pos_tags']['noun']) and 
                child.dep_ in (self.config['dependency_tags']['adjectival_modifier'], 
                               self.config['dependency_tags']['noun_modifier'], 
                               self.config['dependency_tags']['compound']) and
                child.lemma_.lower() in self.config['technical_modifiers']):
                has_technical_modifier = True
                break
        
        if not has_technical_modifier:
            if token.dep_ == self.config['dependency_tags']['direct_object']:
                for sibling in token.head.children:
                    if (sibling != token and sibling.pos_ in ('ADJ', self.config['pos_tags']['noun']) and
                        sibling.dep_ in (self.config['dependency_tags']['adjectival_modifier'], 
                                        self.config['dependency_tags']['noun_modifier']) and
                        sibling.lemma_.lower() in self.config['technical_modifiers']):
                        has_technical_modifier = True
                        break
        
        if not has_technical_modifier:
            return False
        
        if token.dep_ == self.config['dependency_tags']['direct_object'] and token.head.pos_ == self.config['pos_tags']['verb']:
            if token.head.lemma_.lower() in self.config['countable_context_verbs']:
                return False
        
        return True

    def _is_technical_coordination(self, token: Token, doc: Doc) -> bool:
        """Detect when a technical term is part of coordination with other technical terms."""
        if token.lemma_.lower() not in self.config['truly_uncountable_technical_terms']:
            return False
        
        plural_form = pyinflect.getInflection(token.lemma_, 'NNS')
        if plural_form and len(plural_form) > 0:
            return False
        
        if token.dep_ == self.config['dependency_tags']['coordination']:
            head = token.head
            if head.lemma_.lower() in self.config['truly_uncountable_technical_terms']:
                return True
        
        for child in token.children:
            if (child.dep_ == self.config['dependency_tags']['coordination'] and 
                child.pos_ == self.config['pos_tags']['noun'] and
                child.lemma_.lower() in self.config['truly_uncountable_technical_terms']):
                return True
        
        return False

    def _is_admonition_context(self, token: Token, context: Optional[Dict[str, Any]]) -> bool:
        """Check if we're in an admonition context."""
        if not context:
            return False
        
        if context.get('block_type') == 'admonition':
            return True
        
        if context.get('next_block_type') == 'admonition':
            return True
        
        if token.lemma_.lower() in self.config['admonition_keywords']:
            return True
        
        return False

    def _calculate_incorrect_article_evidence(self, article_token, next_token, sentence, text: str, context: dict) -> float:
        """Calculate evidence score for incorrect a/an usage."""
        evidence_score = self.config['evidence_thresholds']['incorrect_base']
        evidence_score = self._apply_linguistic_clues_incorrect(evidence_score, article_token, next_token, sentence)
        evidence_score = self._apply_structural_clues_incorrect(evidence_score, article_token, next_token, context)
        evidence_score = self._apply_semantic_clues_incorrect(evidence_score, article_token, next_token, text, context)
        evidence_score = self._apply_feedback_clues_incorrect(evidence_score, article_token, next_token, context)
        
        return max(0.0, min(1.0, evidence_score))

    def _calculate_missing_article_evidence(self, noun_token, sentence, text: str, context: dict, content_classification: str) -> float:
        """Calculate evidence score for missing article before noun."""
        if noun_token.i > 0:
            doc = noun_token.doc
            prev_token = doc[noun_token.i - 1]
            if prev_token.dep_ in (self.config['dependency_tags']['compound'], self.config['dependency_tags']['adjectival_modifier']):
                compound_phrase = f"{prev_token.text.lower()} {noun_token.text.lower()}"
                if compound_phrase in self.config['uncountable_compounds']:
                    return 0.0
        
        if self._is_technical_value_or_keyword(noun_token, sentence):
            return 0.0
        
        if noun_token.i > 0 and noun_token.i < len(noun_token.doc) - 1:
            prev_char_is_quote = noun_token.doc[noun_token.i - 1].text == "'"
            next_char_is_quote = noun_token.doc[noun_token.i + 1].text == "'"
            if prev_char_is_quote and next_char_is_quote:
                return 0.0
        
        if noun_token.dep_ == self.config['dependency_tags']['prep_object']:
            if noun_token.lemma_ in self.config['abstract_tech_nouns']:
                is_unmodified = not any(child.dep_ == self.config['dependency_tags']['adjectival_modifier'] 
                                       for child in noun_token.children)
                if is_unmodified:
                    return 0.0

        evidence_score = self.config['evidence_thresholds']['missing_base']
        
        if self._is_inline_list_item(noun_token, sentence, text, context):
            evidence_score -= self.config['missing_linguistic_weights']['inline_list_reduction']
        
        if self._is_formal_verb_abstract_noun_construction(noun_token, sentence):
            evidence_score -= self.config['missing_linguistic_weights']['formal_abstract_reduction']
        
        evidence_score = self._apply_linguistic_clues_missing(evidence_score, noun_token, sentence)
        evidence_score = self._apply_structural_clues_missing(evidence_score, noun_token, context)
        evidence_score = self._apply_semantic_clues_missing(evidence_score, noun_token, text, context)
        evidence_score = self._apply_feedback_clues_missing(evidence_score, noun_token, context)
        
        return max(0.0, min(1.0, evidence_score))

    def _is_inline_list_item(self, noun_token, sentence, text: str, context: dict) -> bool:
        """Detect if a noun is the first word after an inline list marker."""
        import re
        
        block_type = context.get('block_type', 'paragraph') if context else 'paragraph'
        if block_type not in self.config['block_types']['inline_list_containers']:
            return False
        
        sentence_text = sentence.text
        noun_start_in_sentence = noun_token.idx - sentence.start_char
        
        if noun_start_in_sentence < 0:
            return False
        
        text_before_noun = sentence_text[:noun_start_in_sentence].strip()
        
        for pattern in self.config['inline_list_patterns']:
            if re.match(pattern, text_before_noun, re.IGNORECASE):
                remaining_text = re.sub(pattern, '', text_before_noun, count=1, flags=re.IGNORECASE).strip()
                word_count_between = len(remaining_text.split()) if remaining_text else 0
                
                if word_count_between <= self.config['inline_list_thresholds']['max_words_between']:
                    return True
        
        sentence_start = sentence.start_char
        lines_before_sentence = text[:sentence_start].split('\n')
        current_line_start = sentence_start - len(lines_before_sentence[-1]) if len(lines_before_sentence) > 1 else 0
        
        remaining_text = text[sentence_start:]
        next_newline = remaining_text.find('\n')
        current_line_end = sentence_start + (next_newline if next_newline != -1 else len(remaining_text))
        
        full_line = text[current_line_start:current_line_end].strip()
        
        for pattern in self.config['inline_list_patterns']:
            if re.match(pattern, full_line, re.IGNORECASE):
                marker_match = re.match(pattern, full_line, re.IGNORECASE)
                if marker_match:
                    text_after_marker = full_line[marker_match.end():].strip()
                    words_after_marker = text_after_marker.split()
                    
                    noun_text = noun_token.text.lower()
                    if len(words_after_marker) > 0 and words_after_marker[0].lower() == noun_text:
                        return True
                    elif len(words_after_marker) > 1 and words_after_marker[1].lower() == noun_text:
                        return True
                    elif len(words_after_marker) > 2 and words_after_marker[2].lower() == noun_text:
                        return True
        
        return False

    def _is_formal_verb_abstract_noun_construction(self, noun_token, sentence) -> bool:
        """Detect formal verb + abstract noun constructions where articles are omitted."""
        if not noun_token or not hasattr(noun_token, 'head') or not hasattr(noun_token, 'dep_'):
            return False
        
        noun_text = noun_token.text.lower()
        
        if noun_text not in self.config['abstract_nouns']:
            return False
        
        if noun_token.dep_ != self.config['dependency_tags']['direct_object']:
            return False
        
        head_token = noun_token.head
        
        if not hasattr(head_token, 'lemma_') or not hasattr(head_token, 'pos_'):
            return False
        
        if head_token.pos_ not in [self.config['pos_tags']['verb'], self.config['pos_tags']['auxiliary']]:
            return False
        
        head_lemma = head_token.lemma_.lower()
        head_text = head_token.text.lower()
        
        if head_lemma in self.config['formal_verbs'] or head_text in self.config['formal_verbs']:
            return True
        
        if (head_token.dep_ == self.config['dependency_tags']['auxiliary_passive'] or 
            head_token.dep_ == self.config['dependency_tags']['passive_subject']):
            for token in sentence:
                if (hasattr(token, 'lemma_') and 
                    token.lemma_.lower() in self.config['formal_verbs'] and
                    token.pos_ in [self.config['pos_tags']['verb'], self.config['pos_tags']['auxiliary']]):
                    return True
        
        return False

    def _apply_linguistic_clues_incorrect(self, evidence_score: float, article_token, next_token, sentence) -> float:
        """Apply linguistic analysis clues for incorrect a/an usage."""
        w = self.config['incorrect_linguistic_weights']
        word = next_token.text.lower()
        
        if word in self.config['common_words_vowel_sound'] or word in self.config['common_words_consonant_sound']:
            evidence_score += w['common_word_boost']
        
        if word.isupper() and len(word) <= 5:
            evidence_score += w['abbreviation_boost']
        
        if next_token.pos_ == self.config['pos_tags']['proper_noun']:
            evidence_score -= w['proper_noun_reduction']
        
        if next_token.ent_type_ in self.config['entity_types']['foreign']:
            evidence_score -= w['foreign_reduction']
        
        if article_token.dep_ == self.config['dependency_tags']['determiner'] and next_token.dep_ == self.config['dependency_tags']['subject']:
            evidence_score += w['subject_position_boost']
        elif article_token.dep_ == self.config['dependency_tags']['determiner'] and next_token.dep_ == self.config['dependency_tags']['direct_object']:
            evidence_score += w['object_position_boost']
        
        return evidence_score

    def _apply_structural_clues_incorrect(self, evidence_score: float, article_token, next_token, context: dict) -> float:
        """Apply document structure clues for incorrect a/an usage."""
        if not context:
            return evidence_score
        
        w = self.config['incorrect_structural_weights']
        block_type = context.get('block_type', 'paragraph')
        
        if block_type in self.config['block_types']['headings']:
            evidence_score += w['heading_boost']
        elif block_type == 'paragraph':
            evidence_score += w['paragraph_boost']
        elif block_type in self.config['block_types']['code']:
            evidence_score -= w['code_block_reduction']
        elif block_type == 'inline_code':
            evidence_score -= w['inline_code_reduction']
        elif block_type in self.config['block_types']['lists']:
            evidence_score -= w['list_reduction']
        elif block_type in self.config['block_types']['tables']:
            evidence_score -= w['table_reduction']
        elif block_type == 'admonition':
            admonition_type = context.get('admonition_type', '').upper()
            if admonition_type in self.config['admonition_types']['important']:
                evidence_score += w['important_admonition_boost']
        
        return evidence_score

    def _apply_semantic_clues_incorrect(self, evidence_score: float, article_token, next_token, text: str, context: dict) -> float:
        """Apply semantic and content-type clues for incorrect a/an usage."""
        if not context:
            return evidence_score
        
        w = self.config['incorrect_semantic_weights']
        content_type = context.get('content_type', 'general')
        
        if content_type == 'technical':
            evidence_score -= w['technical_reduction']
        elif content_type == 'academic':
            evidence_score += w['academic_boost']
        elif content_type == 'legal':
            evidence_score += w['legal_boost']
        elif content_type == 'marketing':
            evidence_score -= w['marketing_reduction']
        elif content_type == 'api':
            evidence_score -= w['api_reduction']
        
        audience = context.get('audience', 'general')
        if audience in self.config['content_types']['formal']:
            evidence_score += w['academic_audience_boost']
        elif audience in ['developer', 'technical']:
            evidence_score -= w['technical_audience_reduction']
        elif audience in ['beginner', 'student']:
            evidence_score += w['beginner_audience_boost']
        
        formal_indicators = self._count_formal_indicators(text)
        if formal_indicators > w['formality_high_threshold']:
            evidence_score += w['high_formality_boost']
        elif formal_indicators < w['formality_low_threshold']:
            evidence_score -= w['low_formality_reduction']
        
        return evidence_score

    def _apply_feedback_clues_incorrect(self, evidence_score: float, article_token, next_token, context: dict) -> float:
        """Apply feedback patterns for incorrect a/an usage."""
        w = self.config['feedback_weights']
        feedback_patterns = self._get_cached_feedback_patterns('articles')
        
        word = next_token.text.lower()
        article = article_token.text.lower()
        
        word_feedback = feedback_patterns.get('word_article_corrections', {})
        if word in word_feedback:
            expected_article = word_feedback[word]
            if article != expected_article:
                evidence_score += w['word_correction_boost']
            else:
                evidence_score -= w['word_accepted_reduction']
        
        error_pattern = f"{article} {word}"
        if error_pattern in feedback_patterns.get('common_article_corrections', set()):
            evidence_score += w['common_correction_boost']
        
        return evidence_score

    def _apply_linguistic_clues_missing(self, evidence_score: float, noun_token, sentence) -> float:
        """Apply linguistic analysis clues for missing articles."""
        w = self.config['missing_linguistic_weights']
        
        if noun_token.dep_ == self.config['dependency_tags']['subject']:
            evidence_score += w['subject_boost']
        elif noun_token.dep_ == self.config['dependency_tags']['direct_object']:
            evidence_score += w['direct_object_boost']
        elif noun_token.dep_ == self.config['dependency_tags']['prep_object']:
            evidence_score += w['prep_object_boost']
            
            has_adj_modifier = any(child.dep_ == self.config['dependency_tags']['adjectival_modifier'] 
                                  for child in noun_token.children)
            
            if not has_adj_modifier and noun_token.head.pos_ == 'ADP':
                has_adj_modifier = any(child.dep_ == self.config['dependency_tags']['adjectival_modifier'] 
                                      for child in noun_token.head.children)
            
            if has_adj_modifier:
                evidence_score -= w['modified_prep_object_reduction']
        
        elif noun_token.dep_ in [self.config['dependency_tags']['compound'], 'npadvmod']:
            evidence_score -= w['compound_reduction']
        
        if self._is_uncountable(noun_token):
            evidence_score -= w['uncountable_reduction']
        
        if self._is_technical_compound_phrase(noun_token, noun_token.doc):
            evidence_score -= w['technical_compound_reduction']
        
        if self._is_technical_coordination(noun_token, noun_token.doc):
            evidence_score -= w['technical_coordination_reduction']
        
        if self._has_specific_reference(noun_token):
            evidence_score += w['specific_reference_boost']
        elif self._has_generic_reference(noun_token):
            evidence_score += w['generic_reference_boost']
        
        if self._is_mass_noun_context(noun_token):
            evidence_score -= w['mass_noun_context_reduction']
        
        return evidence_score

    def _is_complete_sentence_in_list(self, sent) -> bool:
        """Determine if a list item contains a complete sentence or fragment."""
        if not sent or len(sent) == 0:
            return False
        
        root_verb = None
        has_subject = False
        
        for token in sent:
            if token.dep_ == 'ROOT':
                root_verb = token
            if token.dep_ in [self.config['dependency_tags']['subject'], self.config['dependency_tags']['passive_subject']]:
                has_subject = True
        
        if root_verb and root_verb.pos_ == self.config['pos_tags']['verb'] and root_verb.tag_ == self.config['pos_tags']['imperative'] and not has_subject:
            return False
        
        if root_verb and root_verb.pos_ in [self.config['pos_tags']['noun'], self.config['pos_tags']['proper_noun']]:
            return False
        
        if root_verb and root_verb.tag_ == self.config['pos_tags']['gerund']:
            return False
        
        if root_verb and root_verb.pos_ in [self.config['pos_tags']['verb'], self.config['pos_tags']['auxiliary']]:
            if root_verb.tag_ in self.config['pos_tags']['finite_verbs']:
                if has_subject or root_verb.tag_ == self.config['pos_tags']['finite_verbs'][-1]:
                    return True
        
        return False

    def _apply_structural_clues_missing(self, evidence_score: float, noun_token, context: dict) -> float:
        """Apply document structure clues for missing articles."""
        if not context:
            return evidence_score
        
        w = self.config['missing_structural_weights']
        block_type = context.get('block_type', 'paragraph')
        
        if block_type in self.config['block_types']['lists'][:2]:
            sent = noun_token.sent
            if len(sent) > 0:
                first_token = sent[0]
                if first_token.pos_ == self.config['pos_tags']['verb'] and first_token.tag_ == self.config['pos_tags']['imperative']:
                    evidence_score -= w['procedural_list_imperative_reduction']
        
        if block_type in self.config['block_types']['headings']:
            evidence_score -= w['heading_reduction']
        elif block_type == 'paragraph':
            evidence_score += w['paragraph_boost']
        elif block_type in self.config['block_types']['code']:
            evidence_score -= w['code_block_reduction']
        elif block_type == 'inline_code':
            evidence_score -= w['inline_code_reduction']
        elif block_type in self.config['block_types']['lists']:
            sent = noun_token.sent
            is_complete_sentence = self._is_complete_sentence_in_list(sent)
            
            if is_complete_sentence:
                evidence_score += w['complete_sentence_list_boost']
            else:
                evidence_score -= w['fragment_list_reduction']
                if context.get('list_depth', 1) > 1:
                    evidence_score -= w['nested_list_reduction']
        elif block_type in self.config['block_types']['tables']:
            evidence_score -= w['table_reduction']
        elif block_type == 'admonition':
            evidence_score -= w['admonition_reduction']
        
        return evidence_score

    def _apply_semantic_clues_missing(self, evidence_score: float, noun_token, text: str, context: dict) -> float:
        """Apply semantic and content-type clues for missing articles."""
        if not context:
            return evidence_score
        
        w = self.config['missing_semantic_weights']
        content_type = context.get('content_type', 'general')
        
        if content_type in self.config['content_types']['procedural']:
            if noun_token.dep_ == self.config['dependency_tags']['prep_object']:
                has_adj = any(child.dep_ == self.config['dependency_tags']['adjectival_modifier'] 
                             for child in noun_token.children)
                if has_adj:
                    evidence_score -= w['procedural_modified_pobj_reduction']
        
        if content_type == 'technical':
            evidence_score -= w['technical_reduction']
        elif content_type == 'api':
            evidence_score -= w['api_reduction']
        elif content_type == 'procedural':
            evidence_score -= w['procedural_reduction']
        elif content_type == 'academic':
            evidence_score += w['academic_boost']
        elif content_type == 'legal':
            evidence_score += w['legal_boost']
        elif content_type == 'marketing':
            evidence_score -= w['marketing_reduction']
        
        domain = context.get('domain', 'general')
        if domain in self.config['content_types']['domains_abbreviated']:
            evidence_score -= w['technical_domain_reduction']
        elif domain in ['documentation', 'tutorial']:
            evidence_score -= w['documentation_domain_reduction']
        
        audience = context.get('audience', 'general')
        if audience in ['developer', 'technical', 'expert']:
            evidence_score -= w['technical_audience_reduction']
        elif audience in self.config['content_types']['formal']:
            evidence_score += w['professional_audience_boost']
        elif audience in ['beginner', 'general']:
            evidence_score += w['general_audience_boost']
        
        if self._is_procedural_documentation(text):
            evidence_score -= w['procedural_docs_reduction']
        elif self._is_reference_documentation(text):
            evidence_score -= w['reference_docs_reduction']
        
        if self._has_high_technical_density(text):
            evidence_score -= w['high_technical_density_reduction']
        
        return evidence_score

    def _apply_feedback_clues_missing(self, evidence_score: float, noun_token, context: dict) -> float:
        """Apply feedback patterns for missing articles."""
        w = self.config['feedback_weights']
        feedback_patterns = self._get_cached_feedback_patterns('articles')
        
        noun = noun_token.text.lower()
        
        if noun in feedback_patterns.get('commonly_no_article_nouns', set()):
            evidence_score -= w['no_article_accepted_reduction']
        
        if noun in feedback_patterns.get('commonly_missing_article_nouns', set()):
            evidence_score += w['missing_article_boost']
        
        block_type = context.get('block_type', 'paragraph') if context else 'paragraph'
        context_patterns = feedback_patterns.get(f'{block_type}_article_patterns', {})
        
        if noun in context_patterns.get('acceptable_without_article', set()):
            evidence_score -= w['context_acceptable_reduction']
        elif noun in context_patterns.get('needs_article', set()):
            evidence_score += w['context_needs_boost']
        
        return evidence_score

    def _has_specific_reference(self, noun_token) -> bool:
        """Check if noun refers to something specific."""
        modifiers = [child.text.lower() for child in noun_token.children]
        
        if any(mod in self.config['demonstratives'] for mod in modifiers):
            return True
        
        if any(child.tag_ in self.config['pos_tags']['superlative'] for child in noun_token.children):
            return True
        
        if any(child.like_num and any(ord_word in child.text.lower() for ord_word in self.config['ordinal_words']) 
               for child in noun_token.children):
            return True
        
        return False

    def _has_generic_reference(self, noun_token) -> bool:
        """Check if noun refers to something generic."""
        if noun_token.dep_ == self.config['dependency_tags']['subject'] and noun_token.head.lemma_ in self.config['copula_verbs']:
            return True
        
        if any(child.lemma_ in self.config['comparison_lemmas'] for child in noun_token.ancestors):
            return True
        
        return False

    def _count_formal_indicators(self, text: str) -> int:
        """Count indicators of formal writing style."""
        text_lower = text.lower()
        return sum(1 for indicator in self.config['formal_indicators'] if indicator in text_lower)

    def _get_contextual_message_incorrect(self, article_token, next_token, evidence_score: float) -> str:
        """Generate context-aware error messages for incorrect a/an usage."""
        correct_article = 'an' if self._starts_with_vowel_sound(next_token.text) else 'a'
        thresholds = self.config['evidence_thresholds']
        
        if evidence_score > thresholds['incorrect_high']:
            return f"Incorrect article: Use '{correct_article}' before '{next_token.text}' (phonetic rule)."
        elif evidence_score > thresholds['incorrect_medium']:
            return f"Consider using '{correct_article}' before '{next_token.text}' for standard pronunciation."
        else:
            return f"Article usage: '{correct_article}' is typically used before '{next_token.text}'."

    def _get_contextual_message_missing(self, noun_token, evidence_score: float) -> str:
        """Generate context-aware error messages for missing articles."""
        thresholds = self.config['evidence_thresholds']
        
        if evidence_score > thresholds['missing_high']:
            return f"Missing article: Singular noun '{noun_token.text}' typically requires an article (a/an/the)."
        elif evidence_score > thresholds['missing_medium']:
            return f"Consider adding an article before '{noun_token.text}' for clarity."
        else:
            return f"Article usage: '{noun_token.text}' might benefit from an article depending on context."

    def _generate_smart_suggestions_incorrect(self, article_token, next_token, evidence_score: float, context: dict) -> List[str]:
        """Generate context-aware suggestions for incorrect a/an usage."""
        
        suggestions = []
        correct_article = 'an' if self._starts_with_vowel_sound(next_token.text) else 'a'
        
        # Base correction
        suggestions.append(f"Change '{article_token.text} {next_token.text}' to '{correct_article} {next_token.text}'.")
        
        # Explanation based on evidence
        if evidence_score > 0.7:
            suggestions.append(f"'{next_token.text}' starts with a {'vowel' if self._starts_with_vowel_sound(next_token.text) else 'consonant'} sound, requiring '{correct_article}'.")
        
        # Context-specific advice
        if context:
            content_type = context.get('content_type', 'general')
            if content_type in ['academic', 'legal', 'professional']:
                suggestions.append("Correct article usage is important in formal writing.")
            elif content_type == 'technical':
                suggestions.append("While technical writing is concise, article accuracy aids readability.")
        
        return suggestions

    def _generate_smart_suggestions_missing(self, noun_token, evidence_score: float, context: dict) -> List[str]:
        """Generate context-aware suggestions for missing articles."""
        suggestions = []
        thresholds = self.config['evidence_thresholds']
        
        if self._has_specific_reference(noun_token):
            suggestions.append(f"Consider adding 'the' before '{noun_token.text}' for specific reference.")
        else:
            suggestions.append(f"Consider adding 'a/an/the' before '{noun_token.text}' as appropriate.")
        
        if context:
            content_type = context.get('content_type', 'general')
            block_type = context.get('block_type', 'paragraph')
            
            if content_type == 'technical' and block_type in self.config['block_types']['lists']:
                suggestions.append("Technical lists often omit articles, but consider your style guide.")
            elif content_type in self.config['content_types']['formal']:
                suggestions.append("Formal writing typically includes articles for completeness.")
            elif content_type == 'procedural':
                suggestions.append("Instructions may omit articles for brevity, but clarity is important.")
        
        if evidence_score < thresholds['missing_low']:
            suggestions.append("This usage may be acceptable in your context, depending on style preferences.")
        elif evidence_score > 0.7:
            suggestions.append("Adding an article would improve grammatical completeness.")
        
        return suggestions

    def _is_technical_value_or_keyword(self, noun_token, sentence) -> bool:
        """Detect technical keywords and configuration values that don't need articles."""
        doc = noun_token.doc
        noun_text = noun_token.text.lower()
        
        if noun_text not in self.config['technical_keywords']:
            return False
        
        if noun_token.i > 0 and noun_token.i < len(doc) - 1:
            prev_token = doc[noun_token.i - 1]
            next_token = doc[noun_token.i + 1]
            
            if (prev_token.text in self.config['technical_value_detection']['markup_chars']['open'] and 
                next_token.text in self.config['technical_value_detection']['markup_chars']['close']):
                return True
        
        lookback_limit = min(self.config['technical_value_detection']['lookback_limit'], noun_token.i)
        for offset in range(1, lookback_limit + 1):
            prev_token = doc[noun_token.i - offset]
            prev_text = prev_token.text.lower()
            
            if prev_text in self.config['value_setting_patterns']:
                return True
            
            if offset < lookback_limit:
                prev_prev_token = doc[noun_token.i - offset - 1]
                two_word_phrase = f"{prev_prev_token.text.lower()} {prev_text}"
                if two_word_phrase in self.config['value_setting_patterns']:
                    return True
            
            if offset < lookback_limit - 1:
                prev_prev_token = doc[noun_token.i - offset - 1]
                prev_prev_prev_token = doc[noun_token.i - offset - 2]
                three_word_phrase = f"{prev_prev_prev_token.text.lower()} {prev_prev_token.text.lower()} {prev_text}"
                if any(pattern in three_word_phrase for pattern in ['is set to', 'was set to', 'are set to', 'were set to']):
                    return True
        
        if noun_token.dep_ in [self.config['dependency_tags']['attribute'], self.config['dependency_tags']['attribute_complement']]:
            head = noun_token.head
            if head.lemma_ in self.config['copula_verbs']:
                return True
        
        if noun_token.i < len(doc) - 3:
            if (doc[noun_token.i + 1].text == '(' and doc[noun_token.i + 3].text == ')'):
                paren_word = doc[noun_token.i + 2].text.lower()
                if paren_word in self.config['technical_keywords']:
                    return True
        
        return False

    def _is_hyphenated_compound_element(self, token, doc) -> bool:
        """
        Check if token is part of a hyphenated compound - surgical precision approach.
        
        This prevents false positives for cases like:
        - "fault" in "fault-tolerant" 
        - "real" in "real-time"
        - "state" in "state-of-the-art"
        
        Uses surgical detection to avoid over-flagging.
        """
        
        # METHOD 1: Check if token itself contains hyphen (compound word)
        if '-' in token.text:
            return True
        
        # METHOD 2: Check ONLY immediately adjacent tokens for direct hyphen connection
        token_idx = token.i
        
        # Check next token - must be exactly a hyphen or start with hyphen
        if token_idx + 1 < len(doc):
            next_token = doc[token_idx + 1]
            if next_token.text == '-':  # Exact hyphen token
                return True
        
        # Check previous token - must be exactly a hyphen or end with hyphen  
        if token_idx > 0:
            prev_token = doc[token_idx - 1]
            if prev_token.text == '-':  # Exact hyphen token
                return True
        
        # METHOD 3: SURGICAL check for known patterns using YAML
        # Only check if token is the first part of a known hyphenated compound
        if self._is_first_part_of_known_hyphenated(token, doc):
            return True
        
        return False

    def _is_first_part_of_known_hyphenated(self, token, doc) -> bool:
        """
        SURGICAL: Check if token is the first part of a known hyphenated compound.
        
        Only returns True if the actual text sequence in the document matches 
        a known hyphenated pattern, not just a prefix match.
        """
        phonetics_data = self.vocabulary_service.get_articles_phonetics()
        hyphenated_patterns = phonetics_data.get('hyphenated_compounds', {})
        
        if not hyphenated_patterns:
            return False
        
        # Build the actual text sequence starting from this token
        token_idx = token.i
        max_pattern_length = 5  # Reasonable limit for hyphenated compounds
        
        # Extract text sequence from this token forward
        sequence_tokens = []
        for i in range(token_idx, min(len(doc), token_idx + max_pattern_length)):
            sequence_tokens.append(doc[i].text)
        
        sequence_text = " ".join(sequence_tokens).lower()
        
        # Check if this sequence matches any known hyphenated pattern
        for category, patterns in hyphenated_patterns.items():
            if isinstance(patterns, list):
                for pattern in patterns:
                    if isinstance(pattern, str):
                        pattern_lower = pattern.lower()
                        # Check if the pattern is contained in our sequence
                        # and starts with our token
                        if (pattern_lower in sequence_text and 
                            sequence_text.startswith(token.text.lower()) and
                            '-' in pattern):
                            return True
        
        return False
