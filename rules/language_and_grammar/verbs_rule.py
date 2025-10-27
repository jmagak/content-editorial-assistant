"""
Verbs Rule
Based on IBM Style Guide: Checks verb tense, voice, agreement, and nominalization.
"""
from typing import List, Dict, Any, Optional
import pyinflect
from .services.language_vocabulary_service import get_verbs_vocabulary

from .base_language_rule import BaseLanguageRule
from .passive_voice_analyzer import PassiveVoiceAnalyzer, ContextType, PassiveConstruction

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

class VerbsRule(BaseLanguageRule):
    """Checks passive voice, verb tense, agreement, and nominalization."""
    
    def __init__(self):
        super().__init__()
        self.passive_analyzer = PassiveVoiceAnalyzer()
        self.vocabulary_service = get_verbs_vocabulary()
        self.config = self.vocabulary_service.get_verbs_config()
        self.adapter = get_adapter() if ENHANCEMENTS_AVAILABLE else None
    
    def _get_rule_type(self) -> str:
        return 'verbs'

    def analyze(self, text: str, sentences: List[str], nlp=None, context=None) -> List[Dict[str, Any]]:
        if context and context.get('block_type') in ['listing', 'literal', 'code_block', 'inline_code']:
            return []
        
        errors = []
        if not nlp or not self.passive_analyzer:
            return errors

        for i, sent_text in enumerate(sentences):
            if not sent_text.strip():
                continue
            
            doc = nlp(sent_text)
            corrections = self.adapter.enhance_doc_analysis(doc, self._get_rule_type()) if self.adapter else {}
            
            passive_constructions = self.passive_analyzer.find_passive_constructions(doc)
            
            for construction in passive_constructions:
                evidence_score = self.passive_analyzer.calculate_passive_voice_evidence(construction, doc, text, context)
                
                if evidence_score > self.config['evidence_thresholds']['passive_voice_min']:
                    suggestions = self._generate_context_aware_suggestions(construction, doc, sent_text)
                    message = self._get_contextual_passive_voice_message(construction, evidence_score)
                    
                    errors.append(self._create_error(
                        sentence=sent_text,
                        sentence_index=i,
                        message=message,
                        suggestions=suggestions,
                        severity='medium',
                        text=text,
                        context=context,
                        evidence_score=evidence_score,
                        span=(construction.span_start, construction.span_end),
                        flagged_text=construction.flagged_text
                    ))

            for token in doc:
                if token.lemma_.lower() == "will" and token.tag_ == "MD":
                    head_verb = token.head
                    if head_verb.pos_ == "VERB":
                        evidence_score = self._calculate_future_tense_evidence(token, head_verb, doc, sent_text, text, context or {})
                        if evidence_score > self.config['evidence_thresholds']['future_tense_min']:
                            flagged_text = f"{token.text} {head_verb.text}"
                            span_start = token.idx
                            span_end = head_verb.idx + len(head_verb.text)
                            errors.append(self._create_error(
                                sentence=sent_text,
                                sentence_index=i,
                                message=self._get_contextual_future_tense_message(flagged_text, evidence_score, context or {}),
                                suggestions=self._generate_smart_future_tense_suggestions(token, head_verb, evidence_score, context or {}),
                                severity='low' if evidence_score < 0.6 else 'medium',
                                text=text,
                                context=context,
                                evidence_score=evidence_score,
                                span=(span_start, span_end),
                                flagged_text=flagged_text
                            ))

            root_verb = self._find_root_token(doc)
            if (root_verb and root_verb.pos_ == 'VERB' and 'Tense=Past' in str(root_verb.morph)
                and not self._is_passive_construction(root_verb, doc)):
                
                if self._is_attributive_adjective(root_verb, doc):
                    continue

                evidence_score_past = self._calculate_past_tense_evidence(root_verb, doc, sent_text, text, context or {})

                if evidence_score_past > self.config['evidence_thresholds']['past_tense_min']:
                    flagged_text = root_verb.text
                    span_start = root_verb.idx
                    span_end = span_start + len(flagged_text)
                    errors.append(self._create_error(
                        sentence=sent_text,
                        sentence_index=i,
                        message=self._get_contextual_past_tense_message(flagged_text, evidence_score_past, context or {}),
                        suggestions=self._generate_smart_past_tense_suggestions(root_verb, evidence_score_past, context or {}, doc),
                        severity='low',
                        text=text,
                        context=context,
                        evidence_score=evidence_score_past,
                        span=(span_start, span_end),
                        flagged_text=flagged_text
                    ))

            for token in doc:
                if self._is_noun_form_as_verb_issue(token, doc, sent_text):
                    evidence_score = self._calculate_noun_verb_evidence(token, doc, sent_text, text, context or {})
                    if evidence_score > self.config['evidence_thresholds']['noun_verb_min']:
                        flagged_text = token.text
                        span_start = token.idx
                        span_end = span_start + len(flagged_text)
                        errors.append(self._create_error(
                            sentence=sent_text,
                            sentence_index=i,
                            message=self._get_contextual_noun_verb_message(flagged_text, evidence_score, context or {}),
                            suggestions=self._generate_smart_noun_verb_suggestions(token, evidence_score, context or {}),
                            severity='medium',
                            text=text,
                            context=context,
                            evidence_score=evidence_score,
                            span=(span_start, span_end),
                            flagged_text=flagged_text
                        ))
            
            for token in doc:
                if token.pos_ == 'VERB' and token.dep_ in ['ROOT', 'conj', 'ccomp', 'xcomp', 'advcl']:
                    subject = self._find_verb_subject_for_agreement(token, doc)
                    
                    if subject and self._has_subject_verb_disagreement(subject, token, doc, corrections):
                        evidence_score = self._calculate_agreement_evidence(subject, token, doc, sent_text, text, context or {})
                        
                        if evidence_score > self.config['evidence_thresholds']['agreement_min']:
                            flagged_text = f"{subject.text} ... {token.text}"
                            span_start = subject.idx
                            span_end = token.idx + len(token.text)
                            
                            errors.append(self._create_error(
                                sentence=sent_text,
                                sentence_index=i,
                                message=self._get_contextual_agreement_message(subject, token, evidence_score),
                                suggestions=self._generate_agreement_suggestions(subject, token, doc),
                                severity='high',
                                text=text,
                                context=context,
                                evidence_score=evidence_score,
                                span=(span_start, span_end),
                                flagged_text=flagged_text,
                                subtype='subject_verb_agreement'
                            ))
            
            root = self._find_root_token(doc)
            if root and root.tag_ == 'VB':
                for token in doc:
                    if token.tag_ == 'VBD':
                        evidence_score = 0.75
                        errors.append(self._create_error(
                            sentence=sent_text,
                            sentence_index=i,
                            message=f"Mixed verb tenses: The instruction starts with the imperative '{root.text}' but also includes the past tense verb '{token.text}'.",
                            suggestions=["Use a consistent tense, typically the present tense for all verbs in an instruction."],
                            severity='medium',
                            text=text,
                            context=context,
                            evidence_score=evidence_score,
                            span=(token.idx, token.idx + len(token.text)),
                            flagged_text=token.text,
                            subtype='tense_inconsistency'
                        ))
                        break  # Flag only the first occurrence in the sentence
            
            for token in doc:
                if self._is_weak_verb_with_action_noun(token):
                    action_noun = self._get_action_noun_object(token)
                    if action_noun:
                        evidence_score = self._calculate_nominalization_evidence(token, action_noun, doc, text, context or {})
                        if evidence_score > self.config['evidence_thresholds']['passive_voice_min']:
                            flagged_text = f"{token.text} ... {action_noun.text}"
                            span_start = token.idx
                            span_end = action_noun.idx + len(action_noun.text)
                            errors.append(self._create_error(
                                sentence=sent_text,
                                sentence_index=i,
                                message=self._get_contextual_nominalization_message(token, action_noun),
                                suggestions=self._generate_smart_nominalization_suggestions(token, action_noun),
                                severity='suggestion',
                                text=text,
                                context=context,
                                evidence_score=evidence_score,
                                span=(span_start, span_end),
                                flagged_text=flagged_text,
                                subtype='nominalized_verb'
                            ))
        
        return errors

    def _generate_context_aware_suggestions(self, construction, doc: Doc, sentence: str) -> List[str]:
        """Generate intelligent suggestions based on context classification."""
        suggestions = []
        
        try:
            # Use the analyzer's context classification
            context_type = construction.context_type
            base_verb = construction.main_verb.lemma_
            passive_subject = construction.passive_subject
            
            # Find agent in by-phrase
            agent = self._find_agent_in_by_phrase(doc, construction.main_verb)
            
            # Check for negative context
            is_negative = self._has_negative_context(doc, construction.main_verb)
            
            if is_negative:
                # Handle negative constructions
                antonym = self._get_positive_alternative(base_verb, doc)
                if antonym:
                    if context_type == ContextType.DESCRIPTIVE:
                        suggestions.append(f"Rewrite positively: 'The system {antonym}s {passive_subject.text.lower()}' (use descriptive active voice)")
                    else:
                        suggestions.append(f"Rewrite positively: 'You must {antonym} {passive_subject.text.lower()}' (use positive action)")
                else:
                    suggestions.append(f"Rewrite as requirement: 'You must ensure {passive_subject.text.lower()} is {self._get_past_participle(base_verb)}' (specify the requirement)")
            
            elif context_type == ContextType.DESCRIPTIVE:
                # Generate descriptive suggestions
                self._add_descriptive_suggestions(suggestions, base_verb, passive_subject, agent, doc)
            
            elif context_type == ContextType.INSTRUCTIONAL:
                # Generate instructional suggestions
                self._add_instructional_suggestions(suggestions, base_verb, passive_subject, agent)
            
            else:
                # Uncertain context - provide both options
                suggestions.append(f"For descriptions: 'The system {self._conjugate_verb(base_verb, 'system')} {passive_subject.text.lower()}'")
                suggestions.append(f"For instructions: '{base_verb.capitalize()} the {passive_subject.text.lower()}'")
            
            # Add agent-based suggestion if available
            if agent and len(suggestions) < 2:
                verb_form = self._conjugate_verb(base_verb, agent)
                suggestions.append(f"Make the agent active: '{agent.capitalize()} {verb_form} {passive_subject.text.lower()}'")
            
            # Fallback
            if not suggestions:
                suggestions.append("Rewrite in active voice by identifying who or what performs the action")
                
        except Exception as e:
            suggestions = ["Rewrite in active voice by identifying who or what performs the action"]
        
        return suggestions[:3]

    def _add_descriptive_suggestions(self, suggestions: List[str], base_verb: str, 
                                   passive_subject: Token, agent: str, doc: Doc) -> None:
        """Add suggestions appropriate for descriptive context."""
        descriptive_actors = self._get_descriptive_actors(base_verb, passive_subject, doc)
        
        for actor in descriptive_actors:
            final_verb = self._get_stylistically_appropriate_verb(base_verb, actor)
            verb_form = self._conjugate_verb(final_verb, actor)
            
            if passive_subject and passive_subject.text.lower() in ['it', 'this', 'that']:
                suggestions.append(f"Use descriptive active voice: '{actor.capitalize()} {verb_form} {passive_subject.text.lower()}'")
            else:
                suggestions.append(f"Use descriptive active voice: '{actor.capitalize()} {verb_form} the {passive_subject.text.lower()}'")
            
            if len(suggestions) >= 2:
                break
        
        if not suggestions:
            if base_verb in ['document', 'describe', 'specify', 'define']:
                fallback_verb = self._get_stylistically_appropriate_verb(base_verb, 'the documentation')
                suggestions.append(f"Use descriptive active voice: 'The documentation {self._conjugate_verb(fallback_verb, 'documentation')} {passive_subject.text.lower()}'")
            else:
                suggestions.append(f"Use descriptive active voice: 'The system {self._conjugate_verb(base_verb, 'system')} {passive_subject.text.lower()}'")

    def _get_stylistically_appropriate_verb(self, base_verb: str, actor: str) -> str:
        """Return a stylistically appropriate verb, avoiding same-root awkwardness."""
        actor_root = actor.replace('the ', '').replace('a ', '').replace('an ', '')
        
        if self._is_same_root_awkward(actor_root, base_verb):
            alternative = self.config['verb_stylistic_alternatives'].get(base_verb)
            if alternative:
                return alternative
        
        return base_verb
    
    def _is_same_root_awkward(self, actor_root: str, verb: str) -> bool:
        """Check if using actor + verb creates awkward same-root construction."""
        return ([actor_root, verb] in self.config['same_root_awkward_pairs'] or 
                actor_root.startswith(verb) or verb.startswith(actor_root))

    def _add_instructional_suggestions(self, suggestions: List[str], base_verb: str, 
                                     passive_subject: Token, agent: str) -> None:
        """Add suggestions appropriate for instructional context."""
        if passive_subject and passive_subject.text.lower() in ['it', 'this', 'that']:
            suggestions.append(f"Use imperative: '{base_verb.capitalize()} {passive_subject.text.lower()}' (make the user the actor)")
        else:
            suggestions.append(f"Use imperative: '{base_verb.capitalize()} the {passive_subject.text.lower()}' (make the user the actor)")
        
        if base_verb in ['configure', 'install', 'setup', 'deploy', 'enable', 'disable']:
            suggestions.append(f"Specify the actor: 'The administrator {self._conjugate_verb(base_verb, 'administrator')} the {passive_subject.text.lower()}'")
        elif base_verb in ['test', 'verify', 'check', 'validate']:
            suggestions.append(f"Specify the actor: 'You {base_verb} the {passive_subject.text.lower()}'")

    def _get_descriptive_actors(self, base_verb: str, passive_subject: Token, doc: Doc) -> List[str]:
        """Get appropriate actors for descriptive active voice."""
        actors = self.config['descriptive_actors'].get(base_verb, self.config['descriptive_actors']['default'])
        
        context_words = [token.lemma_.lower() for token in doc]
        if any(word in context_words for word in self.config['database_context_words']):
            if base_verb in self.config['database_verbs']:
                actors = self.config['descriptive_actors']['store'] + actors
        
        return actors[:self.config['max_actors']]

    def _find_root_token(self, doc: Doc) -> Token:
        """Find the root token of the sentence."""
        for token in doc:
            if token.dep_ == "ROOT":
                return token
        return None

    def _is_passive_construction(self, verb_token: Token, doc: Doc) -> bool:
        """Check if a verb token is part of a passive voice construction."""
        if not verb_token:
            return False
        
        # Use analyzer for consistency
        constructions = self.passive_analyzer.find_passive_constructions(doc)
        return any(c.main_verb == verb_token for c in constructions)
    
    def _is_attributive_adjective(self, token: Token, doc: Doc) -> bool:
        """
        Check if a past participle is being used as an attributive adjective.
        
        Examples:
        - "forked sample" - forked modifies sample (adjective use)
        - "cloned repository" - cloned modifies repository (adjective use)
        - "updated files" - updated modifies files (adjective use)
        
        Returns True if this is an adjective, not a verb.
        """
        if not token:
            return False
        
        # Check if token is modifying a noun (amod dependency)
        if token.dep_ == 'amod':
            return True
        
        # Check if any children are nouns that this token modifies
        for child in token.children:
            if child.pos_ == 'NOUN' and child.dep_ in ['dobj', 'nsubj', 'compound']:
                # This "verb" has noun modifiers, suggesting it's actually an adjective
                return True
        
        # Check if token's head is a noun (common pattern: "forked repositories")
        if hasattr(token, 'head') and token.head.pos_ == 'NOUN':
            return True
        
        return False

    def _find_agent_in_by_phrase(self, doc: Doc, main_verb: Token) -> str:
        """Find the agent in a 'by' phrase."""
        for token in doc:
            if token.lemma_ == 'by' and token.head == main_verb:
                for child in token.children:
                    if child.dep_ == 'pobj':
                        return child.text
        return None

    def _has_negative_context(self, doc: Doc, main_verb: Token) -> bool:
        """Check if the passive construction is in a negative context."""
        for token in doc:
            if token.dep_ == 'neg' or token.lemma_ in ['not', 'never', 'cannot']:
                if token.head == main_verb or token.head.head == main_verb:
                    return True
        
        sentence_text = doc.text.lower()
        return any(pattern in sentence_text for pattern in self.config['negative_patterns'])

    def _get_positive_alternative(self, verb_lemma: str, doc: Doc) -> str:
        """Get a positive alternative for a negated verb."""
        return self.config['negative_verb_alternatives'].get(verb_lemma)

    def _get_past_participle(self, verb_lemma: str) -> str:
        """Get the past participle form of a verb."""
        if verb_lemma in self.config['irregular_past_participles']:
            return self.config['irregular_past_participles'][verb_lemma]
        
        if verb_lemma.endswith('e'):
            return verb_lemma + 'd'
        else:
            return verb_lemma + 'ed'

    def _conjugate_verb(self, verb_lemma: str, subject: str) -> str:
        """Enhanced verb conjugation for both singular and descriptive subjects."""
        if subject.startswith('the '):
            subject_noun = subject[4:]
        else:
            subject_noun = subject.lower()
        
        if subject_noun in self.config['third_person_singular_subjects']:
            return self._add_s_ending(verb_lemma)
        else:
            return verb_lemma

    def _has_legitimate_temporal_context(self, doc, sentence: str) -> bool:
        """Detect when past tense is appropriate due to temporal context."""
        sentence_lower = sentence.lower()
        
        if any(indicator in sentence_lower for indicator in self.config['temporal_indicators']['strong']):
            return True
        
        if any(indicator in sentence_lower for indicator in self.config['temporal_indicators']['weak']):
            return True
        
        if self._detect_temporal_patterns(doc):
            return True
        
        if self._detect_release_notes_context(sentence_lower):
            return True
        
        return False
    
    def _detect_temporal_patterns(self, doc) -> bool:
        """Use spaCy to detect temporal patterns."""
        if len(doc) > 0:
            first_token = doc[0]
            if first_token.lemma_.lower() in self.config['temporal_prepositions']:
                return True
        
        for token in doc:
            if (token.lemma_.lower() in self.config['temporal_nouns'] and 
                any(child.text.lower() in {'this', 'the', 'previous', 'earlier'} 
                    for child in token.children)):
                return True
        
        for token in doc:
            if (token.lemma_.lower() in self.config['modal_past_verbs'] and token.tag_ == 'MD'):
                head_verb = token.head
                if head_verb and head_verb.lemma_.lower() in self.config['temporal_context_verbs']:
                    if any(child.lemma_.lower() == 'not' for child in token.children):
                        return True
        
        return False
    
    def _detect_release_notes_context(self, sentence_lower: str) -> bool:
        """Detect if this appears to be release notes or changelog context."""
        if any(indicator in sentence_lower for indicator in self.config['release_notes_indicators']):
            return True
        
        if any(pattern in sentence_lower for pattern in self.config['issue_indicators']):
            return True
        
        return False

    def _get_contextual_passive_voice_message(self, construction: PassiveConstruction, evidence_score: float) -> str:
        """Generate context-aware error messages for passive voice."""
        
        context_type = construction.context_type
        
        # Base message varies by evidence strength
        if evidence_score > 0.8:
            base_msg = "Sentence is in the passive voice."
        elif evidence_score > 0.5:
            base_msg = "Sentence may be in the passive voice."
        else:
            base_msg = "Sentence appears to use passive voice."
        
        # Add context-specific guidance
        if context_type == ContextType.INSTRUCTIONAL:
            return f"{base_msg} Consider using active voice for clearer instructions."
        elif context_type == ContextType.DESCRIPTIVE:
            return f"{base_msg} While acceptable for descriptions, active voice may be clearer."
        else:
            return f"{base_msg} Consider using active voice for clarity."

    # === EVIDENCE-BASED: FUTURE TENSE ===

    def _calculate_future_tense_evidence(self, will_token: Token, head_verb: Token, doc: Doc, sentence: str, text: str, context: Dict[str, Any]) -> float:
        """Calculate evidence score for future tense concerns."""
        evidence_score = self._get_base_future_tense_evidence(will_token, head_verb, doc, sentence)
        
        if evidence_score == 0.0:
            return 0.0
        
        evidence_score = self._apply_linguistic_clues_future_tense(evidence_score, will_token, head_verb, doc, sentence)
        evidence_score = self._apply_structural_clues_future_tense(evidence_score, context)
        evidence_score = self._apply_semantic_clues_future_tense(evidence_score, will_token, head_verb, text, context)
        evidence_score = self._apply_feedback_clues_future_tense(evidence_score, will_token, head_verb, context)
        
        return max(0.0, min(1.0, evidence_score))

    def _get_base_future_tense_evidence(self, will_token: Token, head_verb: Token, doc: Doc, sentence: str) -> float:
        """Get base evidence score for future tense concerns."""
        construction = f"{will_token.text.lower()} {head_verb.lemma_.lower()}"
        constructions = self.config['future_tense_constructions']
        base_evidence = self.config['future_base_evidence']
        
        if construction in constructions['inappropriate_in_instructions']:
            return base_evidence['inappropriate']
        elif construction in constructions['better_as_present']:
            return base_evidence['better_present']
        elif construction in constructions['context_dependent']:
            return base_evidence['context_dependent']
        else:
            return base_evidence['default']

    def _apply_linguistic_clues_future_tense(self, evidence_score: float, will_token: Token, head_verb: Token, doc: Doc, sentence: str) -> float:
        """Apply linguistic clues for future tense detection."""
        w = self.config['future_linguistic_weights']
        sent_lower = sentence.lower()
        
        if sentence.strip().endswith('?'):
            evidence_score -= w['question_reduction']
        
        if any(marker in sent_lower for marker in [f"{m} " for m in self.config['conditional_markers'][:5]]):
            evidence_score -= w['conditional_reduction']
        
        if any(indicator in sent_lower for indicator in self.config['scheduled_indicators']):
            evidence_score -= w['scheduled_reduction']
        
        if 'will be able to' in sent_lower:
            evidence_score -= w['will_be_able_reduction']
        
        if any(combo in sent_lower for combo in self.config['modal_combinations']):
            evidence_score -= w['modal_combo_reduction']
        
        if any(indicator in sent_lower for indicator in self.config['instruction_indicators']):
            evidence_score += w['instruction_boost']
        
        if 'will not' in sent_lower or "won't" in sent_lower:
            evidence_score -= w['negative_reduction']
        
        for token in doc:
            if hasattr(token, 'ent_type_') and token.ent_type_:
                ent_type = token.ent_type_
                if ent_type in self.config['entity_types']['organizational'] and abs(token.i - will_token.i) <= 3:
                    evidence_score -= w['entity_org_reduction']
                elif ent_type in self.config['entity_types']['personal'] and abs(token.i - will_token.i) <= 2:
                    evidence_score -= w['entity_person_reduction']
                elif ent_type in self.config['entity_types']['temporal'] and abs(token.i - will_token.i) <= 4:
                    evidence_score -= w['entity_temporal_reduction']
        
        if head_verb.lemma_.lower() in self.config['action_verbs']:
            evidence_score += w['action_verb_boost']
        
        if head_verb.lemma_.lower() in self.config['state_verbs']:
            evidence_score -= w['state_verb_reduction']
        
        first_two_tokens = [token.text.lower() for token in doc[:2]]
        if 'will' in first_two_tokens:
            evidence_score += w['early_position_boost']
        
        return evidence_score

    def _apply_structural_clues_future_tense(self, evidence_score: float, context: Dict[str, Any]) -> float:
        """Apply document structure clues for future tense."""
        w = self.config['future_structural_weights']
        block_type = context.get('block_type', 'paragraph')
        
        if block_type in ['code_block', 'literal_block']:
            evidence_score -= w['code_block_reduction']
        elif block_type == 'inline_code':
            evidence_score -= w['inline_code_reduction']
        elif block_type == 'heading':
            heading_level = context.get('block_level', 1)
            if heading_level == 1:
                evidence_score -= w['h1_reduction']
            elif heading_level == 2:
                evidence_score -= w['h2_reduction']
            elif heading_level >= 3:
                evidence_score -= w['h3_plus_reduction']
        elif block_type in ['ordered_list_item', 'unordered_list_item']:
            evidence_score += w['list_boost']
            if context.get('list_depth', 1) > 1:
                evidence_score += w['nested_list_boost']
        elif block_type in ['table_cell', 'table_header']:
            evidence_score += w['table_boost']
        elif block_type == 'admonition':
            admonition_type = context.get('admonition_type', '').upper()
            if admonition_type in self.config['admonition_types']['informational']:
                evidence_score -= w['note_tip_reduction']
            elif admonition_type in self.config['admonition_types']['warning']:
                evidence_score -= w['warning_reduction']
            elif admonition_type in self.config['admonition_types']['important']:
                evidence_score -= w['important_reduction']
        elif block_type in ['block_quote', 'citation']:
            evidence_score -= w['quote_reduction']
        elif block_type in ['example', 'sample']:
            evidence_score -= w['example_reduction']
        
        return evidence_score

    def _apply_semantic_clues_future_tense(self, evidence_score: float, will_token: Token, head_verb: Token, text: str, context: Dict[str, Any]) -> float:
        """Apply semantic and content-type clues for future tense."""
        w = self.config['future_semantic_weights']
        content_type = context.get('content_type', 'general')
        domain = context.get('domain', 'general')
        audience = context.get('audience', 'general')
        
        if content_type == 'procedural':
            evidence_score += w['procedural_boost']
        elif content_type == 'api':
            evidence_score += w['api_boost']
        elif content_type == 'technical':
            evidence_score += w['technical_boost']
        elif content_type == 'tutorial':
            evidence_score += w['tutorial_boost']
        elif content_type == 'legal':
            evidence_score += w['legal_boost']
        elif content_type == 'academic':
            evidence_score += w['academic_boost']
        elif content_type == 'marketing':
            evidence_score -= w['marketing_reduction']
        elif content_type == 'narrative':
            evidence_score -= w['narrative_reduction']
        
        if domain in ['software', 'engineering', 'devops']:
            evidence_score += w['software_domain_boost']
        elif domain in ['user-documentation', 'help']:
            evidence_score += w['user_docs_boost']
        elif domain in ['training', 'education']:
            evidence_score += w['training_boost']
        elif domain in ['legal', 'compliance']:
            evidence_score += w['legal_domain_boost']
        elif domain in ['planning', 'roadmap']:
            evidence_score -= w['planning_domain_reduction']
        
        if audience in ['beginner', 'general', 'consumer']:
            evidence_score += w['beginner_audience_boost']
        elif audience in ['professional', 'business']:
            evidence_score += w['professional_boost']
        elif audience in ['developer', 'technical', 'expert']:
            evidence_score += w['technical_audience_boost']
        
        if self._is_installation_documentation(text):
            evidence_score += w['installation_docs_boost']
        if self._is_troubleshooting_documentation(text):
            evidence_score += w['troubleshooting_docs_boost']
        if self._is_ui_documentation(text):
            evidence_score += w['ui_docs_boost']
        if self._is_release_notes_documentation(text):
            evidence_score -= w['release_notes_reduction']
        if self._is_roadmap_documentation(text):
            evidence_score -= w['roadmap_reduction']
        if self._is_planning_documentation(text):
            evidence_score -= w['planning_docs_reduction']
        
        if self._is_consequence_or_conditional_context(will_token, text):
            evidence_score -= w['consequence_reduction']
        
        doc_length = len(text.split())
        if doc_length < 100:
            evidence_score += w['doc_short_boost']
        elif doc_length > 5000:
            evidence_score += w['doc_long_boost']
        
        return evidence_score

    def _is_consequence_or_conditional_context(self, will_token: Token, text: str) -> bool:
        """Detect if 'will' appears in consequence or conditional context."""
        if not will_token or not hasattr(will_token, 'doc') or not hasattr(will_token, 'i'):
            return False
        
        doc = will_token.doc
        will_index = will_token.i
        sentence = will_token.sent
        
        sentence_start = sentence.start
        tokens_before_will = doc[sentence_start:will_index]
        text_before_will = ' '.join([token.text.lower() for token in tokens_before_will])
        
        for indicator in self.config['conditional_markers']:
            if indicator in text_before_will:
                return True
        for indicator in self.config['failure_indicators']:
            if indicator in text_before_will:
                return True
        for indicator in self.config['causal_indicators']:
            if indicator in text_before_will:
                return True
        
        sentences = list(doc.sents)
        current_sent_index = next((i for i, sent in enumerate(sentences) if sent == sentence), None)
        
        if current_sent_index is not None and current_sent_index > 0:
            prev_sent_text = sentences[current_sent_index - 1].text.lower()
            for indicator in self.config['conditional_markers']:
                if indicator in prev_sent_text:
                    return True
            for indicator in self.config['failure_indicators']:
                if indicator in prev_sent_text:
                    return True
        
        sentence_text = sentence.text.lower()
        if 'failure to' in sentence_text and 'will' in sentence_text:
            return True
        if 'if' in sentence_text and ('don\'t' in sentence_text or 'do not' in sentence_text) and 'will' in sentence_text:
            return True
        if 'should' in sentence_text and ('fail' in sentence_text or 'error' in sentence_text) and 'will' in sentence_text:
            return True
        if 'when' in sentence_text and ('happens' in sentence_text or 'occurs' in sentence_text or 'restarts' in sentence_text) and 'will' in sentence_text:
            return True
        if 'in case' in sentence_text and 'will' in sentence_text:
            return True
        
        return False

    def _apply_feedback_clues_future_tense(self, evidence_score: float, will_token: Token, head_verb: Token, context: Dict[str, Any]) -> float:
        """Apply feedback patterns for future tense."""
        w = self.config['feedback_weights']
        feedback_patterns = self._get_cached_feedback_patterns('verbs_future_tense')
        phrase = f"{will_token.text.lower()} {head_verb.lemma_.lower()}"
        
        if phrase in feedback_patterns.get('accepted_future_phrases', set()):
            evidence_score -= w['accepted_phrase_reduction']
        
        if phrase in feedback_patterns.get('flagged_future_phrases', set()):
            evidence_score += w['flagged_phrase_boost']
        
        content_type = context.get('content_type', 'general')
        context_patterns = feedback_patterns.get(f'{content_type}_future_patterns', {})
        if phrase in context_patterns.get('acceptable', set()):
            evidence_score -= w['context_acceptable_reduction']
        elif phrase in context_patterns.get('problematic', set()):
            evidence_score += w['context_problematic_boost']
        
        verb_lemma = head_verb.lemma_.lower()
        verb_patterns = feedback_patterns.get('verb_specific_patterns', {})
        if verb_lemma in verb_patterns.get('often_accepted_with_will', set()):
            evidence_score -= w['verb_accepted_reduction']
        elif verb_lemma in verb_patterns.get('problematic_with_will', set()):
            evidence_score += w['verb_problematic_boost']
        
        correction_success = feedback_patterns.get('correction_success', {}).get(phrase, 0.5)
        if correction_success > 0.8:
            evidence_score += w['high_correction_success_boost']
        elif correction_success < 0.3:
            evidence_score -= w['low_correction_success_reduction']
        
        return evidence_score

    def _get_contextual_future_tense_message(self, flagged_text: str, ev: float, context: Dict[str, Any]) -> str:
        """Generate context-aware error messages for future tense."""
        content_type = context.get('content_type', 'general')
        thresholds = self.config['evidence_thresholds']
        
        if ev > thresholds['future_high']:
            return f"Future tense '{flagged_text}' should be avoided in {content_type} documentation. Use present or imperative."
        elif ev > thresholds['future_medium']:
            return f"Consider replacing '{flagged_text}' with present tense for clearer {content_type} writing."
        elif ev > thresholds['future_low']:
            return f"Present tense is preferred over '{flagged_text}' in most technical writing."
        else:
            return f"The phrase '{flagged_text}' may benefit from present tense for clarity."

    def _generate_smart_future_tense_suggestions(self, will_token: Token, head_verb: Token, ev: float, context: Dict[str, Any]) -> List[str]:
        """Generate context-aware suggestions for future tense patterns with proper subject-verb agreement."""
        
        base = head_verb.lemma_
        suggestions = []
        content_type = context.get('content_type', 'general')
        
        # Find the subject of the verb to ensure proper conjugation
        subject_token = self._find_verb_subject(will_token, head_verb)
        
        # Base suggestions based on evidence strength and context with linguistically-aware conjugation
        if ev > 0.8:
            if content_type == 'procedural':
                suggestions.append(f"Use imperative: '{base.capitalize()}' (direct instruction)")
                if subject_token:
                    present_form = self._generate_smart_verb_suggestions(head_verb, subject_token, 'present')
                    suggestions.append(f"Use present: '{present_form}' (describe current behavior)")
                else:
                    suggestions.append(f"Use present: 'The system {self._conjugate_verb_for_subject(base, 'system')}' (describe current behavior)")
            elif content_type == 'api':
                method_form = self._conjugate_verb_for_subject(base, 'method')
                suggestions.append(f"Use present: 'The method {method_form}' (current API behavior)")
                this_form = self._conjugate_verb_for_subject(base, 'this')
                suggestions.append(f"Use present: 'This {this_form} the resource' (API functionality)")
            else:
                if subject_token:
                    present_form = self._generate_smart_verb_suggestions(head_verb, subject_token, 'present')
                    suggestions.append(f"Use present: '{present_form}' (current state/behavior)")
                    suggestions.append(f"Use imperative: '{base.capitalize()}' (direct action)")
                else:
                    suggestions.append(f"Use present with proper subject-verb agreement")
                    suggestions.append(f"Use imperative: '{base.capitalize()}' (direct action)")
        else:
            if subject_token:
                present_form = self._generate_smart_verb_suggestions(head_verb, subject_token, 'present')
                suggestions.append(f"Consider present: '{present_form}'")
            else:
                suggestions.append(f"Consider present with proper subject-verb agreement")
            suggestions.append(f"Consider imperative: '{base.capitalize()}'")
        
        # Context-specific advice
        if content_type == 'procedural':
            suggestions.append("Use imperative mood for clear step-by-step instructions.")
        elif content_type == 'api':
            suggestions.append("Describe API behavior in present tense for clarity.")
        elif content_type == 'technical':
            suggestions.append("Technical documentation should describe current system behavior.")
        elif content_type == 'tutorial':
            suggestions.append("Tutorial steps should be in imperative mood.")
        
        return suggestions[:3]

    def _calculate_past_tense_evidence(self, root_verb: Token, doc: Doc, sentence: str, text: str, context: Dict[str, Any]) -> float:
        """Calculate evidence score for past tense concerns."""
        evidence_score = self._get_base_past_tense_evidence(root_verb, doc, sentence)
        
        if evidence_score == 0.0:
            return 0.0
        
        if context.get('is_link_text'):
            evidence_score += 0.3
        
        if self._is_narrative_or_blog_content(text, context):
            evidence_score -= self.config['past_semantic_weights']['narrative_blog_reduction']
        
        if self._is_technical_compound_noun(root_verb, doc):
            return 0.0
        
        evidence_score = self._apply_linguistic_clues_past_tense(evidence_score, root_verb, doc, sentence)
        evidence_score = self._apply_structural_clues_past_tense(evidence_score, context)
        evidence_score = self._apply_semantic_clues_past_tense(evidence_score, root_verb, text, context)
        evidence_score = self._apply_feedback_clues_past_tense(evidence_score, root_verb, context)
        
        return max(0.0, min(1.0, evidence_score))

    def _is_narrative_or_blog_content(self, text: str, context: Dict[str, Any]) -> bool:
        """Detect if content is narrative/blog style."""
        if not text:
            return False
            
        try:
            from validation.confidence.context_analyzer import ContextAnalyzer
            analyzer = ContextAnalyzer()
            content_result = analyzer.detect_content_type(text, context)
            
            if content_result.content_type.value == 'narrative' and content_result.confidence > 0.4:
                return True
            
            text_lower = text.lower()
            strong_indicator_count = sum(1 for indicator in self.config['blog_strong_indicators'] if indicator in text_lower)
            
            if strong_indicator_count >= self.config['min_counts']['blog_strong']:
                return True
                
            narrative_verb_count = sum(1 for verb in self.config['narrative_verbs'] if verb in text_lower)
            
            words = text_lower.split()
            if len(words) > 20:
                first_person_count = sum(1 for word in words if word in self.config['first_person_pronouns'])
                first_person_ratio = first_person_count / len(words)
                
                if first_person_ratio > self.config['min_counts']['first_person_threshold'] and narrative_verb_count > 0:
                    return True
                    
        except ImportError:
            text_lower = text.lower()
            if any(indicator in text_lower for indicator in self.config['blog_strong_indicators'][:5]):
                return True
        
        return False

    def _get_base_past_tense_evidence(self, root_verb: Token, doc: Doc, sentence: str) -> float:
        """Get base evidence score for past tense concerns."""
        verb_text = root_verb.text.lower()
        verb_lemma = root_verb.lemma_.lower()
        verbs = self.config['past_tense_verbs']
        base_evidence = self.config['past_base_evidence']
        
        if verb_text in verbs['inappropriate_in_current_docs'] or verb_lemma in verbs['inappropriate_in_current_docs']:
            return base_evidence['inappropriate']
        elif verb_text in verbs['better_as_present'] or verb_lemma in verbs['better_as_present']:
            return base_evidence['better_present']
        elif verb_text in verbs['temporal_acceptable'] or verb_lemma in verbs['temporal_acceptable']:
            return base_evidence['temporal_acceptable']
        else:
            return base_evidence['default']

    def _apply_linguistic_clues_past_tense(self, evidence_score: float, root_verb: Token, doc: Doc, sentence: str) -> float:
        """Apply linguistic clues for past tense detection."""
        w = self.config['past_linguistic_weights']
        sent_lower = sentence.lower()
        
        for token in doc:
            if token.lemma_.lower() in self.config['perfect_auxiliaries']:
                if (token.head == root_verb or root_verb.head == token or abs(token.i - root_verb.i) <= 2):
                    return 0.0
        
        if any(indicator in sent_lower for indicator in self.config['temporal_indicators']['strong']):
            evidence_score -= w['strong_temporal_reduction']
        
        if any(indicator in sent_lower for indicator in self.config['temporal_indicators']['weak']):
            evidence_score -= w['weak_temporal_reduction']
        
        if self._has_legitimate_temporal_context(doc, sentence):
            evidence_score -= w['legitimate_temporal_reduction']
        
        if any(indicator in sent_lower for indicator in self.config['issue_indicators']):
            evidence_score -= w['issue_indicator_reduction']
        
        if any(pattern in sent_lower for pattern in self.config['negative_past_patterns']):
            evidence_score -= w['negative_past_reduction']
        
        for token in doc:
            if token.dep_ == 'mark' and token.lemma_.lower() in self.config['subordinate_markers']:
                if token.head and 'Tense=Past' in str(token.head.morph):
                    evidence_score -= w['subordinate_clause_reduction']
        
        if any(marker in sent_lower for marker in self.config['conditional_markers'][:5]):
            evidence_score -= w['conditional_reduction']
        
        if '"' in sentence or "'" in sentence:
            evidence_score -= w['quote_reduction']
        
        for token in doc:
            if hasattr(token, 'ent_type_') and token.ent_type_:
                ent_type = token.ent_type_
                if ent_type in self.config['entity_types']['organizational'] and abs(token.i - root_verb.i) <= 3:
                    evidence_score -= w['entity_org_reduction']
                elif ent_type in self.config['entity_types']['personal'] and abs(token.i - root_verb.i) <= 2:
                    evidence_score -= w['entity_person_reduction']
                elif ent_type in self.config['entity_types']['temporal'] and abs(token.i - root_verb.i) <= 4:
                    evidence_score -= w['entity_temporal_reduction']
                elif ent_type in self.config['entity_types']['numeric']:
                    if any(word in doc.text.lower() for word in ['version', 'release', 'update']):
                        if abs(token.i - root_verb.i) <= 3:
                            evidence_score -= w['entity_cardinal_version_reduction']
        
        if any(indicator in sent_lower for indicator in self.config['comparison_indicators']):
            evidence_score -= w['comparison_reduction']
        
        return evidence_score

    def _apply_structural_clues_past_tense(self, evidence_score: float, context: Dict[str, Any]) -> float:
        """Apply document structure clues for past tense."""
        w = self.config['past_structural_weights']
        block_type = context.get('block_type', 'paragraph')
        
        preceding_heading = context.get('preceding_heading', '').lower()
        current_heading = context.get('current_heading', '').lower()
        parent_title = context.get('parent_title', '').lower()
        
        is_prerequisites_context = (
            any(keyword in preceding_heading for keyword in self.config['prerequisite_keywords']) or
            any(keyword in current_heading for keyword in self.config['prerequisite_keywords']) or
            any(keyword in parent_title for keyword in self.config['prerequisite_keywords'])
        )
        
        if is_prerequisites_context:
            if block_type in ['list_item', 'ordered_list_item', 'unordered_list_item', 'paragraph']:
                evidence_score -= w['prerequisites_reduction']
        
        if block_type in ['code_block', 'literal_block']:
            evidence_score -= w['code_block_reduction']
        elif block_type == 'inline_code':
            evidence_score -= w['inline_code_reduction']
        elif block_type == 'heading':
            heading_level = context.get('block_level', 1)
            if heading_level == 1:
                evidence_score += w['h1_boost']
            elif heading_level == 2:
                evidence_score += w['h2_boost']
            elif heading_level >= 3:
                evidence_score += w['h3_plus_boost']
        elif block_type in ['ordered_list_item', 'unordered_list_item']:
            evidence_score += w['list_boost']
            if context.get('list_depth', 1) > 1:
                evidence_score += w['nested_list_boost']
        elif block_type in ['table_cell', 'table_header']:
            evidence_score += w['table_boost']
        elif block_type == 'admonition':
            admonition_type = context.get('admonition_type', '').upper()
            if admonition_type in self.config['admonition_types']['informational']:
                evidence_score -= w['note_tip_reduction']
            elif admonition_type in self.config['admonition_types']['warning']:
                evidence_score -= w['warning_reduction']
            elif admonition_type in self.config['admonition_types']['important']:
                evidence_score += w['important_boost']
        elif block_type in ['block_quote', 'citation']:
            evidence_score -= w['quote_reduction']
        elif block_type in ['example', 'sample']:
            evidence_score -= w['example_reduction']
        elif block_type in ['changelog', 'release_notes']:
            evidence_score -= w['changelog_reduction']
        
        return evidence_score

    def _apply_semantic_clues_past_tense(self, evidence_score: float, root_verb: Token, text: str, context: Dict[str, Any]) -> float:
        """Apply semantic and content-type clues for past tense."""
        w = self.config['past_semantic_weights']
        content_type = context.get('content_type', 'general')
        domain = context.get('domain', 'general')
        audience = context.get('audience', 'general')
        
        if content_type == 'procedural':
            evidence_score += w['procedural_boost']
        elif content_type == 'api':
            evidence_score += w['api_boost']
        elif content_type == 'technical':
            evidence_score += w['technical_boost']
        elif content_type == 'tutorial':
            evidence_score += w['tutorial_boost']
        elif content_type == 'legal':
            evidence_score += w['legal_boost']
        elif content_type == 'academic':
            evidence_score += w['academic_boost']
        elif content_type == 'marketing':
            evidence_score += w['marketing_boost']
        elif content_type == 'narrative':
            evidence_score -= w['narrative_reduction']
        
        if content_type in self.config['past_appropriate_content_types']:
            evidence_score -= w['past_appropriate_reduction']
        
        if domain in ['software', 'engineering', 'devops']:
            evidence_score += w['software_domain_boost']
        elif domain in ['user-documentation', 'help']:
            evidence_score += w['user_docs_boost']
        elif domain in ['training', 'education']:
            evidence_score += w['training_boost']
        elif domain in ['legal', 'compliance']:
            evidence_score += w['legal_domain_boost']
        elif domain in ['support', 'troubleshooting']:
            evidence_score -= w['support_domain_reduction']
        
        if audience in ['beginner', 'general', 'consumer']:
            evidence_score += w['beginner_audience_boost']
        elif audience in ['professional', 'business']:
            evidence_score += w['professional_boost']
        elif audience in ['developer', 'technical', 'expert']:
            evidence_score += w['technical_audience_boost']
        
        if self._is_installation_documentation(text):
            evidence_score += w['installation_docs_boost']
        if self._is_troubleshooting_documentation(text):
            evidence_score -= w['troubleshooting_docs_reduction']
        if self._is_ui_documentation(text):
            evidence_score += w['ui_docs_boost']
        if self._is_release_notes_documentation(text):
            evidence_score -= w['release_notes_reduction']
        if self._is_changelog_documentation(text):
            evidence_score -= w['changelog_reduction']
        if self._is_bug_report_documentation(text):
            evidence_score -= w['bug_report_reduction']
        
        doc_length = len(text.split())
        if doc_length < 100:
            evidence_score += w['doc_short_boost']
        elif doc_length > 5000:
            evidence_score += w['doc_long_boost']
        
        return evidence_score

    def _apply_feedback_clues_past_tense(self, evidence_score: float, root_verb: Token, context: Dict[str, Any]) -> float:
        """Apply feedback patterns for past tense."""
        w = self.config['feedback_weights']
        feedback_patterns = self._get_cached_feedback_patterns('verbs_past_tense')
        verb_lemma = root_verb.lemma_.lower()
        verb_text = root_verb.text.lower()
        
        if verb_lemma in feedback_patterns.get('often_accepted_past_verbs', set()) or verb_text in feedback_patterns.get('often_accepted_past_verbs', set()):
            evidence_score -= w['accepted_phrase_reduction']
        
        if verb_lemma in feedback_patterns.get('often_flagged_past_verbs', set()) or verb_text in feedback_patterns.get('often_flagged_past_verbs', set()):
            evidence_score += w['flagged_phrase_boost']
        
        content_type = context.get('content_type', 'general')
        context_patterns = feedback_patterns.get(f'{content_type}_past_patterns', {})
        if verb_lemma in context_patterns.get('acceptable', set()):
            evidence_score -= w['context_acceptable_reduction']
        elif verb_lemma in context_patterns.get('problematic', set()):
            evidence_score += w['context_problematic_boost']
        
        if verb_lemma in feedback_patterns.get('temporal_context_patterns', {}).get('acceptable_in_temporal', set()):
            evidence_score -= w['temporal_acceptable_reduction']
        
        correction_success = feedback_patterns.get('correction_success', {}).get(verb_lemma, 0.5)
        if correction_success > 0.8:
            evidence_score += w['high_correction_success_boost']
        elif correction_success < 0.3:
            evidence_score -= w['low_correction_success_reduction']
        
        return evidence_score

    def _get_contextual_past_tense_message(self, flagged_text: str, ev: float, context: Dict[str, Any]) -> str:
        """Generate context-aware error messages for past tense."""
        content_type = context.get('content_type', 'general')
        thresholds = self.config['evidence_thresholds']
        
        if ev > thresholds['past_high']:
            return f"Past tense '{flagged_text}' should be avoided in {content_type} documentation. Use present tense."
        elif ev > thresholds['past_medium']:
            return f"Consider replacing '{flagged_text}' with present tense for clearer {content_type} writing."
        elif ev > thresholds['past_low']:
            return f"Present tense is preferred over '{flagged_text}' in most technical writing."
        else:
            return f"The verb '{flagged_text}' may benefit from present tense for clarity."

    def _generate_smart_past_tense_suggestions(self, root_verb: Token, ev: float, context: Dict[str, Any], doc: Doc = None) -> List[str]:
        """Generate context-aware suggestions for past tense patterns with proper subject-verb agreement."""
        
        base = root_verb.lemma_
        suggestions = []
        content_type = context.get('content_type', 'general')
        
        # Find the subject of the verb to ensure proper conjugation
        subject_token = self._find_verb_subject(root_verb, root_verb, doc)
        
        # Base suggestions based on evidence strength and context with linguistically-aware conjugation
        if ev > 0.8:
            if content_type == 'procedural':
                suggestions.append(f"Use imperative: '{base.capitalize()}' (direct instruction)")
                if subject_token:
                    present_form = self._generate_smart_verb_suggestions(root_verb, subject_token, 'present')
                    suggestions.append(f"Use present: '{present_form}' (describe current behavior)")
                else:
                    suggestions.append(f"Use present: 'The system {self._conjugate_verb_for_subject(base, 'system')}' (describe current behavior)")
            elif content_type == 'api':
                method_form = self._conjugate_verb_for_subject(base, 'method')
                suggestions.append(f"Use present: 'The method {method_form}' (current API behavior)")
                this_form = self._conjugate_verb_for_subject(base, 'this')
                suggestions.append(f"Use present: 'This {this_form} the resource' (API functionality)")
            else:
                if subject_token:
                    present_form = self._generate_smart_verb_suggestions(root_verb, subject_token, 'present')
                    suggestions.append(f"Use present: '{present_form}' (current state/behavior)")
                    suggestions.append(f"Use imperative: '{base.capitalize()}' (direct action)")
                else:
                    suggestions.append(f"Use present with proper subject-verb agreement")
                    suggestions.append(f"Use imperative: '{base.capitalize()}' (direct action)")
        else:
            if subject_token:
                present_form = self._generate_smart_verb_suggestions(root_verb, subject_token, 'present')
                suggestions.append(f"Consider present: '{present_form}'")
            else:
                suggestions.append(f"Consider present with proper subject-verb agreement")
            suggestions.append("Consider if this describes current vs. historical behavior")
        
        # Context-specific advice
        if content_type == 'procedural':
            suggestions.append("Use imperative mood for clear step-by-step instructions.")
        elif content_type == 'api':
            suggestions.append("Describe current API behavior in present tense.")
        elif content_type == 'technical':
            suggestions.append("Technical documentation should describe current system behavior.")
        elif content_type == 'tutorial':
            suggestions.append("Tutorial steps should be in imperative mood.")
        else:
            suggestions.append("Use present tense for current behavior and instructions.")
        
        return suggestions[:3]

    # === LINGUISTICALLY-AWARE VERB SUGGESTION METHODS ===

    def _generate_smart_verb_suggestions(self, verb_token: Token, subject_token: Token, desired_tense: str = 'present') -> str:
        """
        Generates a grammatically correct verb suggestion.
        
        Args:
            verb_token: The verb to change (e.g., 'find')
            subject_token: Its subject (e.g., 'you')
            desired_tense: The desired tense ('present', 'past', etc.)
            
        Returns:
            str: Correctly conjugated verb form
        """
        try:
            verb_lemma = verb_token.lemma_
            subject_text = subject_token.text.lower() if subject_token else 'it'
            
            # Determine the correct verb form based on subject
            if subject_text in ['i', 'you', 'we', 'they'] or (hasattr(subject_token, 'tag_') and subject_token.tag_ == 'NNS'):
                # 1st/2nd person or plural subject -> use base form for present tense (VBP)
                tag = 'VBP' if desired_tense == 'present' else 'VBD'
            else:
                # 3rd person singular subject -> use '-s' form for present tense (VBZ)
                tag = 'VBZ' if desired_tense == 'present' else 'VBD'
            
            # Get the correctly conjugated verb using pyinflect
            conjugated_verb = pyinflect.getInflection(verb_lemma, tag=tag)
            
            if conjugated_verb and len(conjugated_verb) > 0:
                # Combine subject and correctly conjugated verb
                subject_pronoun = subject_token.text if subject_token else 'it'
                return f"{subject_pronoun} {conjugated_verb[0]}"
            else:
                # Fallback to manual conjugation
                correctly_conjugated = self._conjugate_verb_for_subject(verb_lemma, subject_text)
                subject_pronoun = subject_token.text if subject_token else 'it'
                return f"{subject_pronoun} {correctly_conjugated}"
                
        except Exception:
            # Fallback for any errors
            verb_lemma = verb_token.lemma_ if hasattr(verb_token, 'lemma_') else 'use'
            subject_text = subject_token.text if subject_token else 'it'
            correctly_conjugated = self._conjugate_verb_for_subject(verb_lemma, subject_text)
            return f"{subject_text} {correctly_conjugated}"

    def _conjugate_verb_for_subject(self, verb_lemma: str, subject: str) -> str:
        """Conjugate verb correctly based on subject."""
        subject_lower = subject.lower()
        
        if subject_lower in self.config['plural_subjects']:
            return verb_lemma
        elif subject_lower in self.config['third_person_singular_subjects']:
            return self._add_s_ending(verb_lemma)
        else:
            return verb_lemma

    def _add_s_ending(self, verb_lemma: str) -> str:
        """Add appropriate -s ending to verb for 3rd person singular."""
        if verb_lemma.endswith('y') and len(verb_lemma) > 1 and verb_lemma[-2] not in 'aeiou':
            return verb_lemma[:-1] + 'ies'
        elif verb_lemma.endswith(('s', 'sh', 'ch', 'x', 'z')):
            return verb_lemma + 'es'
        elif verb_lemma.endswith('o') and len(verb_lemma) > 1 and verb_lemma[-2] not in 'aeiou':
            return verb_lemma + 'es'
        else:
            return verb_lemma + 's'

    def _find_verb_subject(self, context_token: Token, verb_token: Token, doc: Doc = None) -> Token:
        """Find the subject of a verb using dependency parsing."""
        try:
            if hasattr(verb_token, 'children'):
                for child in verb_token.children:
                    if child.dep_ in self.config['subject_dependencies']:
                        return child
            
            if hasattr(context_token, 'head') and context_token.head:
                head = context_token.head
                if hasattr(head, 'children'):
                    for child in head.children:
                        if child.dep_ in self.config['subject_dependencies']:
                            return child
            
            if hasattr(context_token, 'head') and context_token.head:
                head = context_token.head
                if hasattr(head, 'children'):
                    for sibling in head.children:
                        if sibling.dep_ in self.config['subject_dependencies'] and sibling != context_token:
                            return sibling
            
            if hasattr(context_token, 'i') and hasattr(context_token, 'doc'):
                for i in range(max(0, context_token.i - 5), context_token.i):
                    token = context_token.doc[i]
                    if token.pos_ == 'PRON' and token.dep_ in self.config['subject_dependencies']:
                        return token
                    elif token.pos_ in ['NOUN', 'PROPN'] and token.dep_ in self.config['subject_dependencies']:
                        return token
            
            if doc or (hasattr(context_token, 'sent')):
                sentence = context_token.sent if hasattr(context_token, 'sent') else doc
                if sentence:
                    for token in sentence:
                        if (token.pos_ == 'PRON' and 
                            token.text.lower() in self.config['plural_subjects'] + self.config['third_person_singular_subjects'][:3]):
                            return token
            
            return None
            
        except Exception:
            return None

    def _is_technical_compound_noun(self, token: Token, doc: Doc) -> bool:
        """Check if token is part of a known technical compound noun."""
        corrections = self.vocabulary_service.get_verbs_corrections()
        compound_nouns = corrections.get('technical_compound_nouns', {})
        
        all_compound_heads = []
        for category, heads in compound_nouns.items():
            all_compound_heads.extend(heads)
        
        if hasattr(token, 'head') and token.head:
            head_text = token.head.text.lower()
            if head_text in all_compound_heads:
                return True
        
        if token.i < len(doc) - 1:
            next_token = doc[token.i + 1]
            next_text = next_token.text.lower()
            if next_text in all_compound_heads:
                return True
        
        return False

    def _is_release_notes_documentation(self, text: str) -> bool:
        """Check if text appears to be release notes documentation."""
        text_lower = text.lower()
        count = sum(1 for indicator in self.config['release_notes_indicators'] if indicator in text_lower)
        return count >= self.config['min_counts']['release_notes']

    def _is_roadmap_documentation(self, text: str) -> bool:
        """Check if text appears to be roadmap documentation."""
        text_lower = text.lower()
        count = sum(1 for indicator in self.config['roadmap_indicators'] if indicator in text_lower)
        return count >= self.config['min_counts']['roadmap']

    def _is_planning_documentation(self, text: str) -> bool:
        """Check if text appears to be planning documentation."""
        text_lower = text.lower()
        count = sum(1 for indicator in self.config['planning_indicators'] if indicator in text_lower)
        return count >= self.config['min_counts']['planning']

    def _is_changelog_documentation(self, text: str) -> bool:
        """Check if text appears to be changelog documentation."""
        text_lower = text.lower()
        count = sum(1 for indicator in self.config['changelog_indicators'] if indicator in text_lower)
        return count >= self.config['min_counts']['changelog']

    def _is_bug_report_documentation(self, text: str) -> bool:
        """Check if text appears to be bug report documentation."""
        text_lower = text.lower()
        count = sum(1 for indicator in self.config['bug_report_indicators'] if indicator in text_lower)
        return count >= self.config['min_counts']['bug_report']

    def _is_noun_form_as_verb_issue(self, token, doc, sentence: str) -> bool:
        """Check if token is a noun form being used incorrectly as a verb using YAML vocabulary."""
        if token.pos_ != 'VERB':
            return False
        
        # Load noun forms from YAML vocabulary
        corrections = self.vocabulary_service.get_verbs_corrections()
        noun_forms_as_verbs = corrections.get('noun_forms_as_verbs', {})
        
        token_lower = token.text.lower()
        return token_lower in noun_forms_as_verbs

    def _calculate_noun_verb_evidence(self, token, doc, sentence: str, text: str, context: dict) -> float:
        """Calculate evidence score for noun-forms-as-verbs issues."""
        evidence_score = self.config['noun_verb_base_evidence']
        content_type = context.get('content_type', '')
        w = self.config['noun_verb_content_weights']
        
        if content_type in ['documentation', 'tutorial', 'guide']:
            evidence_score += w['documentation_boost']
        
        if content_type in ['chat', 'social', 'informal']:
            evidence_score -= w['informal_reduction']
        
        return max(0.0, min(1.0, evidence_score))

    def _get_contextual_noun_verb_message(self, flagged_text: str, evidence_score: float, context: dict) -> str:
        """Generate contextual message for noun-forms-as-verbs issues using YAML vocabulary."""
        # Load corrections from YAML vocabulary
        corrections = self.vocabulary_service.get_verbs_corrections()
        noun_forms_as_verbs = corrections.get('noun_forms_as_verbs', {})
        
        flagged_lower = flagged_text.lower()
        
        if flagged_lower in noun_forms_as_verbs:
            correction_info = noun_forms_as_verbs[flagged_lower]
            correct_form = correction_info.get('correct_form', 'use proper verb form')
        else:
            correct_form = 'use proper verb form'
        
        return f"Use '{correct_form}' instead of '{flagged_text}' as a verb."

    def _generate_smart_noun_verb_suggestions(self, token, evidence_score: float, context: dict) -> List[str]:
        """Generate smart suggestions for noun-forms-as-verbs issues using YAML vocabulary."""
        # Load corrections from YAML vocabulary
        corrections = self.vocabulary_service.get_verbs_corrections()
        noun_forms_as_verbs = corrections.get('noun_forms_as_verbs', {})
        
        token_lower = token.text.lower()
        correction_info = noun_forms_as_verbs.get(token_lower)
        
        if correction_info:
            correct_form = correction_info.get('correct_form')
            if correct_form:
                # Preserve original capitalization
                if token.text[0].isupper():
                    correct_form = correct_form.capitalize()
                return [correct_form]
        
        return ['use proper verb form']

    def _is_ui_documentation(self, text: str) -> bool:
        """Check if text appears to be user interface documentation."""
        text_lower = text.lower()
        count = sum(1 for indicator in self.config['ui_documentation_indicators'] if indicator in text_lower)
        return count >= self.config['min_counts']['ui_docs']
    
    def _find_verb_subject_for_agreement(self, verb_token: Token, doc: Doc) -> Optional[Token]:
        """Find the grammatical subject of a verb for agreement checking."""
        for child in verb_token.children:
            if child.dep_ in self.config['subject_dependencies']:
                return child
        
        if verb_token.dep_ != 'ROOT' and verb_token.head.pos_ == 'VERB':
            for child in verb_token.head.children:
                if child.dep_ in self.config['subject_dependencies']:
                    return child
        
        return None
    
    def _has_subject_verb_disagreement(self, subject: Token, verb: Token, doc: Doc, corrections: Dict = None) -> bool:
        """Check if subject and verb disagree in number."""
        if corrections is None:
            corrections = {}
        
        if self.adapter:
            subject_is_plural = self.adapter.is_plural_corrected(subject, corrections)
        else:
            subject_is_plural = self._is_plural_subject(subject, doc)
        
        verb_is_plural_form = (verb.tag_ == self.config['verb_tags']['plural_base'])
        
        if subject_is_plural and verb_is_plural_form:
            return False 
        if not subject_is_plural and not verb_is_plural_form:
            return False 
        
        # GUARD 1: Pronoun "you" always takes plural verb form
        if subject.lemma_.lower() == 'you':
            return False
        
        # GUARD 2: Verbs following modal verbs always use base form
        for child in verb.children:
            if child.dep_ == 'aux' and child.tag_ == self.config['verb_tags']['modal']:
                return False
        
        # GUARD 3: Participles and infinitives never inflect for subject-verb agreement
        if verb.tag_ in [self.config['verb_tags']['gerund'], self.config['verb_tags']['past_participle'], self.config['verb_tags']['base_form']]:
            return False
        
        subject_is_plural = self._is_plural_subject(subject, doc)
        verb_is_plural = self._is_plural_verb(verb)
        
        if subject.lemma_.lower() == 'there':
            return False
        
        if self._is_collective_noun(subject):
            return False
        
        if subject_is_plural and not verb_is_plural:
            return True
        if not subject_is_plural and verb_is_plural:
            return True
        
        return False
    
    def _is_plural_subject(self, subject: Token, doc: Doc) -> bool:
        """Determine if a subject is plural."""
        has_conjunction = any(child.dep_ == 'conj' for child in subject.children)
        if has_conjunction:
            return True
        
        number = subject.morph.get('Number')
        if number:
            if 'Plur' in number:
                return True
            if 'Sing' in number:
                return False
        
        if subject.tag_ in [self.config['noun_tags']['plural_common'], self.config['noun_tags']['plural_proper']]:
            return True
        if subject.tag_ in [self.config['noun_tags']['singular_common'], self.config['noun_tags']['singular_proper']]:
            return False
        
        if subject.lemma_.lower() in self.config['plural_pronouns']:
            return True
        if subject.lemma_.lower() in self.config['singular_pronouns']:
            return False
        
        return False
    
    def _is_plural_verb(self, verb: Token) -> bool:
        """Determine if a verb is in plural form."""
        number = verb.morph.get('Number')
        if number:
            if 'Plur' in number:
                return True
            if 'Sing' in number:
                return False
        
        if verb.tag_ == self.config['verb_tags']['third_singular']:
            return False
        if verb.tag_ in [self.config['verb_tags']['plural_base'], self.config['verb_tags']['base_form']]:
            return True
        
        if verb.lemma_.lower() == 'be':
            if verb.text.lower() == 'was':
                return False
            if verb.text.lower() == 'were':
                return True
        
        if verb.tag_ in [self.config['verb_tags']['modal'], self.config['verb_tags']['base_form']]:
            return None
        
        return None
    
    def _is_collective_noun(self, token: Token) -> bool:
        """Check if token is a collective noun."""
        return token.lemma_.lower() in self.config['collective_nouns']
    
    def _calculate_agreement_evidence(
        self, subject: Token, verb: Token, doc: Doc, sent_text: str, 
        text: str, context: dict
    ) -> float:
        """Calculate evidence score for subject-verb agreement error."""
        w = self.config['agreement_weights']
        evidence_score = w['base_evidence']
        
        words_between = abs(verb.i - subject.i)
        if words_between > w['distance_threshold']:
            evidence_score -= w['distance_penalty']
        
        for child in subject.children:
            if child.dep_ == 'prep':
                evidence_score -= w['prep_phrase_penalty']
        
        if verb.i == subject.i + 1:
            evidence_score += w['immediate_follow_boost']
        
        if context.get('block_type') in ['code_block', 'literal_block']:
            evidence_score -= w['code_block_reduction']
        
        return max(0.0, min(1.0, evidence_score))
    
    def _get_contextual_agreement_message(
        self, subject: Token, verb: Token, evidence_score: float
    ) -> str:
        """Generate error message for subject-verb agreement."""
        subject_is_plural = self._is_plural_subject(subject, None)
        
        if subject_is_plural:
            return f"Subject-verb agreement: '{subject.text}' is plural but '{verb.text}' is singular. Consider using the plural form."
        else:
            return f"Subject-verb agreement: '{subject.text}' is singular but '{verb.text}' is plural. Consider using the singular form."
    
    def _generate_agreement_suggestions(
        self, subject: Token, verb: Token, doc: Doc
    ) -> List[str]:
        """Generate suggestions for fixing subject-verb agreement."""
        suggestions = []
        
        subject_is_plural = self._is_plural_subject(subject, doc)
        
        # Try to get the correct verb form using pyinflect
        if subject_is_plural:
            # Need plural form (VBP)
            correct_form = pyinflect.getInflection(verb.lemma_, tag='VBP')
            if correct_form:
                suggestions.append(f"Change '{verb.text}' to '{correct_form[0]}'")
            else:
                suggestions.append(f"Use plural verb form with '{subject.text}'")
        else:
            # Need singular form (VBZ)
            correct_form = pyinflect.getInflection(verb.lemma_, tag='VBZ')
            if correct_form:
                suggestions.append(f"Change '{verb.text}' to '{correct_form[0]}'")
            else:
                suggestions.append(f"Use singular verb form with '{subject.text}'")
        
        # Add explanation
        number_desc = "plural" if subject_is_plural else "singular"
        suggestions.append(f"The subject '{subject.text}' is {number_desc}, so the verb must agree.")
        
        return suggestions
    
    def _is_weak_verb_with_action_noun(self, token: Token) -> bool:
        """Check if token is a weak verb with an action noun object."""
        if token.pos_ == 'VERB' and token.lemma_ in self.config['weak_verbs']:
            for child in token.children:
                if child.dep_ == 'dobj' and self._is_action_noun(child):
                    return True
        return False
    
    def _get_action_noun_object(self, verb_token: Token) -> Optional[Token]:
        """Get the action noun that is the direct object of the weak verb."""
        for child in verb_token.children:
            if child.dep_ == 'dobj' and self._is_action_noun(child):
                return child
        return None
    
    def _is_action_noun(self, token: Token) -> bool:
        """Check if a noun is derived from a verb (nominalization)."""
        if token.pos_ != 'NOUN':
            return False
        
        token_lower = token.text.lower()
        if token_lower.endswith(tuple(self.config['action_noun_suffixes'])):
            return True
        
        if token_lower in self.config['verb_noun_forms']:
            return True
        
        return False
    
    def _calculate_nominalization_evidence(
        self, verb: Token, noun: Token, doc, text: str, context: dict
    ) -> float:
        """Calculate evidence score for nominalized verb construction."""
        w = self.config['nominalization_weights']
        evidence_score = w['base_evidence']
        
        content_type = context.get('content_type', 'general')
        if content_type in ['technical', 'procedural', 'reference', 'tutorial']:
            evidence_score += w['technical_procedural_boost']
        elif content_type in ['api', 'documentation']:
            evidence_score += w['api_docs_boost']
        elif content_type in ['narrative', 'blog', 'marketing']:
            evidence_score -= w['narrative_reduction']
        
        if hasattr(doc, '__len__'):
            if len(doc) > w['length_thresholds']['very_long']:
                evidence_score += w['very_long_sentence_boost']
            elif len(doc) > w['length_thresholds']['long']:
                evidence_score += w['long_sentence_boost']
        
        block_type = context.get('block_type', 'paragraph')
        if block_type in ['heading', 'title']:
            evidence_score += w['heading_boost']
        
        if block_type in ['ordered_list_item', 'unordered_list_item']:
            evidence_score += w['list_boost']
        
        if block_type in ['code_block', 'literal_block', 'inline_code']:
            evidence_score -= w['code_reduction']
        
        return max(0.0, min(1.0, evidence_score))
    
    def _get_contextual_nominalization_message(self, verb: Token, noun: Token) -> str:
        """Generate a message for a nominalized verb construction."""
        return f"Consider replacing the weak verb-noun pair '{verb.text} {noun.text}' with a stronger, more direct verb."
    
    def _generate_smart_nominalization_suggestions(self, verb: Token, noun: Token) -> List[str]:
        """Generate suggestions for de-nominalizing the verb."""
        suggestions = []
        noun_lower = noun.text.lower()
        
        if noun_lower in self.config['nominalization_to_verb']:
            strong_verb = self.config['nominalization_to_verb'][noun_lower]
            suggestions.append(f"Use a stronger verb: '{strong_verb}'. For example: 'The hardware and driver {strong_verb}...'")
        else:
            strong_verb = self._derive_verb_from_noun(noun.text)
            if strong_verb and strong_verb != noun.text.lower():
                suggestions.append(f"Rewrite using a stronger verb, such as '{strong_verb}'.")
            else:
                suggestions.append(f"Rewrite using a more direct verb instead of '{verb.text} {noun.text}'.")
        
        suggestions.append("Using direct verbs makes your writing clearer and more concise.")
        
        return suggestions[:3]
    
    def _derive_verb_from_noun(self, noun_text: str) -> Optional[str]:
        """
        Attempt to derive a verb form from a nominalized noun.
        
        Uses simple heuristics to strip common noun suffixes and add verb endings.
        
        Args:
            noun_text: The noun text
            
        Returns:
            Derived verb form, or None if unable to derive
        """
        noun_lower = noun_text.lower()
        
        # Handle -tion/-sion endings
        if noun_lower.endswith('tion'):
            base = noun_lower[:-4]  # Remove 'tion'
            # configuration → configure
            if base.endswith('a'):
                return base + 'te'
            else:
                return base + 'e'
        elif noun_lower.endswith('sion'):
            base = noun_lower[:-4]  # Remove 'sion'
            return base + 'e'
        
        # Handle -ment endings
        elif noun_lower.endswith('ment'):
            base = noun_lower[:-4]  # Remove 'ment'
            # requirement → require, deployment → deploy
            if base.endswith('e'):
                return base
            else:
                return base + 'e'
        
        # Handle -ance/-ence endings
        elif noun_lower.endswith('ance'):
            base = noun_lower[:-4]  # Remove 'ance'
            # assistance → assist, performance → perform
            if base.endswith('t'):
                return base
            else:
                return base + 'e'
        elif noun_lower.endswith('ence'):
            base = noun_lower[:-4]  # Remove 'ence'
            return base + 'e'
        
        # Handle -al endings
        elif noun_lower.endswith('al'):
            base = noun_lower[:-2]  # Remove 'al'
            # approval → approve
            return base + 'e'
        
        # Handle -ing endings (gerunds)
        elif noun_lower.endswith('ing'):
            base = noun_lower[:-3]  # Remove 'ing'
            # processing → process, logging → log
            return base
        
        # If we can't derive it, return None
        return None
