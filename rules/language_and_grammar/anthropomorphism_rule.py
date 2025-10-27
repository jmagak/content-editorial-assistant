"""
Anthropomorphism Rule
Based on IBM Style Guide: Detects inappropriate anthropomorphism using YAML-based configuration.
"""
from typing import List, Dict, Any
from .base_language_rule import BaseLanguageRule
from .services.language_vocabulary_service import get_anthropomorphism_vocabulary

try:
    from spacy.tokens import Doc
except ImportError:
    Doc = None

class AnthropomorphismRule(BaseLanguageRule):
    """Detects inappropriate anthropomorphism using dependency parsing and evidence-based scoring."""
    
    def __init__(self):
        super().__init__()
        self.vocabulary_service = get_anthropomorphism_vocabulary()
        self.config = self.vocabulary_service.get_anthropomorphism_config()
    
    def _get_rule_type(self) -> str:
        return 'anthropomorphism'

    def analyze(self, text: str, sentences: List[str], nlp=None, context=None) -> List[Dict[str, Any]]:
        """Analyze sentences for inappropriate anthropomorphism."""
        if context and context.get('block_type') in ['listing', 'literal', 'code_block', 'inline_code']:
            return []
        
        errors = []
        if not nlp:
            return errors

        doc = nlp(text)
        
        for i, sent in enumerate(doc.sents):
            for token in sent:
                anthropomorphism_data = self._detect_potential_anthropomorphism(token, sent)
                
                if anthropomorphism_data:
                    evidence_score = self._calculate_anthropomorphism_evidence(
                        token, anthropomorphism_data['subject'], sent, text, context
                    )
                    
                    if evidence_score > self.config['evidence_thresholds']['min_threshold']:
                        errors.append(self._create_error(
                            sentence=sent.text,
                            sentence_index=i,
                            message=self._get_contextual_message(token, anthropomorphism_data['subject'], evidence_score),
                            suggestions=self._generate_smart_suggestions(token, anthropomorphism_data['subject'], evidence_score, context),
                            severity='low',
                            text=text,
                            context=context,
                            evidence_score=evidence_score,
                            span=(anthropomorphism_data['subject'].idx, token.idx + len(token.text)),
                            flagged_text=f"{anthropomorphism_data['subject'].text} {token.text}"
                        ))
        return errors

    def _detect_potential_anthropomorphism(self, token, sent):
        """Detect potential anthropomorphic patterns."""
        if not self._could_be_anthropomorphic_verb(token):
            return None
            
        subject_tokens = [child for child in token.children if child.dep_ == self.config['dependency_relations']['subject']]
        if not subject_tokens:
            return None
            
        subject = subject_tokens[0]
        
        if self._could_be_inappropriately_anthropomorphized(subject):
            return {'subject': subject, 'verb': token}
        
        return None

    def _could_be_anthropomorphic_verb(self, token):
        """Check if token could be a verb that anthropomorphizes inanimate subjects using YAML vocabulary."""
        
        # Load verb categories from YAML vocabulary
        entities_vocab = self.vocabulary_service.get_anthropomorphism_entities()
        anthropomorphic_verbs = entities_vocab.get('anthropomorphic_verbs', {})
        
        # Extract verb sets from YAML
        core_human_verbs = set(anthropomorphic_verbs.get('core_human_verbs', []))
        communication_verbs = set(anthropomorphic_verbs.get('communication_verbs', []))
        decision_verbs = set(anthropomorphic_verbs.get('decision_verbs', []))
        perception_verbs = set(anthropomorphic_verbs.get('perception_verbs', []))
        action_verbs = set(anthropomorphic_verbs.get('action_verbs', []))
        
        # Load technical verbs that are acceptable
        technical_verbs = set(entities_vocab.get('technical_verbs', {}).get('acceptable_system_verbs', []))
        
        verb_lemma = token.lemma_.lower()
        
        # Don't flag technical verbs that are acceptable for systems
        if verb_lemma in technical_verbs:
            return False
        
        # Return True if it's any kind of potentially anthropomorphic verb
        return (verb_lemma in core_human_verbs or 
                verb_lemma in communication_verbs or 
                verb_lemma in decision_verbs or 
                verb_lemma in perception_verbs or 
                verb_lemma in action_verbs)

    def _could_be_inappropriately_anthropomorphized(self, subject):
        """Check if subject could be inappropriately anthropomorphized using YAML vocabulary."""
        
        # Load entities from YAML vocabulary
        entities_vocab = self.vocabulary_service.get_anthropomorphism_entities()
        technical_entities = set(entities_vocab.get('technical_entities', {}).get('systems', []))
        hardware_entities = set(entities_vocab.get('technical_entities', {}).get('infrastructure', []))
        interface_entities = set(entities_vocab.get('technical_entities', {}).get('interfaces', []))
        content_entities = set(entities_vocab.get('inanimate_entities', {}).get('objects', []))
        business_entities = set(entities_vocab.get('inanimate_entities', {}).get('concepts', []))
        
        subject_lemma = subject.lemma_.lower()
        
        # Check if it's a potentially anthropomorphizable entity
        return (subject_lemma in technical_entities or
                subject_lemma in hardware_entities or
                subject_lemma in interface_entities or
                subject_lemma in content_entities or
                subject_lemma in business_entities)

    def _calculate_anthropomorphism_evidence(self, verb_token, subject_token, sentence, text: str, context: dict) -> float:
        """Calculate evidence score for inappropriate anthropomorphism."""
        evidence_score = self._get_base_verb_evidence(verb_token)
        evidence_score = self._apply_linguistic_clues_anthropomorphism(evidence_score, verb_token, subject_token, sentence)
        evidence_score = self._apply_structural_clues_anthropomorphism(evidence_score, verb_token, subject_token, context)
        evidence_score = self._apply_semantic_clues_anthropomorphism(evidence_score, verb_token, subject_token, text, context)
        evidence_score = self._apply_feedback_clues_anthropomorphism(evidence_score, verb_token, subject_token, context)
        
        verb_lemma = verb_token.lemma_.lower()
        if verb_lemma in self.config['decision_verbs']:
            min_threshold = self.config['evidence_thresholds']['decision_verb_minimum']
            if 0.0 < evidence_score < min_threshold:
                evidence_score = min_threshold
        
        return max(0.0, min(1.0, evidence_score))

    def _get_base_verb_evidence(self, verb_token) -> float:
        """Get base evidence score based on verb category."""
        verb_lemma = verb_token.lemma_.lower()
        verb_categories = self.config['verb_categories']
        base_evidence = self.config['base_verb_evidence']
        
        for category, verbs in verb_categories.items():
            if verb_lemma in verbs:
                return base_evidence[category]
        
        return base_evidence['default']

    def _apply_linguistic_clues_anthropomorphism(self, evidence_score: float, verb_token, subject_token, sentence) -> float:
        """Apply linguistic analysis clues for anthropomorphism."""
        w = self.config['linguistic_weights']
        
        subject_lemma = subject_token.lemma_.lower()
        verb_lemma = verb_token.lemma_.lower()
        
        if subject_lemma in self.config['technical_subjects'] and verb_lemma in self.config['cognitive_agent_verbs']:
            evidence_score += w['technical_cognitive_boost']
        
        if verb_token.tag_ in self.config['pos_tags']['present_tense']:
            evidence_score += w['present_tense_boost']
        elif verb_token.tag_ in self.config['pos_tags']['past_tense']:
            evidence_score -= w['past_tense_reduction']
        elif verb_token.tag_ in self.config['pos_tags']['present_participle']:
            evidence_score += w['present_participle_boost']
        
        if hasattr(verb_token, 'morph') and verb_token.morph:
            morph_str = str(verb_token.morph)
            m = self.config['morph_features']
            
            if m['tense_present'] in morph_str:
                evidence_score += w['tense_pres_boost']
            elif m['tense_past'] in morph_str:
                evidence_score -= w['tense_past_reduction']
            
            if m['person_3'] in morph_str:
                evidence_score += w['person_3_boost']
            
            if m['number_sing'] in morph_str:
                evidence_score += w['number_sing_boost']
            
            if m['voice_pass'] in morph_str:
                evidence_score -= w['voice_pass_reduction']
        
        dep_rel = self.config['dependency_relations']
        if verb_token.dep_ == dep_rel['root']:
            evidence_score += w['root_verb_boost']
        elif verb_token.dep_ in dep_rel['complement']:
            evidence_score -= w['complement_reduction']
        elif verb_token.dep_ in dep_rel['relative_clause']:
            evidence_score -= w['relative_clause_reduction']
        
        if hasattr(subject_token, 'morph') and subject_token.morph:
            subj_morph = str(subject_token.morph)
            if self.config['morph_features']['number_plur'] in subj_morph:
                evidence_score -= w['plural_subject_reduction']
        
        if subject_token.tag_ in self.config['pos_tags']['proper_nouns']:
            if subject_token.ent_type_ in self.config['entity_types']['semi_animate']:
                evidence_score -= w['proper_noun_product_reduction']
        elif subject_token.tag_ in self.config['pos_tags']['common_nouns']:
            evidence_score += w['common_noun_boost']
        
        if subject_token.ent_type_:
            if hasattr(subject_token, 'ent_iob_'):
                if subject_token.ent_iob_ == self.config['entity_iob']['beginning']:
                    evidence_score -= w['entity_iob_b_reduction']
                elif subject_token.ent_iob_ == self.config['entity_iob']['inside']:
                    evidence_score -= w['entity_iob_i_reduction']
        
        if hasattr(subject_token, 'pos_'):
            if subject_token.pos_ == self.config['pos_categories']['proper_noun']:
                evidence_score -= w['propn_reduction']
            elif subject_token.pos_ == self.config['pos_categories']['common_noun']:
                evidence_score += w['noun_boost']
            elif subject_token.pos_ == self.config['pos_categories']['pronoun']:
                evidence_score -= w['pron_reduction']
        
        if hasattr(verb_token, 'pos_'):
            if verb_token.pos_ == self.config['pos_categories']['verb']:
                evidence_score += w['main_verb_boost']
            elif verb_token.pos_ == self.config['pos_categories']['auxiliary']:
                evidence_score -= w['aux_verb_reduction']
        
        if subject_token.ent_type_ in self.config['entity_types']['animate']:
            evidence_score -= w['person_org_reduction']
        elif subject_token.ent_type_ in self.config['entity_types']['semi_animate']:
            evidence_score -= w['product_fac_reduction']
        
        direct_objects = [child for child in verb_token.children if child.dep_ == dep_rel['direct_object']]
        if direct_objects:
            obj = direct_objects[0]
            if obj.ent_type_ in self.config['entity_types']['animate']:
                evidence_score += w['person_object_boost']
            elif obj.lemma_.lower() in self.config['technical_objects']:
                evidence_score -= w['technical_object_reduction']
        
        adverbs = [child for child in verb_token.children if child.dep_ == dep_rel['adverbial_modifier']]
        for adv in adverbs:
            if adv.lemma_.lower() in self.config['technical_adverbs']:
                evidence_score -= w['technical_adverb_reduction']
            elif adv.lemma_.lower() in self.config['humanlike_adverbs']:
                evidence_score += w['humanlike_adverb_boost']
        
        auxiliaries = [child for child in verb_token.children if child.dep_ == dep_rel['auxiliary']]
        for aux in auxiliaries:
            if aux.lemma_.lower() in self.config['modal_auxiliaries']['technical']:
                evidence_score -= w['modal_aux_reduction']
            elif aux.lemma_.lower() in self.config['modal_auxiliaries']['humanlike']:
                evidence_score += w['conditional_aux_boost']
        
        return evidence_score

    def _apply_structural_clues_anthropomorphism(self, evidence_score: float, verb_token, subject_token, context: dict) -> float:
        """Apply document structure clues for anthropomorphism."""
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
        elif block_type in ['ordered_list_item', 'unordered_list_item']:
            evidence_score -= w['list_reduction']
            if context.get('list_depth', 1) > 1:
                evidence_score -= w['nested_list_reduction']
        elif block_type == 'heading':
            heading_level = context.get('block_level', 1)
            if heading_level <= 2:
                evidence_score -= w['h1_h2_reduction']
            else:
                evidence_score -= w['h3_plus_reduction']
        elif block_type == 'admonition':
            admonition_type = context.get('admonition_type', '').upper()
            if admonition_type in self.config['admonition_types']['informational']:
                evidence_score -= w['note_tip_reduction']
            elif admonition_type in self.config['admonition_types']['warning']:
                evidence_score -= w['warning_caution_reduction']
        
        return evidence_score

    def _apply_semantic_clues_anthropomorphism(self, evidence_score: float, verb_token, subject_token, text: str, context: dict) -> float:
        """Apply semantic and content-type clues for anthropomorphism."""
        if not context:
            return evidence_score
        
        w = self.config['semantic_weights']
        content_type = context.get('content_type', 'general')
        
        if self._is_api_documentation(text):
            evidence_score -= w['api_docs_reduction']
        elif self._is_technical_specification(text, context):
            evidence_score -= w['tech_spec_reduction']
        elif self._is_user_interface_documentation(text):
            evidence_score -= w['ui_docs_reduction']
        elif self._is_system_administration_content(text, context):
            evidence_score -= w['sysadmin_reduction']
        elif self._is_software_architecture_content(text, context):
            evidence_score -= w['architecture_reduction']
        elif self._is_troubleshooting_content(text, context):
            evidence_score -= w['troubleshooting_reduction']
        elif content_type == 'technical':
            evidence_score -= w['technical_content_reduction']
            if self._has_technical_context_words(verb_token.sent.text, distance=10):
                evidence_score -= w['technical_context_reduction']
        elif content_type == 'api':
            evidence_score -= w['api_content_reduction']
        elif content_type == 'procedural':
            evidence_score -= w['procedural_reduction']
        elif content_type == 'academic':
            evidence_score += w['academic_boost']
        elif content_type == 'legal':
            evidence_score += w['legal_boost']
        elif content_type == 'marketing':
            evidence_score -= w['marketing_reduction']
        elif content_type == 'narrative':
            evidence_score -= w['narrative_reduction']
        
        domain = context.get('domain', 'general')
        if domain in ['software', 'engineering', 'devops', 'api']:
            evidence_score -= w['technical_domain_reduction']
        elif domain in ['documentation', 'tutorial']:
            evidence_score -= w['documentation_domain_reduction']
        
        audience = context.get('audience', 'general')
        if audience in ['developer', 'technical', 'expert']:
            evidence_score -= w['technical_audience_reduction']
        elif audience in ['academic', 'scientific']:
            evidence_score += w['academic_audience_boost']
        
        verb_lemma = verb_token.lemma_.lower()
        subject_lemma = subject_token.lemma_.lower()
        
        if [subject_lemma, verb_lemma] in self.config['conventional_patterns']:
            evidence_score -= w['conventional_pattern_reduction']
        
        if [subject_lemma, verb_lemma] in self.config['problematic_patterns']:
            evidence_score += w['problematic_pattern_boost']
        
        return evidence_score

    def _apply_feedback_clues_anthropomorphism(self, evidence_score: float, verb_token, subject_token, context: dict) -> float:
        """Apply feedback patterns for anthropomorphism."""
        w = self.config['feedback_weights']
        feedback_patterns = self._get_cached_feedback_patterns('anthropomorphism')
        
        verb_lemma = verb_token.lemma_.lower()
        subject_lemma = subject_token.lemma_.lower()
        pattern = f"{subject_lemma} {verb_lemma}"
        
        accepted_patterns = feedback_patterns.get('accepted_anthropomorphic_patterns', set())
        if pattern in accepted_patterns:
            evidence_score -= w['accepted_pattern_reduction']
        
        flagged_patterns = feedback_patterns.get('flagged_anthropomorphic_patterns', set())
        if pattern in flagged_patterns:
            evidence_score += w['flagged_pattern_boost']
        
        verb_acceptance = feedback_patterns.get('verb_acceptance_rates', {})
        acceptance_rate = verb_acceptance.get(verb_lemma, 0.5)
        
        if acceptance_rate > w['acceptance_thresholds']['high']:
            evidence_score -= w['high_verb_acceptance_reduction']
        elif acceptance_rate < w['acceptance_thresholds']['low']:
            evidence_score += w['low_verb_acceptance_boost']
        
        subject_acceptance = feedback_patterns.get('subject_acceptance_rates', {})
        subject_rate = subject_acceptance.get(subject_lemma, 0.5)
        
        if subject_rate > w['acceptance_thresholds']['high']:
            evidence_score -= w['high_subject_acceptance_reduction']
        elif subject_rate < w['acceptance_thresholds']['low']:
            evidence_score += w['low_subject_acceptance_boost']
        
        return evidence_score

    def _is_technical_specification(self, text: str, context: dict) -> bool:
        """Check if content is technical specification."""
        domain = context.get('domain', '')
        content_type = context.get('content_type', '')
        
        if content_type in ['specification', 'technical'] or domain in ['engineering', 'technical']:
            return True
        
        config = self.config['content_indicators']['technical_specification']
        text_lower = text.lower()
        count = sum(1 for indicator in config['indicators'] if indicator in text_lower)
        return count >= config['min_count']

    def _is_system_administration_content(self, text: str, context: dict) -> bool:
        """Check if content is system administration related."""
        domain = context.get('domain', '')
        content_type = context.get('content_type', '')
        
        if domain in ['sysadmin', 'devops', 'administration'] or content_type in ['administration', 'devops']:
            return True
        
        config = self.config['content_indicators']['system_administration']
        text_lower = text.lower()
        count = sum(1 for indicator in config['indicators'] if indicator in text_lower)
        return count >= config['min_count']

    def _is_software_architecture_content(self, text: str, context: dict) -> bool:
        """Check if content is software architecture related."""
        domain = context.get('domain', '')
        content_type = context.get('content_type', '')
        
        if domain in ['architecture', 'software'] or content_type in ['architecture', 'design']:
            return True
        
        config = self.config['content_indicators']['software_architecture']
        text_lower = text.lower()
        count = sum(1 for indicator in config['indicators'] if indicator in text_lower)
        return count >= config['min_count']

    def _is_troubleshooting_content(self, text: str, context: dict) -> bool:
        """Check if content is troubleshooting/debugging related."""
        content_type = context.get('content_type', '')
        domain = context.get('domain', '')
        
        if content_type in ['troubleshooting', 'debugging'] or domain in ['support', 'troubleshooting']:
            return True
        
        config = self.config['content_indicators']['troubleshooting']
        text_lower = text.lower()
        count = sum(1 for indicator in config['indicators'] if indicator in text_lower)
        return count >= config['min_count']


    def _get_contextual_message(self, verb_token, subject_token, evidence_score: float) -> str:
        """Generate context-aware error messages."""
        verb = verb_token.text
        subject = subject_token.text
        thresholds = self.config['evidence_thresholds']
        
        if evidence_score > thresholds['high_inappropriate']:
            return f"Avoid anthropomorphism: '{subject}' {verb}' gives human characteristics to an inanimate object."
        elif evidence_score > thresholds['moderate_inappropriate']:
            return f"Consider rephrasing: '{subject} {verb}' may be overly anthropomorphic for this context."
        else:
            return f"The phrase '{subject} {verb}' could be less anthropomorphic depending on your style guide."

    def _generate_smart_suggestions(self, verb_token, subject_token, evidence_score: float, context: dict) -> List[str]:
        """Generate context-aware suggestions."""
        suggestions = []
        verb_lemma = verb_token.lemma_.lower()
        subject_text = subject_token.text
        
        if verb_lemma in self.config['verb_alternatives']:
            alternatives = self.config['verb_alternatives'][verb_lemma]
            suggestions.append(f"Replace '{verb_token.text}' with a more technical verb: {alternatives}.")
        
        suggestions.append("Focus on what the system does rather than what it 'thinks' or 'feels'.")
        
        if context:
            content_type = context.get('content_type', 'general')
            
            if content_type == 'technical':
                suggestions.append("In technical writing, describe system behavior objectively.")
            elif content_type == 'api':
                suggestions.append("API documentation should describe functionality, not intentions.")
            elif content_type == 'procedural':
                suggestions.append("Focus on the actions users take and system responses.")
        
        thresholds = self.config['evidence_thresholds']
        if evidence_score > 0.7:
            suggestions.append("Consider completely rewriting the sentence to avoid anthropomorphism.")
            suggestions.append(f"Instead of '{subject_text} {verb_token.text}', describe the actual process or user action.")
        elif evidence_score < thresholds['low_inappropriate']:
            suggestions.append("This usage may be acceptable in your context, but consider your style guide.")
        
        return suggestions
