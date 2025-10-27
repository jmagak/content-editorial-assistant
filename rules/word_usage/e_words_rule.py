"""
Word Usage Rule for words starting with 'E'.
Enhanced with spaCy PhraseMatcher for efficient pattern detection.
"""
from typing import List, Dict, Any
from .base_word_usage_rule import BaseWordUsageRule
import re

try:
    from spacy.tokens import Doc
except ImportError:
    Doc = None

class EWordsRule(BaseWordUsageRule):
    """
    Checks for the incorrect usage of specific words starting with 'E'.
    Enhanced with spaCy PhraseMatcher for efficient detection.
    Includes consistency checking for 'enter' vs 'type' usage.
    """
    
    def __init__(self):
        super().__init__()
        # Document-level state for consistency checking
        self.enter_type_usage = {'enter': [], 'type': []}  # Track locations
        self.current_document_id = None
    
    def _get_rule_type(self) -> str:
        return 'word_usage_e'
    
    def _reset_document_state(self, context: dict = None) -> None:
        """Reset consistency tracking for new documents."""
        document_id = context.get('source_location', '') if context else ''
        if self.current_document_id != document_id:
            self.enter_type_usage = {'enter': [], 'type': []}
            self.current_document_id = document_id

    def analyze(self, text: str, sentences: List[str], nlp=None, context=None) -> List[Dict[str, Any]]:
        """
        Evidence-based analysis for E-word usage violations.
        Computes a nuanced evidence score per occurrence considering linguistic,
        structural, semantic, and feedback clues.
        
        NOTE: 'enter' is NOT flagged as incorrect. Per IBM Style Guide:
        "Use 'enter' or 'type' to refer to entering text in fields" - use one consistently.
        We track consistency but don't prohibit 'enter'.
        """
        # === UNIVERSAL CODE CONTEXT GUARD ===
        # Skip analysis for code blocks, listings, and literal blocks (technical syntax, not prose)
        if context and context.get('block_type') in ['listing', 'literal', 'code_block', 'inline_code']:
            return []
        errors: List[Dict[str, Any]] = []
        if not nlp:
            return errors
        
        # Reset state for new documents
        self._reset_document_state(context)
            
        doc = nlp(text)
        
        # Define E-word patterns with evidence categories
        # NOTE: "enter" is REMOVED per IBM Style Guide guidance
        e_word_patterns = {
            "e-business": {"alternatives": ["business"], "category": "outdated_term", "severity": "high"},
            "e-mail": {"alternatives": ["email"], "category": "hyphenation", "severity": "high"},
            "easy": {"alternatives": ["simple", "straightforward"], "category": "subjective_claim", "severity": "high"},
            "effortless": {"alternatives": ["simple", "quick"], "category": "subjective_claim", "severity": "high"},
            "e.g.": {"alternatives": ["for example"], "category": "abbreviation", "severity": "medium"},
            "enable": {"alternatives": ["you can", "turn on"], "category": "user_focus", "severity": "medium"},
            "end user": {"alternatives": ["user"], "category": "redundant", "severity": "medium"},
            # "enter" REMOVED - both "enter" and "type" are acceptable per IBM Style Guide
            "etc": {"alternatives": ["and others", "and more"], "category": "abbreviation", "severity": "medium"},
            "execute": {"alternatives": ["run", "start"], "category": "word_choice", "severity": "low"},
        }
        
        # Track 'enter' vs 'type' usage for consistency checking
        self._track_enter_type_usage(doc, text)

        # Evidence-based analysis for E-words using lemma-based matching and phrase detection
        
        # 1. Single-word matches
        for token in doc:
            # Check if token lemma matches any of our target words
            token_lemma = token.lemma_.lower()
            token_text = token.text.lower()
            matched_pattern = None
            
            # Check exact lemma matches first (single words)
            if token_lemma in e_word_patterns and ' ' not in token_lemma and '-' not in token_lemma:
                matched_pattern = token_lemma
            # Also check for exact text matches (for abbreviations like "e.g.", "etc.")
            elif token_text in e_word_patterns and ' ' not in token_text and '-' not in token_text:
                matched_pattern = token_text
            
            if matched_pattern:
                details = e_word_patterns[matched_pattern]
                
                # === CRITICAL FIX: Skip comparative and superlative forms ===
                # "easier", "easiest" are valid comparative/superlative forms - don't flag them
                # Only flag the base adjective form "easy" when used as subjective claim
                if details["category"] == "subjective_claim":
                    # Check if this is a comparative (-er) or superlative (-est) form
                    if token.tag_ in ['JJR', 'RBR']:  # Comparative adjective/adverb
                        continue
                    if token.tag_ in ['JJS', 'RBS']:  # Superlative adjective/adverb
                        continue
                    # Also check token text directly for -er/-est endings
                    if token_text.endswith('er') and token_lemma + 'er' == token_text:
                        continue  # This is a comparative form (easy -> easier)
                    if token_text.endswith('est') and token_lemma + 'est' == token_text:
                        continue  # This is a superlative form (easy -> easiest)
                
                # Apply surgical guards with exception for abbreviations we want to flag
                should_skip = self._apply_surgical_zero_false_positive_guards_word_usage(token, context or {})
                
                # Override guard for abbreviations in our patterns - we want to flag these
                if should_skip and matched_pattern in ['e.g.', 'etc']:
                    # Check if this is actually our target abbreviation, not a legitimate entity
                    if self._is_target_abbreviation_e_words(token, matched_pattern):
                        should_skip = False
                
                if should_skip:
                    continue
                
                sent = token.sent
                sentence_index = 0
                for i, s in enumerate(doc.sents):
                    if s == sent:
                        sentence_index = i
                        break
                
                evidence_score = self._calculate_e_word_evidence(
                    matched_pattern, token, sent, text, context or {}, details["category"]
                )
                
                if evidence_score > 0.1:
                    errors.append(self._create_error(
                        sentence=sent.text,
                        sentence_index=sentence_index,
                        message=self._generate_evidence_aware_word_usage_message(matched_pattern, evidence_score, details["category"]),
                        suggestions=self._generate_evidence_aware_word_usage_suggestions(matched_pattern, details["alternatives"], evidence_score, context or {}, details["category"]),
                        severity=details["severity"] if evidence_score < 0.7 else 'high',
                        text=text,
                        context=context,
                        evidence_score=evidence_score,
                        span=(token.idx, token.idx + len(token.text)),
                        flagged_text=token.text
                    ))
        
        # 2. Multi-word and hyphenated phrase detection (e-mail, end user)
        multi_word_patterns = {pattern: details for pattern, details in e_word_patterns.items() if ' ' in pattern or '-' in pattern}
        
        if multi_word_patterns:
            # Handle hyphenated words (e-mail, e-business) - need special logic for spaCy tokenization
            hyphen_patterns = {pattern: details for pattern, details in multi_word_patterns.items() if '-' in pattern}
            for pattern, details in hyphen_patterns.items():
                # For patterns like "e-mail", look for "e", "-", "mail" sequence
                pattern_parts = pattern.split('-')
                if len(pattern_parts) == 2:
                    first_part, second_part = pattern_parts
                    
                    # Scan through tokens looking for the pattern
                    for i in range(len(doc) - 2):
                        if (doc[i].text.lower() == first_part and 
                            doc[i + 1].text == '-' and 
                            doc[i + 2].text.lower() == second_part):
                            
                            # Found the pattern
                            start_token = doc[i]
                            end_token = doc[i + 2]
                            
                            # Apply surgical guards
                            if self._apply_surgical_zero_false_positive_guards_word_usage(start_token, context or {}):
                                continue
                            
                            sent = start_token.sent
                            sentence_index = 0
                            for j, s in enumerate(doc.sents):
                                if s == sent:
                                    sentence_index = j
                                    break
                            
                            evidence_score = self._calculate_e_word_evidence(
                                pattern, start_token, sent, text, context or {}, details["category"]
                            )
                            
                            if evidence_score > 0.1:
                                errors.append(self._create_error(
                                    sentence=sent.text,
                                    sentence_index=sentence_index,
                                    message=self._generate_evidence_aware_word_usage_message(pattern, evidence_score, details["category"]),
                                    suggestions=self._generate_evidence_aware_word_usage_suggestions(pattern, details["alternatives"], evidence_score, context or {}, details["category"]),
                                    severity=details["severity"] if evidence_score < 0.7 else 'high',
                                    text=text,
                                    context=context,
                                    evidence_score=evidence_score,
                                    span=(start_token.idx, end_token.idx + len(end_token.text)),
                                    flagged_text=f"{start_token.text}{doc[i + 1].text}{end_token.text}"
                                ))
            
            # Handle space-separated phrases (end user)
            space_patterns = {pattern: details for pattern, details in multi_word_patterns.items() if ' ' in pattern}
            if space_patterns:
                phrase_matches = self._find_multi_word_phrases_with_lemma(doc, list(space_patterns.keys()), case_sensitive=False)
                
                for match in phrase_matches:
                    pattern = match['phrase']
                    details = space_patterns[pattern]
                    
                    # Apply surgical guards on the first token
                    if self._apply_surgical_zero_false_positive_guards_word_usage(match['start_token'], context or {}):
                        continue
                    
                    sent = match['start_token'].sent
                    sentence_index = 0
                    for i, s in enumerate(doc.sents):
                        if s == sent:
                            sentence_index = i
                            break
                    
                    evidence_score = self._calculate_e_word_evidence(
                        pattern, match['start_token'], sent, text, context or {}, details["category"]
                    )
                    
                    if evidence_score > 0.1:
                        errors.append(self._create_error(
                            sentence=sent.text,
                            sentence_index=sentence_index,
                            message=self._generate_evidence_aware_word_usage_message(pattern, evidence_score, details["category"]),
                            suggestions=self._generate_evidence_aware_word_usage_suggestions(pattern, details["alternatives"], evidence_score, context or {}, details["category"]),
                            severity=details["severity"] if evidence_score < 0.7 else 'high',
                            text=text,
                            context=context,
                            evidence_score=evidence_score,
                            span=(match['start_char'], match['end_char']),
                            flagged_text=match['actual_text']
                        ))
        
        # Check for 'enter' vs 'type' consistency (IBM Style Guide compliance)
        errors = self._check_enter_type_consistency(errors, doc, text, context or {})
        
        return errors

    def _calculate_e_word_evidence(self, word: str, token, sentence, text: str, context: Dict[str, Any], category: str) -> float:
        """Calculate evidence score for E-word usage violations."""
        evidence_score = self._get_base_e_word_evidence_score(word, category, sentence, context)
        
        if evidence_score == 0.0:
            return 0.0
        
        evidence_score = self._apply_linguistic_clues_e_words(evidence_score, word, token, sentence)
        evidence_score = self._apply_structural_clues_e_words(evidence_score, context)
        evidence_score = self._apply_semantic_clues_e_words(evidence_score, word, text, context)
        evidence_score = self._apply_feedback_clues_e_words(evidence_score, word, context)
        
        return max(0.0, min(1.0, evidence_score))
    
    def _get_base_e_word_evidence_score(self, word: str, category: str, sentence, context: Dict[str, Any]) -> float:
        """Set dynamic base evidence score based on E-word category."""
        if category in ['subjective_claim', 'outdated_term']:
            return 0.95  # "easy", "effortless", "e-business" - high priority
        elif category == 'hyphenation':
            return 0.9  # "e-mail" vs "email"
        elif category == 'user_focus':
            return 0.8  # "enable" - shifts focus from user
        elif category in ['abbreviation', 'redundant']:
            return 0.7  # "e.g.", "etc.", "end user"
        elif category in ['action_clarity', 'word_choice']:
            return 0.55  # "enter", "execute"
        return 0.6

    def _apply_linguistic_clues_e_words(self, ev: float, word: str, token, sentence) -> float:
        """Apply E-word-specific linguistic clues."""
        sent_text = sentence.text.lower()
        word_lower = word.lower()
        
        # Subjective claim context
        if word_lower in ['easy', 'effortless']:
            claim_indicators = ['very', 'extremely', 'incredibly', 'so']
            if any(claim in sent_text for claim in claim_indicators):
                ev += 0.2  # Intensifiers make subjective claims worse
        
        # User action context
        if word_lower in ['enable', 'execute']:  # 'enter' removed - acceptable per IBM Style Guide
            action_indicators = ['user', 'you', 'click', 'select']
            if any(action in sent_text for action in action_indicators):
                ev += 0.1  # User-facing language needs precision
        
        # Business/marketing context
        if word_lower == 'e-business':
            business_indicators = ['solution', 'service', 'platform']
            if any(biz in sent_text for biz in business_indicators):
                ev += 0.15  # Outdated business terms particularly problematic
        
        return ev

    def _apply_structural_clues_e_words(self, ev: float, context: Dict[str, Any]) -> float:
        """Apply structural context clues for E-words."""
        block_type = context.get('block_type', 'paragraph')
        
        if block_type in ['step', 'procedure']:
            ev += 0.1  # Procedural content needs precision
        elif block_type == 'heading':
            ev -= 0.1  # Headings more flexible
        
        return ev

    def _apply_semantic_clues_e_words(self, ev: float, word: str, text: str, context: Dict[str, Any]) -> float:
        """
        Apply semantic and content-type clues for E-words.
        
        WORLD-CLASS ENHANCEMENT: Domain-aware guard for technical terminology.
        This method now includes sophisticated detection of technical contexts where
        words like 'enable' and 'execute' are correct, precise terminology rather than
        user-focus violations.
        """
        content_type = context.get('content_type', 'general')
        word_lower = word.lower()
        
        # === DOMAIN-AWARE GUARD for "enable" and "execute" ===
        # In technical domains (firmware, drivers, APIs, system states), "enable" and "execute" 
        # are the correct, precise terminology. This guard detects those contexts and prevents false positives.
        if word_lower in ['enable', 'execute']:
            # Check if this word appears in a technical/system context
            if self._is_technical_domain_context_e_words(word, text, context):
                # Strong suppression - this is correct technical terminology
                ev -= 0.95
                return max(0.0, ev)  # Return early if we've determined this is technical usage
        
        # === SEMANTIC CLUE: Technical/Formal Context ===
        # Drastically reduce evidence for formal/technical documentation
        # where descriptive verbs like 'enable' and 'execute' are standard.
        if content_type in {'api', 'technical', 'reference', 'legal', 'academic', 'procedure', 'procedural'}:
            ev -= 0.95  # Maximum penalty to ensure complete suppression in technical contexts
        
        if content_type == 'tutorial':
            if word_lower in ['easy', 'effortless', 'enable']:
                ev += 0.2  # Tutorials should avoid subjective claims and focus on users
        elif content_type == 'marketing':
            if word_lower in ['easy', 'effortless', 'e-business']:
                ev += 0.15  # Marketing copy needs modern, precise language
        elif content_type == 'technical':
            if word_lower in ['e.g.', 'etc.', 'execute']:
                ev += 0.1  # Technical docs need formal language
        
        return ev

    def _is_technical_domain_context_e_words(self, word: str, text: str, context: Dict[str, Any]) -> bool:
        """
        WORLD-CLASS GUARD: Detect if "enable" or "execute" appears in a technical domain context
        where it is the correct, precise terminology.
        
        This guard checks for technical keywords in the surrounding sentence to identify contexts
        where these verbs are appropriate technical terminology rather than user-focus violations.
        
        Examples of appropriate usage:
        - "enable C-states in the firmware"
        - "disable the driver in the kernel"
        - "execute the API call"
        - "enable the feature flag"
        
        Args:
            word: The word being evaluated ("enable" or "execute")
            text: The full text being analyzed
            context: Context dictionary with metadata about the document
            
        Returns:
            True if this is a technical domain usage (suppress the violation)
            False if this should be flagged for user-focus improvement
        """
        word_lower = word.lower()
        
        # === TIER 1: Comprehensive Technical Keywords ===
        # These keywords indicate firmware, system, driver, API, or configuration contexts
        # where "enable"/"disable"/"execute" are the standard technical verbs.
        technical_keywords = {
            # Hardware & Firmware
            'firmware', 'efi', 'bios', 'uefi', 'bootloader', 'rom', 'nvram',
            
            # System & Kernel
            'kernel', 'driver', 'module', 'daemon', 'service', 'process', 'thread',
            
            # Hardware States & Features
            'c-state', 'p-state', 'd-state', 's-state', 'acpi', 'power state',
            'cpu', 'processor', 'core', 'thread', 'interrupt', 'dma',
            
            # Configuration & Settings
            'setting', 'option', 'parameter', 'configuration', 'config',
            'flag', 'switch', 'toggle', 'property', 'attribute',
            
            # Features & Capabilities
            'feature', 'capability', 'function', 'functionality', 'mode',
            'extension', 'plugin', 'add-on', 'component',
            
            # APIs & Programming
            'api', 'interface', 'endpoint', 'method', 'function', 'call',
            'protocol', 'service', 'routine', 'procedure',
            
            # Network & Communication
            'port', 'socket', 'connection', 'channel', 'stream',
            'protocol', 'network', 'interface',
            
            # Security & Access Control
            'permission', 'privilege', 'access control', 'policy',
            'authentication', 'authorization', 'encryption',
            
            # Virtualization & Containers
            'virtual machine', 'vm', 'container', 'hypervisor', 'guest',
            'namespace', 'cgroup',
            
            # Storage & Filesystems
            'partition', 'volume', 'mount', 'filesystem', 'block device',
            
            # System Management
            'registry', 'systemd', 'sysctl', 'proc', 'sys',
        }
        
        # === TIER 2: Technical Action Verbs ===
        # These verbs often appear alongside "enable"/"disable" in technical contexts
        technical_action_verbs = {
            'configure', 'set', 'modify', 'change', 'adjust', 'tune',
            'initialize', 'load', 'unload', 'start', 'stop', 'restart',
            'activate', 'deactivate', 'invoke', 'trigger', 'call'
        }
        
        # === TIER 3: Document Type Indicators ===
        # Check if document metadata indicates technical reference material
        doc_type = context.get('doc_type', '').lower()
        if doc_type in ['reference', 'man_page', 'api_docs', 'system_admin', 'configuration']:
            return True
        
        # Check content type
        content_type = context.get('content_type', '').lower()
        if content_type in ['concept', 'reference', 'technical', 'procedure', 'procedural']:
            # For CONCEPT documents (like the example), check for technical keywords
            # For others, this is likely technical documentation
            text_lower = text.lower()
            
            # Count technical keyword density
            keyword_hits = sum(1 for keyword in technical_keywords if keyword in text_lower)
            
            # If we have multiple technical keywords in the document, this is technical content
            if keyword_hits >= 2:
                return True
        
        # === TIER 4: Sentence-Level Analysis ===
        # Analyze the immediate sentence context around the word
        # This is the most precise check - look for technical keywords in the same sentence
        
        # Find sentences containing the word (case-insensitive search)
        import re
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if word_lower in sentence.lower():
                sentence_lower = sentence.lower()
                
                # Check for technical keywords in this sentence
                if any(keyword in sentence_lower for keyword in technical_keywords):
                    return True
                
                # Check for technical action verbs near "enable"/"disable"
                if any(verb in sentence_lower for verb in technical_action_verbs):
                    return True
                
                # Check for technical patterns (e.g., "enable/disable the...")
                # which is common in technical documentation
                if re.search(r'\b(enable|disable)\s+(?:the\s+)?(?:individual\s+)?[a-z\-]+\b', sentence_lower):
                    # Check if what follows is a technical term
                    following_words = re.findall(r'\b(enable|disable)\s+(?:the\s+)?(?:individual\s+)?([a-z\-]+)\b', sentence_lower)
                    for _, following_word in following_words:
                        if following_word in technical_keywords or '-' in following_word:
                            return True
        
        # === TIER 5: Code/Command Context ===
        # Check if the word appears in a code block or command-line context
        block_type = context.get('block_type', '')
        if block_type in ['code', 'command', 'terminal', 'shell', 'listing', 'literal']:
            return True
        
        return False  # Not a technical domain context - normal user-focus check applies

    def _is_target_abbreviation_e_words(self, token, pattern: str) -> bool:
        """
        Check if a token flagged by entity recognition is actually our target abbreviation.
        """
        # For "e.g." - if it's tagged as ORG but is clearly the abbreviation
        if pattern == 'e.g.':
            if token.text.lower() in ['e.g.', 'eg', 'e.g']:
                return True
        
        # For "etc" - this should be flagged if it's the Latin abbreviation
        elif pattern == 'etc':
            if token.text.lower() in ['etc.', 'etc', 'et cetera']:
                return True
        
        return False

    def _apply_feedback_clues_e_words(self, ev: float, word: str, context: Dict[str, Any]) -> float:
        """Apply feedback pattern clues for E-words."""
        patterns = {'often_flagged_terms': {'easy', 'e-mail', 'enable', 'effortless'}, 'accepted_terms': set()}
        word_lower = word.lower()
        
        if word_lower in patterns.get('often_flagged_terms', set()):
            ev += 0.1
        elif word_lower in patterns.get('accepted_terms', set()):
            ev -= 0.2
        
        return ev
    
    def _track_enter_type_usage(self, doc, text: str) -> None:
        """
        Track usage of 'enter' and 'type' for consistency checking.
        Per IBM Style Guide: both are acceptable, but use one consistently.
        """
        for token in doc:
            token_lemma = token.lemma_.lower()
            
            # Track "enter" usage (as a verb, not noun)
            if token_lemma == 'enter' and token.pos_ == 'VERB':
                self.enter_type_usage['enter'].append({
                    'sentence': token.sent.text,
                    'position': token.idx
                })
            
            # Track "type" usage (as a verb, not noun)
            elif token_lemma == 'type' and token.pos_ == 'VERB':
                self.enter_type_usage['type'].append({
                    'sentence': token.sent.text,
                    'position': token.idx
                })
    
    def _check_enter_type_consistency(self, errors: List[Dict[str, Any]], 
                                      doc, text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check if document uses both 'enter' and 'type' inconsistently.
        Only flag if BOTH terms are used in the same document.
        
        Per IBM Style Guide: "Use 'enter' or 'type'... use one term consistently"
        """
        enter_count = len(self.enter_type_usage['enter'])
        type_count = len(self.enter_type_usage['type'])
        
        # Only flag if BOTH are used (inconsistency)
        if enter_count > 0 and type_count > 0:
            # Flag the less-used term (or flag both if equal)
            if enter_count <= type_count:
                # Flag 'enter' usage and suggest 'type' (since 'type' is more common in this doc)
                for usage in self.enter_type_usage['enter']:
                    errors.append(self._create_error(
                        sentence=usage['sentence'],
                        sentence_index=0,  # Would need proper tracking
                        message=f"Inconsistent terminology: Document uses both 'enter' and 'type'. "
                                f"Use one term consistently ('{type_count}' uses of 'type' vs '{enter_count}' of 'enter').",
                        suggestions=[
                            f"Replace 'enter' with 'type' for consistency (document uses 'type' more frequently)",
                            "Or replace all uses of 'type' with 'enter' throughout the document"
                        ],
                        severity='low',
                        text=text,
                        context=context,
                        evidence_score=0.6,  # Medium evidence - consistency issue
                        span=(usage['position'], usage['position'] + 5),  # Approximate
                        flagged_text='enter'
                    ))
            
            if type_count < enter_count:
                # Flag 'type' usage and suggest 'enter' (since 'enter' is more common in this doc)
                for usage in self.enter_type_usage['type']:
                    errors.append(self._create_error(
                        sentence=usage['sentence'],
                        sentence_index=0,
                        message=f"Inconsistent terminology: Document uses both 'enter' and 'type'. "
                                f"Use one term consistently ('{enter_count}' uses of 'enter' vs '{type_count}' of 'type').",
                        suggestions=[
                            f"Replace 'type' with 'enter' for consistency (document uses 'enter' more frequently)",
                            "Or replace all uses of 'enter' with 'type' throughout the document"
                        ],
                        severity='low',
                        text=text,
                        context=context,
                        evidence_score=0.6,
                        span=(usage['position'], usage['position'] + 4),  # Approximate
                        flagged_text='type'
                    ))
        
        return errors