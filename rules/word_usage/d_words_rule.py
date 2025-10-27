"""
Word Usage Rule for words starting with 'D'.
Enhanced with spaCy PhraseMatcher for efficient pattern detection.
"""
from typing import List, Dict, Any
from .base_word_usage_rule import BaseWordUsageRule
import re

try:
    from spacy.tokens import Doc
except ImportError:
    Doc = None

class DWordsRule(BaseWordUsageRule):
    """
    Checks for the incorrect usage of specific words starting with 'D'.
    Enhanced with spaCy PhraseMatcher for efficient detection.
    """
    def _get_rule_type(self) -> str:
        return 'word_usage_d'

    def analyze(self, text: str, sentences: List[str], nlp=None, context=None) -> List[Dict[str, Any]]:
        """
        Evidence-based analysis for D-word usage violations.
        Computes a nuanced evidence score per occurrence considering linguistic,
        structural, semantic, and feedback clues.
        """
        # === UNIVERSAL CODE CONTEXT GUARD ===
        # Skip analysis for code blocks, listings, and literal blocks (technical syntax, not prose)
        if context and context.get('block_type') in ['listing', 'literal', 'code_block', 'inline_code']:
            return []
        errors: List[Dict[str, Any]] = []
        if not nlp:
            return errors
            
        doc = nlp(text)
        
        # Define D-word patterns with evidence categories
        d_word_patterns = {
            "data base": {"alternatives": ["database"], "category": "spacing", "severity": "high"},
            "data center": {"alternatives": ["datacenter"], "category": "context_specific", "severity": "low"},
            "data set": {"alternatives": ["dataset"], "category": "spacing", "severity": "medium"},
            "deactivate": {"alternatives": ["deactivate"], "category": "correct_form", "severity": "low"},
            "deallocate": {"alternatives": ["deallocate"], "category": "correct_form", "severity": "low"},
            "deinstall": {"alternatives": ["uninstall"], "category": "word_choice", "severity": "high"},
            "desire": {"alternatives": ["want", "need"], "category": "user_focus", "severity": "medium"},
            "dialogue": {"alternatives": ["dialog"], "category": "spelling", "severity": "low"},
            "disable": {"alternatives": ["turn off", "clear"], "category": "user_clarity", "severity": "medium"},
            "double click": {"alternatives": ["double-click"], "category": "hyphenation", "severity": "low"},
        }

        # Evidence-based analysis for D-words
        for word, details in d_word_patterns.items():
            for match in re.finditer(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE):
                char_start = match.start()
                char_end = match.end()
                matched_text = match.group(0)
                
                # Find the token and sentence
                token = None
                sent = None
                sentence_index = 0
                
                for i, s in enumerate(doc.sents):
                    if s.start_char <= char_start < s.end_char:
                        sent = s
                        sentence_index = i
                        for t in s:
                            if t.idx <= char_start < t.idx + len(t.text):
                                token = t
                                break
                        break
                
                if sent and token:
                    # Apply surgical guards
                    if self._apply_surgical_zero_false_positive_guards_word_usage(token, context or {}):
                        continue
                    
                    evidence_score = self._calculate_d_word_evidence(
                        word, token, sent, text, context or {}, details["category"]
                    )
                    
                    if evidence_score > 0.1:
                        errors.append(self._create_error(
                            sentence=sent.text,
                            sentence_index=sentence_index,
                            message=self._generate_evidence_aware_word_usage_message(word, evidence_score, details["category"]),
                            suggestions=self._generate_evidence_aware_word_usage_suggestions(word, details["alternatives"], evidence_score, context or {}, details["category"]),
                            severity=details["severity"] if evidence_score < 0.7 else 'high',
                            text=text,
                            context=context,
                            evidence_score=evidence_score,
                            span=(char_start, char_end),
                            flagged_text=matched_text
                        ))
        
        return errors

    def _calculate_d_word_evidence(self, word: str, token, sentence, text: str, context: Dict[str, Any], category: str) -> float:
        """Calculate evidence score for D-word usage violations."""
        evidence_score = self._get_base_d_word_evidence_score(word, category, sentence, context)
        
        if evidence_score == 0.0:
            return 0.0
        
        evidence_score = self._apply_linguistic_clues_d_words(evidence_score, word, token, sentence)
        evidence_score = self._apply_structural_clues_d_words(evidence_score, context)
        evidence_score = self._apply_semantic_clues_d_words(evidence_score, word, text, context)
        evidence_score = self._apply_feedback_clues_d_words(evidence_score, word, context)
        
        return max(0.0, min(1.0, evidence_score))
    
    def _get_base_d_word_evidence_score(self, word: str, category: str, sentence, context: Dict[str, Any]) -> float:
        """Set dynamic base evidence score based on D-word category."""
        if category == 'spacing':
            return 0.9  # "data base" vs "database", "data set" vs "dataset"
        elif category == 'word_choice':
            return 0.85  # "deinstall" vs "uninstall"
        elif category in ['user_focus', 'user_clarity']:
            return 0.75  # "desire", "disable" - user-centered language
        elif category == 'hyphenation':
            return 0.6  # "double click" vs "double-click"
        elif category in ['spelling', 'context_specific', 'correct_form']:
            return 0.5  # "dialogue"/"dialog", "data center" context
        return 0.6

    def _apply_linguistic_clues_d_words(self, ev: float, word: str, token, sentence) -> float:
        """Apply D-word-specific linguistic clues."""
        sent_text = sentence.text.lower()
        word_lower = word.lower()
        
        # User action context
        action_indicators = ['click', 'select', 'choose', 'enable', 'turn']
        if any(action in sent_text for action in action_indicators):
            if word_lower in ['disable', 'double click']:
                ev += 0.1  # User action clarity important
        
        # Data/database context
        if word_lower in ['data base', 'data set']:
            data_indicators = ['store', 'query', 'table', 'record']
            if any(data in sent_text for data in data_indicators):
                ev += 0.15  # Database terminology needs precision
        
        return ev

    def _apply_structural_clues_d_words(self, ev: float, context: Dict[str, Any]) -> float:
        """Apply structural context clues for D-words."""
        block_type = context.get('block_type', 'paragraph')
        
        if block_type in ['step', 'procedure']:
            ev += 0.1  # Procedural content needs precision
        elif block_type == 'heading':
            ev -= 0.1  # Headings more flexible
        
        return ev

    def _apply_semantic_clues_d_words(self, ev: float, word: str, text: str, context: Dict[str, Any]) -> float:
        """
        Apply semantic and content-type clues for D-words.
        
        WORLD-CLASS ENHANCEMENT: Domain-aware guard for technical terminology.
        This method now includes sophisticated detection of technical contexts where
        'disable' is the correct, precise terminology rather than a user-focus violation.
        """
        content_type = context.get('content_type', 'general')
        word_lower = word.lower()
        
        # === DOMAIN-AWARE GUARD for "disable" ===
        # In technical domains (firmware, drivers, APIs, system states), "disable" 
        # is the correct, precise terminology. This guard detects those contexts and prevents false positives.
        if word_lower == 'disable':
            # Check if this word appears in a technical/system context
            if self._is_technical_domain_context_d_words(word, text, context):
                # Strong suppression - this is correct technical terminology
                ev -= 0.95
                return max(0.0, ev)  # Return early if we've determined this is technical usage
        
        if content_type == 'tutorial':
            if word_lower in ['disable', 'desire', 'double click']:
                ev += 0.15  # Tutorials need user-focused language
        elif content_type == 'technical':
            if word_lower in ['data base', 'deinstall']:
                ev += 0.1  # Technical docs need standard terminology
        
        return ev

    def _is_technical_domain_context_d_words(self, word: str, text: str, context: Dict[str, Any]) -> bool:
        """
        WORLD-CLASS GUARD: Detect if "disable" appears in a technical domain context
        where it is the correct, precise terminology.
        
        This guard checks for technical keywords in the surrounding sentence to identify contexts
        where "disable" is appropriate technical terminology rather than a user-focus violation.
        
        Examples of appropriate usage:
        - "disable C-states in the firmware"
        - "disable the driver in the kernel"
        - "disable the API endpoint"
        - "disable the feature flag"
        
        Args:
            word: The word being evaluated ("disable")
            text: The full text being analyzed
            context: Context dictionary with metadata about the document
            
        Returns:
            True if this is a technical domain usage (suppress the violation)
            False if this should be flagged for user-focus improvement
        """
        word_lower = word.lower()
        
        # === TIER 1: Comprehensive Technical Keywords ===
        # These keywords indicate firmware, system, driver, API, or configuration contexts
        # where "disable" is the standard technical verb.
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
        # These verbs often appear alongside "disable" in technical contexts
        technical_action_verbs = {
            'configure', 'set', 'modify', 'change', 'adjust', 'tune',
            'initialize', 'load', 'unload', 'start', 'stop', 'restart',
            'activate', 'deactivate', 'enable', 'invoke', 'trigger', 'call'
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
                
                # Check for technical action verbs near "disable"
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

    def _apply_feedback_clues_d_words(self, ev: float, word: str, context: Dict[str, Any]) -> float:
        """Apply feedback pattern clues for D-words."""
        patterns = {'often_flagged_terms': {'data base', 'deinstall', 'desire'}, 'accepted_terms': set()}
        word_lower = word.lower()
        
        if word_lower in patterns.get('often_flagged_terms', set()):
            ev += 0.1
        elif word_lower in patterns.get('accepted_terms', set()):
            ev -= 0.2
        
        return ev