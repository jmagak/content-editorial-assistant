"""
Periods Rule - Evidence-Based Analysis
Based on IBM Style Guide topic: "Periods"

**UPDATED** with evidence-based scoring for nuanced period usage analysis.
"""
from typing import List, Dict, Any, Optional
from .base_punctuation_rule import BasePunctuationRule
from .services.punctuation_config_service import get_punctuation_config

try:
    from spacy.tokens import Doc, Token, Span
except ImportError:
    Doc = None
    Token = None
    Span = None

class PeriodsRule(BasePunctuationRule):
    """
    Checks for incorrect use of periods using evidence-based analysis:
    - Periods within uppercase abbreviations
    - Other period usage violations
    Enhanced with dependency parsing and contextual awareness.
    """
    def __init__(self):
        """Initialize the rule with configuration service."""
        super().__init__()
        self.config = get_punctuation_config()
    
    def _get_rule_type(self) -> str:
        """Returns the unique identifier for this rule."""
        return 'periods'

    def analyze(self, text: str, sentences: List[str], nlp=None, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Enhanced evidence-based analysis for period usage violations:
        - Periods within uppercase abbreviations
        - Duplicate periods (e.g., ".." or "...")
        - Extra periods in lists, headings
        - Missing periods at sentence endings
        - Context-aware period rules for different content types
        """
        # === UNIVERSAL CODE CONTEXT GUARD ===
        # Skip analysis for code blocks, listings, and literal blocks (technical syntax, not prose)
        if context and context.get('block_type') in ['listing', 'literal', 'code_block', 'inline_code']:
            return []
        errors: List[Dict[str, Any]] = []
        context = context or {}
        
        # Fallback analysis when nlp is not available
        if not nlp:
            return self._fallback_enhanced_periods_analysis(text, sentences, context)

        try:
            doc = nlp(text)
            
            # Analyze periods in abbreviations (existing functionality)
            for i, sent in enumerate(doc.sents):
                for token in sent:
                    # Look for tokens that contain periods in abbreviations (e.g., "U.S.A.")
                    if self._is_abbreviation_with_periods(token):
                        evidence_score = self._calculate_period_evidence(token, sent, text, context)
                        
                        if evidence_score > 0.1:
                            errors.append(self._create_error(
                                sentence=sent.text,
                                sentence_index=i,
                                message=self._get_contextual_period_message('abbreviation_periods', evidence_score, context),
                                suggestions=self._generate_smart_period_suggestions(token, evidence_score, context),
                                severity='low' if evidence_score < 0.7 else 'medium',
                                text=text,
                                context=context,
                                evidence_score=evidence_score,
                                span=(token.idx, token.idx + len(token.text)),
                                flagged_text=token.text,
                                violation_type='abbreviation_periods'
                            ))
            
            # NEW: Analyze duplicate periods
            errors.extend(self._analyze_duplicate_periods(doc, text, context))
            
            # NEW: Analyze unnecessary periods in headings and lists
            errors.extend(self._analyze_unnecessary_periods(doc, text, context, nlp))
            
            # NEW: Analyze missing periods at sentence endings
            errors.extend(self._analyze_missing_periods(doc, text, context))
            
        except Exception as e:
            # Graceful degradation for SpaCy errors
            return self._fallback_enhanced_periods_analysis(text, sentences, context)
        
        return errors

    # === EVIDENCE CALCULATION ===

    def _is_abbreviation_with_periods(self, token: 'Token') -> bool:
        """
        Check if this token is an abbreviation containing periods.
        SpaCy tokenizes "U.S.A." as a single token, so we check the token text.
        """
        import re
        # Pattern for abbreviations with periods (e.g., U.S.A., A.P.I., E.U.)
        return bool(re.match(r'^[A-Z]\.(?:[A-Z]\.)*[A-Z]?\.?$', token.text))

    def _calculate_period_evidence(self, abbrev_token: 'Token', sent: 'Span', text: str, context: Dict[str, Any]) -> float:
        """
        Calculate evidence (0.0-1.0) that a period usage is incorrect.
        
        Higher scores indicate stronger evidence of an error.
        Lower scores indicate acceptable usage or ambiguous cases.
        """
        # === SURGICAL ZERO FALSE POSITIVE GUARDS ===
        # Apply surgical guards FIRST to eliminate false positives
        if self._apply_zero_false_positive_guards_punctuation(abbrev_token, context):
            return 0.0
        
        # Creative content commonly uses various punctuation styles
        content_type = context.get('content_type', 'general')
        if content_type in ['creative', 'literary', 'narrative']:
            return 0.0
        
        # Legal documents often require periods in abbreviations
        if content_type == 'legal':
            return 0.0
        
        # Quotes preserve original punctuation
        block_type = context.get('block_type', 'paragraph')
        if block_type in ['quote', 'blockquote']:
            return 0.0
        
        # Citations and academic references have specific formatting
        if block_type in ['citation', 'reference', 'footnote', 'bibliography']:
            return 0.0
        
        # Check for legitimate abbreviations that should keep periods
        if self._is_legitimate_abbreviation_period(abbrev_token, sent, context):
            return 0.0
        
        # === STEP 1: BASE EVIDENCE ASSESSMENT ===
        # Start with strong evidence for abbreviation periods
        evidence_score = 0.8
        
        # === STEP 2: LINGUISTIC CLUES ===
        evidence_score = self._apply_common_linguistic_clues_punctuation(evidence_score, abbrev_token, sent)
        
        # === STEP 3: STRUCTURAL CLUES ===
        evidence_score = self._apply_common_structural_clues_punctuation(evidence_score, abbrev_token, context)
        
        # === STEP 4: SEMANTIC CLUES ===
        evidence_score = self._apply_common_semantic_clues_punctuation(evidence_score, abbrev_token, context)
        
        return max(0.0, min(1.0, evidence_score))

    def _is_legitimate_abbreviation_period(self, abbrev_token: 'Token', sent: 'Span', context: Dict[str, Any]) -> bool:
        """
        Check if this abbreviation period usage is legitimate in this context.
        """
        # Check for legitimate abbreviations that should keep periods (from YAML configuration)
        if self.config.is_legitimate_abbreviation(abbrev_token.text)[0]:
            return True
        
        return False

    # === ENHANCED PERIODS ANALYSIS METHODS ===

    def _fallback_enhanced_periods_analysis(self, text: str, sentences: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced fallback analysis when spaCy is not available."""
        errors = []
        
        # Apply basic guards for fallback analysis
        content_type = context.get('content_type', 'general')
        block_type = context.get('block_type', 'paragraph')
        
        # Skip if in contexts where periods might be used differently
        if content_type in ['creative', 'literary', 'narrative', 'legal']:
            return errors  # No errors for creative or legal content
        if block_type in ['quote', 'blockquote', 'code_block', 'literal_block', 'citation']:
            return errors  # No errors for quotes, code, and citations
        
        import re
        
        # 1. Existing: Abbreviations with periods (e.g., U.S.A., A.P.I., E.U.)
        for i, sentence in enumerate(sentences):
            for match in re.finditer(r'\b[A-Z]\.(?:[A-Z]\.)*[A-Z]?\.?\b', sentence):
                match_text = match.group(0)
                
                # Check for legitimate abbreviations that should keep periods
                legitimate_patterns = ['P.M.', 'A.M.', 'P.O.', 'U.K.', 'U.S.', 'P.M', 'A.M', 'P.O', 'U.K', 'U.S']
                if match_text.upper() in legitimate_patterns:
                    continue  # Skip legitimate time/location abbreviations
                
                errors.append(self._create_error(
                    sentence=sentence,
                    sentence_index=i,
                    message="Consider removing periods from this abbreviation for modern style.",
                    suggestions=["Remove periods from abbreviations (e.g., 'USA' instead of 'U.S.A.')", "Modern style guides prefer clean abbreviations without periods."],
                    severity='low',
                    text=text,
                    context=context,
                    evidence_score=0.7,  # Default evidence for fallback analysis
                    span=(match.start(), match.end()),
                    flagged_text=match_text,
                    violation_type='abbreviation_periods'
                ))
        
        # 2. NEW: Duplicate periods (exclude legitimate ellipses)
        for match in re.finditer(r'\.{2,}(?!\.)', text):
            if len(match.group()) == 2:  # Double periods are usually errors
                evidence_score = 0.8
            else:
                continue  # Skip potential ellipses (3+ periods)
            
            errors.append(self._create_error(
                sentence=self._get_sentence_for_position(match.start(), text),
                sentence_index=0,
                message="Double periods detected - likely a typo.",
                suggestions=["Remove extra period.", "Use single period for sentence ending.", "Use ellipsis (...) if trailing off is intended."],
                severity='medium',
                text=text,
                context=context,
                evidence_score=evidence_score,
                span=(match.start(), match.end()),
                flagged_text=match.group(),
                violation_type='duplicate_periods'
            ))
        
        return errors

    def _analyze_duplicate_periods(self, doc: 'Doc', text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze text for duplicate periods using spaCy."""
        errors = []
        
        import re
        # Pattern for duplicate periods (excluding legitimate ellipses)
        for match in re.finditer(r'\.{2}(?!\.)', text):  # Exactly two periods (not ellipses)
            evidence_score = self._calculate_duplicate_periods_evidence(match, text, context)
            
            if evidence_score > 0.1:
                errors.append(self._create_error(
                    sentence=self._get_sentence_for_position(match.start(), text),
                    sentence_index=self._get_sentence_index_for_position(match.start(), doc),
                    message=self._get_contextual_period_message('duplicate_periods', evidence_score, context),
                    suggestions=self._generate_duplicate_periods_suggestions(evidence_score, context),
                    severity='medium',
                    text=text,
                    context=context,
                    evidence_score=evidence_score,
                    span=(match.start(), match.end()),
                    flagged_text=match.group(),
                    violation_type='duplicate_periods'
                ))
        
        return errors

    def _analyze_unnecessary_periods(self, doc: 'Doc', text: str, context: Dict[str, Any], nlp) -> List[Dict[str, Any]]:
        """Analyze text for unnecessary periods in headings and lists."""
        errors = []
        
        block_type = context.get('block_type', 'paragraph')
        
        # Check headings for unnecessary periods
        if block_type == 'heading':
            if text.strip().endswith('.'):
                evidence_score = self._calculate_unnecessary_period_evidence('heading', text, context, nlp)
                
                if evidence_score > 0.1:
                    errors.append(self._create_error(
                        sentence=text.strip(),
                        sentence_index=0,
                        message=self._get_contextual_period_message('unnecessary_period_in_heading', evidence_score, context),
                        suggestions=self._generate_unnecessary_period_suggestions('heading', evidence_score, context),
                        severity='low',
                        text=text,
                        context=context,
                        evidence_score=evidence_score,
                        span=(len(text.rstrip()) - 1, len(text.rstrip())),
                        flagged_text='.',
                        violation_type='unnecessary_period_in_heading'
                    ))
        
        # Check list items for unnecessary periods
        elif block_type in ['ordered_list_item', 'unordered_list_item']:
            stripped_text = text.strip()
            if stripped_text.endswith('.'):
                evidence_score = self._calculate_unnecessary_period_evidence('list_item', text, context, nlp)
                
                if evidence_score > 0.1:
                    errors.append(self._create_error(
                        sentence=stripped_text,
                        sentence_index=0,
                        message=self._get_contextual_period_message('unnecessary_period_in_list', evidence_score, context),
                        suggestions=self._generate_unnecessary_period_suggestions('list_item', evidence_score, context),
                        severity='low',
                        text=text,
                        context=context,
                        evidence_score=evidence_score,
                        span=(len(text.rstrip()) - 1, len(text.rstrip())),
                        flagged_text='.',
                        violation_type='unnecessary_period_in_list'
                    ))
        
        return errors

    def _analyze_missing_periods(self, doc: 'Doc', text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze text for missing periods at sentence endings."""
        errors = []
        
        # Skip contexts where missing periods are acceptable
        block_type = context.get('block_type', 'paragraph')
        if block_type in ['heading', 'ordered_list_item', 'unordered_list_item', 'table_cell']:
            return errors
        
        # Check each sentence for missing end punctuation
        for i, sent in enumerate(doc.sents):
            sentence_text = sent.text.strip()
            
            # Skip very short sentences and questions/exclamations
            if len(sentence_text) < 10 or sentence_text.endswith(('?', '!')):
                continue
            
            # Check if sentence ends without proper punctuation
            if not sentence_text.endswith(('.', ':', ';')):
                evidence_score = self._calculate_missing_period_evidence(sent, text, context)
                
                if evidence_score > 0.1:
                    errors.append(self._create_error(
                        sentence=sentence_text,
                        sentence_index=i,
                        message=self._get_contextual_period_message('missing_period', evidence_score, context),
                        suggestions=self._generate_missing_period_suggestions(evidence_score, context),
                        severity='medium' if evidence_score > 0.7 else 'low',
                        text=text,
                        context=context,
                        evidence_score=evidence_score,
                        span=(sent[-1].idx + len(sent[-1].text), sent[-1].idx + len(sent[-1].text)),
                        flagged_text=sent[-1].text,
                        violation_type='missing_period'
                    ))
        
        return errors

    # === ENHANCED EVIDENCE CALCULATION ===
    
    def _is_complete_sentence(self, text: str, nlp) -> bool:
        """
        Checks if a given string is a grammatically complete sentence.
        A complete sentence must have a subject and a main verb (predicate).
        Handles passive voice and complex structures.
        """
        if not text:
            return False
        
        doc = nlp(text.strip())
        
        # Ensure it's parsed as a single sentence
        if len(list(doc.sents)) > 1:
            return True  # If spaCy thinks it's multiple sentences, it's definitely complete enough.
        
        has_subject = False
        has_root_verb = False
        
        for token in doc:
            if "subj" in token.dep_:
                has_subject = True
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                has_root_verb = True
        
        # Handle passive voice where subject might be `nsubjpass`
        if not has_subject:
            has_subject = any(t.dep_ == 'nsubjpass' for t in doc)
        
        return has_subject and has_root_verb
    
    def _calculate_duplicate_periods_evidence(self, match, text: str, context: Dict[str, Any]) -> float:
        """Calculate evidence for duplicate period violations."""
        # === SURGICAL ZERO FALSE POSITIVE GUARDS ===
        if context.get('block_type') in ['code_block', 'inline_code', 'literal_block']:
            return 0.0
        
        if context.get('content_type') in ['creative', 'literary']:
            return 0.0  # Creative writing may use unusual punctuation
        
        # === STEP 1: BASE EVIDENCE ASSESSMENT ===
        evidence_score = 0.9  # Very strong evidence for duplicate periods
        
        # === STEP 2: CONTEXT CLUES ===
        # Double periods are almost always errors in standard text
        return max(0.0, min(1.0, evidence_score))

    def _calculate_unnecessary_period_evidence(self, location_type: str, text: str, context: Dict[str, Any], nlp) -> float:
        """Calculate evidence for unnecessary period violations."""
        # === SURGICAL ZERO FALSE POSITIVE GUARDS ===
        if context.get('block_type') in ['code_block', 'inline_code', 'literal_block']:
            return 0.0
        
        if context.get('content_type') in ['creative', 'literary']:
            return 0.0
        
        # === STEP 1: BASE EVIDENCE ASSESSMENT ===
        if location_type == 'heading':
            evidence_score = 0.7  # Good evidence - headings usually don't end with periods
        elif location_type == 'list_item':
            # === WORLD-CLASS GUARD: Check if the list item is a complete sentence ===
            if self._is_complete_sentence(text, nlp):
                return 0.0  # It's a full sentence, so the period is correct. Suppress error.
            
            # If it's a fragment, proceed with evidence calculation.
            word_count = len(text.strip().split())
            if word_count <= 3:
                evidence_score = 0.8  # Strong evidence for short items
            elif word_count <= 6:
                evidence_score = 0.6  # Medium evidence for medium items
            else:
                evidence_score = 0.3  # Weak evidence for long items (might be sentences)
        else:
            evidence_score = 0.5
        
        # === STEP 2: CONTEXT CLUES ===
        content_type = context.get('content_type', 'general')
        if content_type == 'formal':
            evidence_score -= 0.1  # Formal documents might be more flexible
        elif content_type == 'technical':
            evidence_score += 0.1  # Technical docs prefer clean formatting
        
        return max(0.0, min(1.0, evidence_score))

    def _calculate_missing_period_evidence(self, sent: 'Span', text: str, context: Dict[str, Any]) -> float:
        """Calculate evidence for missing period violations."""
        # === SURGICAL ZERO FALSE POSITIVE GUARDS ===
        if context.get('block_type') in ['code_block', 'inline_code', 'literal_block']:
            return 0.0
        
        # Check if sentence ends with other acceptable punctuation
        sentence_text = sent.text.strip()
        if sentence_text.endswith((':', ';', '-', ')', ']')):
            return 0.0
        
        # ZERO FALSE POSITIVE GUARD 7: URLs, Commands, and Technical Content
        if self._is_technical_content_not_prose(sent, sentence_text, context):
            return 0.0  # Technical content doesn't require prose punctuation
        
        # CRITICAL GUARD: Check if this is a list item fragment
        # List fragments (non-sentences) should not require periods
        if self._is_list_item_fragment(sent, context):
            return 0.0  # Fragments don't need periods
        
        # === STEP 1: BASE EVIDENCE ASSESSMENT ===
        evidence_score = 0.6  # Medium evidence for missing periods
        
        # === STEP 2: SENTENCE ANALYSIS ===
        # Complete sentences with verbs are more likely to need periods
        has_verb = any(token.pos_ == 'VERB' for token in sent)
        if has_verb:
            evidence_score += 0.2
        
        # Longer sentences are more likely to need periods
        word_count = len([token for token in sent if token.is_alpha])
        if word_count > 8:
            evidence_score += 0.1
        
        # === STEP 3: CONTEXT CLUES ===
        content_type = context.get('content_type', 'general')
        if content_type == 'formal':
            evidence_score += 0.1  # Formal writing needs proper punctuation
        
        return max(0.0, min(1.0, evidence_score))

    # === HELPER METHODS ===
    
    def _is_technical_content_not_prose(self, sent: 'Span', sentence_text: str, context: Dict[str, Any]) -> bool:
        """
        Detect if this is technical content (URL, command, path, code) that should not be
        treated as prose requiring a period.
        
        Technical content includes:
        - URLs (http://, https://, ftp://, etc.)
        - Shell commands ($ ..., # ..., > ...)
        - File paths (/path/to/file, C:\\path, ./relative, ~/home)
        - Code wrapped in backticks
        - Technical identifiers with special syntax
        
        Returns True if this is pure technical content, False if it's prose.
        """
        import re
        
        # Strip leading/trailing whitespace for analysis
        text_stripped = sentence_text.strip()
        text_lower = text_stripped.lower()
        
        # ─────────────────────────────────────────────────────────────────────
        # GUARD 1: URL Protocols
        # ─────────────────────────────────────────────────────────────────────
        url_protocols = [
            'http://', 'https://', 'ftp://', 'ftps://', 'ssh://', 'git://',
            'file://', 'mailto:', 'tel:', 'ws://', 'wss://'
        ]
        
        # Check if sentence STARTS with a URL (standalone URL line)
        if any(text_lower.startswith(protocol) for protocol in url_protocols):
            return True
        
        # Check if entire sentence is a URL (with potential wrapping chars)
        # Pattern: optional punctuation + protocol + rest of URL
        url_pattern = r'^[<\(\[\{]?((?:https?|ftp|git|ssh)://[^\s\)\]\}>]+)[>\)\]\}]?$'
        if re.match(url_pattern, text_stripped, re.IGNORECASE):
            return True
        
        # ─────────────────────────────────────────────────────────────────────
        # GUARD 2: Shell Command Prompts
        # ─────────────────────────────────────────────────────────────────────
        command_prompts = ['$ ', '# ', '> ', '% ', '>>> ', '~$ ', 'bash$ ', 'sh$ ']
        if any(text_stripped.startswith(prompt) for prompt in command_prompts):
            return True
        
        # ─────────────────────────────────────────────────────────────────────
        # GUARD 3: File Paths
        # ─────────────────────────────────────────────────────────────────────
        # Absolute paths (Unix/Linux)
        if text_stripped.startswith(('/home/', '/usr/', '/var/', '/etc/', '/opt/', '/root/', '/')):
            return True
        
        # Relative paths
        if text_stripped.startswith(('./', '../', '~/', '.\\', '..\\', '~\\')):
            return True
        
        # Windows paths (drive letter)
        if re.match(r'^[A-Z]:\\', text_stripped, re.IGNORECASE):
            return True
        
        path_separator_count = text_stripped.count('/') + text_stripped.count('\\')
        is_likely_url = '://' in text_stripped  # URLs have ://
        
        prose_starters = ('the ', 'a ', 'an ', 'this ', 'that ', 'these ', 'those ', 'in ', 'on ', 'at ', 'is ', 'are ')
        starts_with_prose = text_lower.startswith(prose_starters)
        
        # Check for file-like patterns (has file extension)
        has_file_extension = bool(re.match(r'.*\.\w{1,10}$', text_stripped))
        
        # Only trigger if it's a standalone path:
        # - Has multiple separators
        # - Doesn't start with prose words
        # - Either starts with path char OR has file extension OR high slash density
        is_standalone_path = (not is_likely_url and 
                             path_separator_count >= 2 and 
                             len(sent) < 15 and
                             not starts_with_prose and
                             (text_stripped.startswith(('/', '.', '~')) or 
                              has_file_extension or 
                              path_separator_count / len(text_stripped) > 0.15))
        
        if is_standalone_path:
            return True
        
        # ─────────────────────────────────────────────────────────────────────
        # GUARD 4: Code Syntax Markers
        # ─────────────────────────────────────────────────────────────────────
        # Surrounded by backticks
        if (text_stripped.startswith('`') and text_stripped.endswith('`')):
            return True
        
        # Contains code fence markers
        if text_stripped.startswith('```') or text_stripped.endswith('```'):
            return True
        
        # ─────────────────────────────────────────────────────────────────────
        # GUARD 5: Technical Syntax Patterns
        # ─────────────────────────────────────────────────────────────────────
        # Check for common technical patterns
        technical_patterns = [
            r'^[a-zA-Z_][a-zA-Z0-9_]*\(\)$',  # function()
            r'^[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*$',  # object.method
            r'^[A-Z_][A-Z0-9_]+$',  # CONSTANT_NAME
            r'^[\w\-]+\.[\w\-]+\.[\w\-]+$',  # package.subpackage.module
            r'^\{.*\}$',  # {variable}
            r'^\$\{.*\}$',  # ${variable}
            r'^<.*>$',  # <placeholder>
        ]
        
        for pattern in technical_patterns:
            if re.match(pattern, text_stripped):
                return True
        
        # ─────────────────────────────────────────────────────────────────────
        # GUARD 6: Context-Based Detection
        # ─────────────────────────────────────────────────────────────────────
        # If content_type is explicitly "code" or "command"
        content_type = context.get('content_type', '')
        if content_type in ['code', 'command', 'script', 'terminal']:
            return True
        
        # ─────────────────────────────────────────────────────────────────────
        # GUARD 7: Very Short Technical Identifiers
        # ─────────────────────────────────────────────────────────────────────
        # Single technical word without spaces (e.g., "kubectl", "npm", "git")
        word_count = len([t for t in sent if t.is_alpha])
        if word_count == 1 and not sent[0].is_sent_start:
            # Single word, possibly a command or identifier
            if re.match(r'^[a-z0-9\-_]+$', text_stripped):
                return True
        
        # Default: treat as prose
        return False
    
    def _is_list_item_fragment(self, sent: 'Span', context: Dict[str, Any]) -> bool:
        """
        Detect if this is a list item fragment (non-sentence) that shouldn't require a period.
        
        Fragments include:
        - Short phrases without verbs (e.g., "GitHub or GitLab for repositories")
        - Noun phrases (e.g., "Configuration options")
        - Prepositional phrases (e.g., "For production environments")
        - Imperative fragments at list start (e.g., "Enable security")
        
        Complete sentences that should have periods:
        - Statements with subject and verb (e.g., "The system processes the data")
        - Multiple clauses
        """
        
        # Check if we're in a list-like context
        block_type = context.get('block_type', '')
        is_list_context = any(marker in block_type.lower() for marker in ['list', 'table'])
        
        # If not in a list context, apply normal period rules
        if not is_list_context:
            return False
        
        # Analyze sentence structure
        sentence_text = sent.text.strip()
        word_count = len([token for token in sent if token.is_alpha])
        
        # Very short items are almost always fragments
        if word_count <= 3:
            return True
        
        # Check for verbs
        verbs = [token for token in sent if token.pos_ == 'VERB']
        has_finite_verb = any(token.tag_ in ['VBZ', 'VBP', 'VBD'] for token in verbs)
        
        # No verbs at all = fragment
        if not verbs:
            return True
        
        # Only gerunds or infinitives (VBG, VB with 'to') = likely fragment
        if verbs and not has_finite_verb:
            # Check if it's just gerunds or infinitives
            only_non_finite = all(token.tag_ in ['VBG', 'VB', 'VBN'] for token in verbs)
            if only_non_finite:
                return True
        
        # Check for complete sentence structure
        # A complete sentence typically has a subject (nsubj) and a finite verb
        has_subject = any(token.dep_ == 'nsubj' for token in sent)
        
        # If it has a finite verb AND a subject, it's likely a complete sentence
        if has_finite_verb and has_subject:
            return False  # This is a complete sentence, should have a period
        
        # Check if it's a simple noun phrase or prepositional phrase
        # Look at the root token
        root_token = [token for token in sent if token.dep_ == 'ROOT']
        if root_token:
            root = root_token[0]
            # If root is a noun or preposition, likely a fragment
            if root.pos_ in ['NOUN', 'PROPN', 'ADP']:
                return True
        
        # Check for list markers or enumeration indicators
        list_indicators = ['github', 'bitbucket', 'gitlab', 'quay', 'tekton', 'jenkins']
        text_lower = sentence_text.lower()
        if any(indicator in text_lower for indicator in list_indicators):
            # Common list pattern: "Option A, Option B, or Option C for purpose"
            if ' or ' in text_lower and ' for ' in text_lower:
                return True
        
        # Default: If we're in a list context and it's relatively short without clear sentence markers
        if is_list_context and word_count < 12:
            # If it doesn't have both subject and finite verb, treat as fragment
            if not (has_finite_verb and has_subject):
                return True
        
        # If we get here and it's in a list, default to assuming it's a fragment
        # unless it clearly has sentence structure
        return is_list_context and not (has_finite_verb and has_subject)
    
    def _get_sentence_for_position(self, position: int, text: str) -> str:
        """Get the sentence containing a specific text position."""
        # Find sentence boundaries around the position
        start = max(0, text.rfind('.', 0, position) + 1)
        end = text.find('.', position)
        if end == -1:
            end = len(text)
        
        return text[start:end].strip()
    
    def _get_sentence_index_for_position(self, position: int, doc: 'Doc') -> int:
        """Get the sentence index for a specific text position."""
        for i, sent in enumerate(doc.sents):
            if sent.start_char <= position < sent.end_char:
                return i
        return 0

    # === SMART MESSAGING ===

    def _get_contextual_period_message(self, violation_type: str, evidence_score: float, context: Dict[str, Any]) -> str:
        """Generate context-aware error message for period usage."""
        confidence_phrase = "clearly has" if evidence_score > 0.8 else ("likely has" if evidence_score > 0.6 else "may have")
        
        messages = {
            'abbreviation_periods': f"This text {confidence_phrase} unnecessary periods within abbreviations.",
            'duplicate_periods': f"This text {confidence_phrase} duplicate periods that should be corrected.",
            'unnecessary_period_in_heading': f"This heading {confidence_phrase} an unnecessary period at the end.",
            'unnecessary_period_in_list': f"This list item {confidence_phrase} an unnecessary period.",
            'missing_period': f"This sentence {confidence_phrase} a missing period at the end."
        }
        
        # Fallback for legacy calls without violation_type
        if violation_type == 'abbreviation_periods' or not violation_type:
            if evidence_score > 0.8:
                return "Avoid using periods within uppercase abbreviations."
            elif evidence_score > 0.6:
                return "Consider removing periods from this uppercase abbreviation."
            else:
                return "Review period usage in this abbreviation for style consistency."
        
        return messages.get(violation_type, f"This text {confidence_phrase} a period usage issue.")

    def _generate_smart_period_suggestions(self, period_token: 'Token', evidence_score: float, context: Dict[str, Any]) -> List[str]:
        """Generate context-aware suggestions for period usage."""
        suggestions = []
        
        if evidence_score > 0.7:
            suggestions.append("Remove the periods from the abbreviation (e.g., 'USA' instead of 'U.S.A.').")
            suggestions.append("Modern style guides prefer abbreviations without internal periods.")
        else:
            suggestions.append("Consider removing periods for a cleaner, modern style.")
            suggestions.append("Check your style guide's preference for abbreviation periods.")
        
        # Context-specific suggestions
        content_type = context.get('content_type', 'general')
        if content_type == 'technical':
            suggestions.append("Technical documentation typically uses abbreviations without periods.")
        elif content_type == 'legal':
            suggestions.append("Legal documents may require periods in abbreviations - check your style guide.")
        elif content_type == 'academic':
            suggestions.append("Academic style may vary - consult your institution's guidelines.")
        
        return suggestions[:3]
    
    def _generate_duplicate_periods_suggestions(self, evidence_score: float, context: Dict[str, Any]) -> List[str]:
        """Generate suggestions for duplicate periods."""
        return [
            "Remove the extra period.",
            "Use a single period to end sentences.",
            "Use ellipsis (...) if trailing off is intended."
        ]
    
    def _generate_unnecessary_period_suggestions(self, location_type: str, evidence_score: float, context: Dict[str, Any]) -> List[str]:
        """Generate suggestions for unnecessary periods."""
        if location_type == 'heading':
            return [
                "Remove the period from this heading.",
                "Headings are titles, not complete sentences.",
                "Use periods only for complete sentences."
            ]
        elif location_type == 'list_item':
            return [
                "Remove the period from this list item.",
                "Use periods only for complete sentences in lists.",
                "Keep list items concise without periods."
            ]
        else:
            return ["Remove the unnecessary period."]
    
    def _generate_missing_period_suggestions(self, evidence_score: float, context: Dict[str, Any]) -> List[str]:
        """Generate suggestions for missing periods."""
        return [
            "Add a period to end this sentence.",
            "Complete sentences should end with periods.",
            "Use proper punctuation for sentence endings."
        ]
