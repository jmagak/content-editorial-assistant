"""
Colons Rule
Based on IBM Style Guide topic: "Colons"

"""
from typing import List, Dict, Any, Optional
from .base_punctuation_rule import BasePunctuationRule

try:
    from spacy.tokens import Doc, Token, Span
except ImportError:
    Doc = None
    Token = None
    Span = None

class ColonsRule(BasePunctuationRule):
    """
    Checks for incorrect colon usage using evidence-based analysis,
    with dependency parsing and structural awareness.
    """
    def _get_rule_type(self) -> str:
        return 'colons'

    def analyze(self, text: str, sentences: List[str], nlp=None, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evidence-based analysis for colon usage:
          - Colons should be preceded by complete independent clauses
          - Various contexts legitimize colon usage (times, URLs, titles, lists)
        """
        # === UNIVERSAL CODE CONTEXT GUARD ===
        # Skip analysis for code blocks, listings, and literal blocks (technical syntax, not prose)
        if context and context.get('block_type') in ['listing', 'literal', 'code_block', 'inline_code']:
            return []
        errors: List[Dict[str, Any]] = []
        context = context or {}
        if not nlp:
            return errors

        is_list_introduction = context.get('is_list_introduction', False)
        if is_list_introduction:
            return []

        try:
            doc = nlp(text)
            for i, sent in enumerate(doc.sents):
                for token in sent:
                    if token.text == ':':
                        evidence_score = self._calculate_colon_evidence(token, sent, text, context)
                        
                        # Only flag if evidence suggests it's worth evaluating
                        if evidence_score > 0.1:
                            errors.append(self._create_error(
                                sentence=sent.text,
                                sentence_index=i,
                                message=self._get_contextual_colon_message(token, evidence_score, context),
                                suggestions=self._generate_smart_colon_suggestions(token, evidence_score, sent, context),
                                severity='low' if evidence_score < 0.7 else 'medium',
                                text=text,
                                context=context,
                                evidence_score=evidence_score,
                                span=(token.idx, token.idx + len(token.text)),
                                flagged_text=token.text
                            ))
        except IndexError as e:
            # This catch is a safeguard in case of unexpected SpaCy behavior
            errors.append(self._create_error(
                sentence=text,
                sentence_index=0,
                message=f"Rule ColonsRule failed with an indexing error: {e}",
                suggestions=["This may be a bug in the rule. Please report it."],
                severity='low',
                text=text,
                context=context,
                evidence_score=0.0  # No evidence when analysis fails
            ))
        return errors

    def _is_legitimate_context(self, colon_token: 'Token', sent: 'Span') -> bool:
        """
        Uses linguistic anchors to identify legitimate colon contexts.
        This version uses safe, sentence-relative indexing.
        """
        # colon_token.i is the index in the parent doc.
        # sent.start is the start index of the sentence in the parent doc.
        # So, the token's index *within the sentence* is:
        token_sent_idx = colon_token.i - sent.start

        # Check for time/ratios (e.g., 3:30, 2:1)
        if 0 < token_sent_idx < len(sent) - 1:
            prev_token = sent[token_sent_idx - 1]
            next_token = sent[token_sent_idx + 1]
            if prev_token.like_num and next_token.like_num:
                return True

        # Check for URLs (e.g., http:)
        if "http" in colon_token.head.text.lower():
            return True

        # Check for Title: Subtitle patterns
        if colon_token.head.pos_ in ("NOUN", "PROPN") and colon_token.head.is_title:
             if token_sent_idx < len(sent) - 1 and sent[token_sent_idx + 1].is_title:
                return True

        return False

    def _is_preceded_by_complete_clause(self, colon_token: 'Token', sent: 'Span') -> bool:
        """
        Checks if tokens before the colon form a complete independent clause.
        Enhanced to properly handle:
        - Imperatives (implied "you" subject)
        - Infinitive phrases ending with imperatives (To do X, verb:)
        - Complex imperative clauses (If Y, verb:)
        - Procedural labels (Optional:, Note:)
        - Prepositions and passive voice
        - Standard subject-verb structures
        """
        if colon_token.i <= sent.start:
            return False

        # Create a new doc object from the span before the colon for accurate parsing
        clause_span = sent.doc[sent.start : colon_token.i]
        clause_doc = clause_span.as_doc()
        
        # Get the text for pattern matching
        clause_text = clause_span.text.lower().strip()

        # === ZERO FALSE POSITIVE GUARD 1: Procedural Labels ===
        procedural_labels = [
            'optional', 'note', 'important', 'tip', 'warning', 'caution',
            'example', 'remember', 'attention', 'notice', 'prerequisite'
        ]
        
        # Check if clause is EXACTLY one of these labels (single word before colon)
        # This catches patterns like "Optional:" or "Note:" at the start of a sentence
        if clause_text in procedural_labels:
            return True  # Valid procedural label pattern
        
        for label in procedural_labels:
            if clause_text.startswith(label + ': ') or clause_text.startswith(label + ':'):
                return True  # Valid procedural label pattern (multi-colon case)
            if clause_text.startswith(label + ' ') or clause_text.startswith(label + '\t'):
                return True  # Valid procedural label pattern (e.g., "Optional step:")
        
        # === ZERO FALSE POSITIVE GUARD 2: Contextual Scope Labels ===
        # Pattern: "For Jenkins only:", "For administrators only:", "On Windows:", "In production mode:"
        # These are standard technical documentation conventions for scoping instructions
        #
        # Objective Truth: 
        #   - Technical documentation uses "For X only:" to scope procedural content
        #   - This is standard in product docs (Red Hat, Microsoft, IBM)
        #   - These are introductory phrases, not incomplete clauses
        #
        # False Negative Risk: Minimal
        #   - Real errors like "For:" or "For the:" don't match these patterns
        #   - Patterns require noun/proper noun after "For", "On", "In"
        #
        # Inversion Test: Passed
        #   - Only matches specific technical documentation patterns
        #   - Requires: "For {noun} only:", "On {noun}:", "In {noun} mode:"
        #   - Doesn't match incomplete clauses like "For the:" or "To configure:"
        #
        import re
        
        # Pattern 1: "For X only:" (where X is a noun/proper noun)
        # Examples: "For Jenkins only:", "For administrators only:", "For Linux users only:"
        for_only_pattern = r'^for\s+[\w\s]+\s+only$'
        if re.match(for_only_pattern, clause_text, re.IGNORECASE):
            # Additional validation: ensure there's at least one word between "for" and "only"
            words = clause_text.split()
            if len(words) >= 3:  # "for", at least one word, "only"
                return True  # Valid "For X only:" pattern
        
        # Pattern 2: "On X:" (where X is a platform/system name)
        # Examples: "On Windows:", "On Linux:", "On the production server:"
        on_platform_pattern = r'^on\s+[\w\s]+$'
        if re.match(on_platform_pattern, clause_text, re.IGNORECASE):
            words = clause_text.split()
            if len(words) >= 2:  # "on", at least one word
                return True  # Valid "On X:" pattern
        
        # Pattern 3: "In X mode:" or "In X:" (where X is a mode/environment)
        # Examples: "In production:", "In development mode:", "In the configuration file:"
        in_mode_pattern = r'^in\s+[\w\s]+$'
        if re.match(in_mode_pattern, clause_text, re.IGNORECASE):
            words = clause_text.split()
            if len(words) >= 2:  # "in", at least one word
                return True  # Valid "In X:" pattern
        
        # === NEW GUARD 2: Infinitive Phrase Introductions ===
        # Pattern: "To use DHCP, enter:", "To set a static IP, type:", "To configure X, run:"
        # These are conventional command introductions in technical documentation
        if clause_text.startswith('to '):
            # Common imperative verbs that follow infinitive phrases
            command_verbs = [
                ', enter', ', type', ', run', ', execute', ', use', ', set',
                ', configure', ', install', ', create', ', delete', ', add',
                ', remove', ', modify', ', change', ', update', ', click',
                ', select', ', choose', ', open', ', close', ', start', ', stop'
            ]
            
            # Check if the clause ends with comma + imperative verb
            if any(clause_text.endswith(verb_pattern) for verb_pattern in command_verbs):
                return True  # Valid infinitive phrase + imperative pattern
        
        # === NEW GUARD 3: Step Number Labels ===
        # Pattern: "Step 1:", "Step 2:", "1:", "2:", "a:", "b:"
        # Common in procedural documentation
        import re
        # Match patterns like "step 1:", "1.", "a)", etc.
        if re.match(r'^(step\s+)?\d+[.:\)]?\s*$', clause_text) or \
           re.match(r'^[a-z][.:\)]?\s*$', clause_text):
            return True  # Valid step/list marker
        
        # === NEW GUARD 4: Action + Object Patterns ===
        # Pattern: "Select the option:", "Click the button:", "Choose the file:"
        # These are complete imperative commands
        action_object_pattern = re.match(
            r'^(select|click|choose|press|hit|open|close|start|stop|enable|disable|activate|deactivate)\s+\S+',
            clause_text
        )
        if action_object_pattern:
            return True  # Valid action + object imperative

        has_subject = any(t.dep_ in ('nsubj', 'nsubjpass') for t in clause_doc)
        has_root_verb = any(t.dep_ == 'ROOT' for t in clause_doc)
        
        # === EXISTING FIX: Detect infinitive phrases ending with imperatives ===
        # Pattern: "To use DHCP, enter:" or "To set a static IPv4 address, enter:"
        # Structure: TO + verb + [objects/modifiers] + COMMA + imperative verb + COLON
        if len(clause_doc) >= 3:
            first_token = clause_doc[0]
            # Check if starts with "To" (infinitive marker)
            if first_token.text.lower() == 'to' and first_token.pos_ == 'PART':
                # Look for comma in the clause
                comma_indices = [i for i, t in enumerate(clause_doc) if t.text == ',']
                if comma_indices:
                    # Check if there's an imperative verb after the last comma
                    last_comma_idx = comma_indices[-1]
                    if last_comma_idx < len(clause_doc) - 1:
                        # Get tokens after comma
                        tokens_after_comma = clause_doc[last_comma_idx + 1:]
                        if len(tokens_after_comma) > 0:
                            first_after_comma = tokens_after_comma[0]
                            # Check if it's an imperative verb (VB or VBP tag)
                            # SpaCy sometimes tags imperatives as VBP (present) instead of VB (base)
                            if first_after_comma.tag_ in ['VB', 'VBP'] and first_after_comma.pos_ == 'VERB':
                                # This is a valid infinitive phrase + imperative pattern
                                return True
        
        # === NEW FIX: Detect complex imperative clauses ===
        # Pattern: "If you want to create..., enter:" or "When the system starts, run:"
        # Structure: subordinate clause + COMMA + imperative verb + COLON
        if len(clause_doc) >= 3:
            first_token = clause_doc[0]
            # Check if starts with subordinating conjunction (if, when, while, etc.)
            if first_token.pos_ == 'SCONJ' or (first_token.text.lower() in ['if', 'when', 'while', 'after', 'before', 'unless']):
                # Look for comma in the clause
                comma_indices = [i for i, t in enumerate(clause_doc) if t.text == ',']
                if comma_indices:
                    # Check if there's an imperative verb after the last comma
                    last_comma_idx = comma_indices[-1]
                    if last_comma_idx < len(clause_doc) - 1:
                        # Get tokens after comma
                        tokens_after_comma = clause_doc[last_comma_idx + 1:]
                        if len(tokens_after_comma) > 0:
                            first_after_comma = tokens_after_comma[0]
                            # Check if it's an imperative verb (VB or VBP tag)
                            # SpaCy sometimes tags imperatives as VBP (present) instead of VB (base)
                            if first_after_comma.tag_ in ['VB', 'VBP'] and first_after_comma.pos_ == 'VERB':
                                # This is a valid complex imperative pattern
                                return True
        
        # === ORIGINAL FIX: Detect simple imperative sentences ===
        # Imperatives have an implied "you" subject and start with a base-form verb
        # Example: "List the profiles:" → "List" (VB tag, ROOT or has ROOT verb)
        if len(clause_doc) > 0:
            first_token = clause_doc[0]
            # Check if first token is a base-form verb (VB) indicating imperative
            is_imperative = (
                first_token.tag_ == 'VB' and  # Base form verb
                first_token.pos_ == 'VERB' and  # Is a verb
                (first_token.dep_ == 'ROOT' or has_root_verb)  # Is or has the main verb
            )
            
            if is_imperative:
                # Imperative sentences are complete clauses (implied subject "you")
                # But still check for incomplete patterns
                token_sent_idx = colon_token.i - sent.start
                if token_sent_idx > 0:
                    prev_token = sent[token_sent_idx - 1]
                    # Even imperatives shouldn't end with prepositions/articles before colon
                    # Check both POS and tag because SpaCy can tag articles as PRON
                    if prev_token.pos_ in ['ADP', 'DET', 'CCONJ', 'SCONJ'] or prev_token.tag_ == 'DT':
                        return False
                return True
        
        # Check what directly precedes the colon (for non-imperatives)
        token_sent_idx = colon_token.i - sent.start
        if token_sent_idx > 0:
            prev_token = sent[token_sent_idx - 1]
            
            # Prepositions before colon indicate incomplete clause
            if prev_token.pos_ == 'ADP':  # Preposition like "with:", "for:", "to:"
                return False
            
            # Articles before colon indicate incomplete clause
            if prev_token.pos_ == 'DET':  # "the:", "a:", "an:"
                return False
            
            # Conjunctions before colon indicate incomplete clause
            if prev_token.pos_ in ['CCONJ', 'SCONJ']:  # "and:", "but:", "if:"
                return False
            
            # Verbs directly before colon are usually incomplete
            if prev_token.pos_ == 'VERB' and prev_token.dep_ != 'ROOT':
                return False

        # Standard declarative: Must have both subject and verb
        if not (has_subject and has_root_verb):
            return False  # Does not have basic S-V structure
        
        # === NEW & MORE ROBUST TRANSITIVE VERB CHECK ===
        if len(clause_doc) > 0:
            last_token_before_colon = clause_doc[-1]
            if last_token_before_colon.pos_ == 'VERB':
                common_transitive_verbs = {
                    'cause', 'provide', 'include', 'require', 'contain', 'list', 'describe', 
                    'explain', 'show', 'produce', 'generate', 'result'
                }
                if last_token_before_colon.lemma_ in common_transitive_verbs:
                    # The sentence ends with a transitive verb right before the colon.
                    # This is a strong signal of an incomplete fragment.
                    return False
        
        # If all checks pass, it's a complete clause.
        return True

    def _is_legitimate_context_aware(self, colon_token: 'Token', sent: 'Span', context: Optional[Dict[str, Any]]) -> bool:
        """
        LINGUISTIC ANCHOR: Context-aware colon legitimacy checking using structural information.
        Uses inter-block context to determine if colons are introducing content like admonitions.
        """
        if not context:
            return False
        
        # If this block introduces an admonition, colons are legitimate
        if context.get('next_block_type') == 'admonition':
            return True
        
        # If we're in a list introduction context, colons are legitimate
        if context.get('is_list_introduction', False):
            return True
        
        return False

    def _is_definition_list_item(self, colon_token: 'Token', sent: 'Span', context: Dict[str, Any]) -> bool:
        """
        WORLD-CLASS GUARD: Surgical detection of definition list items.
        
        Detects the pattern: [Term] : [Definition]
        This is a legitimate grammatical pattern in technical writing, especially
        within list structures where terms are defined inline.
        
        Examples of valid definition list patterns:
        - `intel_idle`: This is the default driver on hosts with an Intel CPU
        - `acpi_idle`: RHEL uses this driver on hosts with CPUs from vendors
        - API endpoint: The URL used to access the service
        - Configuration parameter: A setting that controls behavior
        
        Detection Strategy (Multi-Tier):
        1. Structural Context: Must be in a list block
        2. Position Analysis: Colon should appear early in the sentence (short term)
        3. Linguistic Pattern: Term before colon should be noun/technical term/code
        4. Term Characteristics: Short, often highlighted as code with backticks
        
        Args:
            colon_token: The colon token being evaluated
            sent: The sentence containing the colon
            context: Structural context about the document
            
        Returns:
            True if this is a definition list item (suppress the violation)
            False if this should be evaluated for colon correctness
        """
        
        # === TIER 1: STRUCTURAL CONTEXT ===
        # Definition lists appear within list structures in technical documentation
        block_type = context.get('block_type', '')
        
        # Check for list context - various formats used in different markup languages
        list_block_types = {
            'ulist', 'olist', 'dlist',  # AsciiDoc
            'list_item', 'unordered_list_item', 'ordered_list_item',  # Generic
            'bullet_list', 'enumerated_list', 'definition_list',  # reStructuredText
            'ul', 'ol', 'dl',  # HTML-like
            'list', 'item'  # Generic fallback
        }
        
        if block_type not in list_block_types:
            # Not in a list context - not a definition list item
            return False
        
        # === TIER 2: POSITION ANALYSIS ===
        # Definition list colons appear early in the sentence, after a short term
        # Calculate token position within the sentence
        token_sent_idx = colon_token.i - sent.start
        
        # If the colon appears after position 5, it's unlikely to be a definition term
        # Definition terms are typically 1-5 tokens long
        if token_sent_idx > 5:
            return False
        
        # The colon should not be at the very start (position 0)
        if token_sent_idx == 0:
            return False
        
        # === TIER 3: LINGUISTIC PATTERN ANALYSIS ===
        # Extract the term (text before the colon within this sentence)
        term_span = sent[0:token_sent_idx]
        
        if len(term_span) == 0:
            return False
        
        # === TIER 3A: Code/Technical Term Detection ===
        # Technical documentation often highlights terms with backticks or code formatting
        term_text = term_span.text
        
        # Check for code markers (backticks, code tags)
        is_code_term = '`' in term_text or '<code>' in term_text or '</code>' in term_text
        
        # === TIER 3B: Linguistic Features ===
        # Definition terms are typically nouns, proper nouns, or technical identifiers
        has_noun = any(t.pos_ in ['NOUN', 'PROPN'] for t in term_span)
        
        # Check for proper nouns (often product names, API names)
        has_proper_noun = any(t.tag_ == 'NNP' for t in term_span)
        
        # Check for underscores or hyphens (common in technical terms)
        has_technical_chars = '_' in term_text or '-' in term_text
        
        # === TIER 4: TERM LENGTH ANALYSIS ===
        # Definition terms should be concise (1-4 tokens typically)
        term_length = len(term_span)
        is_short_term = term_length <= 4
        
        # === TIER 5: POST-COLON ANALYSIS ===
        # After the colon, there should be explanatory text (the definition)
        # This distinguishes definition lists from other colon uses
        tokens_after_colon = len(sent) - token_sent_idx - 1
        
        # There should be substantial text after the colon (at least 3 tokens)
        has_definition_text = tokens_after_colon >= 3
        
        # === DECISION LOGIC ===
        # Combine multiple signals to make a confident decision
        
        # Strong signals (any one is sufficient with supporting evidence)
        if is_code_term and is_short_term and has_definition_text:
            # Code term in a list with explanation: very likely definition list
            return True
        
        # Multiple moderate signals
        if is_short_term and has_definition_text:
            # Check for at least one linguistic feature
            if has_noun or has_proper_noun or has_technical_chars:
                return True
        
        # Technical term pattern (underscores/hyphens are strong signals)
        if has_technical_chars and is_short_term and has_definition_text:
            return True
        
        # Proper noun pattern (API names, product names)
        if has_proper_noun and is_short_term and has_definition_text:
            return True
        
        return False  # Not a definition list item pattern

    # === EVIDENCE CALCULATION ===

    def _calculate_colon_evidence(self, colon_token: 'Token', sent: 'Span', text: str, context: Dict[str, Any]) -> float:
        """
        Calculate evidence (0.0-1.0) that a colon usage is incorrect.
        
        Higher scores indicate stronger evidence of an error.
        Lower scores indicate acceptable usage or ambiguous cases.
        """
        # === WORLD-CLASS GUARD: Definition List Item Pattern ===
        # Detect and suppress valid definition list patterns: [Term] : [Definition]
        # This is a legitimate grammatical pattern in technical documentation
        if self._is_definition_list_item(colon_token, sent, context):
            return 0.0  # Zero evidence for valid definition list patterns
        
        # === ZERO FALSE POSITIVE GUARD: Instructional phrase before code block ===
        # Standard technical writing pattern: instruction + colon + code block
        # Example: "create the /etc/systemd/network/ directory:" followed by code block
        # This is structurally correct and should never be flagged
        if context and context.get('next_block_type') in ['listing', 'literal', 'code_block']:
            return 0.0  # This is a standard technical writing pattern, not an error.
        
        evidence_score = 0.0
        
        # === STEP 1: BASE EVIDENCE ASSESSMENT ===
        # Start with assumption that problematic colons need high evidence
        if self._is_legitimate_context(colon_token, sent) or self._is_legitimate_context_aware(colon_token, sent, context):
            return 0.0  # Legitimate contexts get no evidence
        
        if not self._is_preceded_by_complete_clause(colon_token, sent):
            evidence_score = 0.8  # Strong evidence of incorrect usage
        else:
            return 0.0  # Complete clause before colon is generally correct
        
        # === STEP 2: LINGUISTIC CLUES (MICRO-LEVEL) ===
        evidence_score = self._apply_linguistic_clues_colon(evidence_score, colon_token, sent)
        
        # === STEP 3: STRUCTURAL CLUES (MESO-LEVEL) ===
        evidence_score = self._apply_structural_clues_colon(evidence_score, colon_token, context)
        
        # === STEP 4: SEMANTIC CLUES (MACRO-LEVEL) ===
        evidence_score = self._apply_semantic_clues_colon(evidence_score, colon_token, text, context)
        
        # === STEP 5: FEEDBACK PATTERNS (LEARNING CLUES) ===
        evidence_score = self._apply_feedback_clues_colon(evidence_score, colon_token, context)
        
        return max(0.0, min(1.0, evidence_score))

    def _apply_linguistic_clues_colon(self, evidence_score: float, colon_token: 'Token', sent: 'Span') -> float:
        """Apply SpaCy-based linguistic analysis clues for colon usage."""
        
        token_sent_idx = colon_token.i - sent.start
        
        # Check preceding token patterns
        if token_sent_idx > 0:
            prev_token = sent[token_sent_idx - 1]
            
            # Verb immediately before colon often indicates incomplete clause
            if prev_token.pos_ == 'VERB':
                evidence_score += 0.2
            
            # Preposition before colon is problematic
            if prev_token.pos_ == 'ADP':
                evidence_score += 0.3
            
            # Article before colon is very problematic
            if prev_token.pos_ == 'DET':
                evidence_score += 0.4
            
            # Conjunction before colon suggests incomplete thought
            if prev_token.pos_ in ['CCONJ', 'SCONJ']:
                evidence_score += 0.3
        
        # Check for common legitimate patterns we might have missed
        if token_sent_idx > 1 and token_sent_idx < len(sent) - 1:
            # Pattern: "Note: ..." or "Warning: ..."
            if token_sent_idx == 1:
                first_token = sent[0]
                if first_token.text.lower() in ['note', 'warning', 'tip', 'important', 'caution']:
                    evidence_score -= 0.6
            
            # Pattern: "Chapter 1: Introduction"
            prev_prev = sent[token_sent_idx - 2] if token_sent_idx > 1 else None
            prev_token = sent[token_sent_idx - 1]
            if prev_prev and prev_prev.text.lower() in ['chapter', 'section', 'part', 'step'] and prev_token.like_num:
                evidence_score -= 0.5
        
        # Check sentence length - very short sentences with colons are often labels
        if len(sent) <= 3:
            evidence_score -= 0.3
        
        return evidence_score

    def _apply_structural_clues_colon(self, evidence_score: float, colon_token: 'Token', context: Dict[str, Any]) -> float:
        """Apply document structure-based clues for colon usage."""
        
        block_type = context.get('block_type', 'paragraph')
        
        # Headings often use colons legitimately
        if block_type in ['heading', 'title']:
            evidence_score -= 0.5
        
        # List items may use colons for definitions
        elif block_type in ['ordered_list_item', 'unordered_list_item']:
            evidence_score -= 0.3
        
        # Admonitions commonly use colons
        elif block_type == 'admonition':
            evidence_score -= 0.4
        
        # Table cells may use colons for ratios or labels
        elif block_type in ['table_cell', 'table_header']:
            evidence_score -= 0.2
        
        # Code blocks have different punctuation rules
        elif block_type in ['code_block', 'literal_block']:
            evidence_score -= 0.7
        
        # Next block type context
        if context.get('next_block_type') in ['ordered_list', 'unordered_list']:
            evidence_score -= 0.4  # Introducing a list
        
        return evidence_score

    def _apply_semantic_clues_colon(self, evidence_score: float, colon_token: 'Token', text: str, context: Dict[str, Any]) -> float:
        """
        Apply semantic and content-type clues for colon usage.
        Enhanced with surgical zero-false-positive guard for procedural imperatives.
        """
        
        content_type = context.get('content_type', 'general')
        domain = context.get('domain', 'general')
        audience = context.get('audience', 'general')
        block_type = context.get('block_type', 'paragraph')
        
        # === SURGICAL ZERO FALSE POSITIVE GUARD FOR PROCEDURAL IMPERATIVES ===
        # This is the CRITICAL fix for procedural step introductions
        # Pattern: COMPLETE imperative + colon in list/procedural context
        # Example: "List the NetworkManager connection profiles:" ✅
        # Counter-example: "Review the:" ❌ (incomplete - ends with article)
        
        # Check if this is a COMPLETE imperative sentence
        sent = colon_token.sent
        clause_before_colon = sent.doc[sent.start : colon_token.i]
        is_complete_imperative = False
        
        if len(clause_before_colon) > 0:
            clause_doc = clause_before_colon.as_doc()
            first_token = clause_doc[0]
            
            # Check if it starts with an imperative verb
            starts_with_imperative = (
                first_token.tag_ == 'VB' and  # Base form verb
                first_token.pos_ == 'VERB'     # Is a verb
            )
            
            if starts_with_imperative:
                # Also verify it doesn't end with incomplete patterns
                token_sent_idx = colon_token.i - sent.start
                if token_sent_idx > 0:
                    prev_token = sent[token_sent_idx - 1]
                    # Must not end with preposition/article/conjunction
                    # Check both POS and tag because SpaCy can tag articles as PRON
                    ends_incomplete = (prev_token.pos_ in ['ADP', 'DET', 'CCONJ', 'SCONJ'] or 
                                      prev_token.tag_ == 'DT')
                    is_complete_imperative = starts_with_imperative and not ends_incomplete
                else:
                    is_complete_imperative = starts_with_imperative
        
        # If this is a COMPLETE imperative in a list/procedural context → ZERO evidence
        if is_complete_imperative:
            is_list_context = block_type in [
                'ordered_list_item', 'unordered_list_item', 
                'list_item', 'olist', 'ulist'
            ]
            is_procedural = content_type in ['procedural', 'procedure', 'task']
            
            if is_list_context or is_procedural:
                # This is the STANDARD pattern for procedural writing
                # Complete imperatives introducing lists/steps are 100% correct
                return 0.0  # Zero evidence - this is correct usage
        
        # === STANDARD CONTENT TYPE ADJUSTMENTS ===
        
        # Technical content often uses colons for definitions and ratios
        if content_type == 'technical':
            evidence_score -= 0.1
        
        # Academic writing has more structured colon usage
        elif content_type == 'academic':
            evidence_score -= 0.05
        
        # Legal writing is very structured
        elif content_type == 'legal':
            evidence_score += 0.05  # Be stricter
        
        # Marketing content more creative but should still follow rules
        elif content_type == 'marketing':
            evidence_score -= 0.05
        
        # Procedural content often uses colons for step introductions
        elif content_type in ['procedural', 'procedure', 'task']:
            evidence_score -= 0.3  # Stronger adjustment for procedural
        
        # Domain-specific adjustments
        if domain in ['software', 'engineering']:
            evidence_score -= 0.1  # More technical contexts
        elif domain in ['finance', 'legal']:
            evidence_score += 0.05  # More formal contexts
        
        # Expert audiences more familiar with technical colon usage
        if audience in ['expert', 'developer']:
            evidence_score -= 0.1
        elif audience in ['beginner', 'general']:
            evidence_score += 0.05  # Be more helpful
        
        return evidence_score

    def _apply_feedback_clues_colon(self, evidence_score: float, colon_token: 'Token', context: Dict[str, Any]) -> float:
        """Apply clues learned from user feedback patterns for colon usage."""
        
        feedback_patterns = self._get_cached_feedback_patterns_colon()
        
        # Get context around the colon
        token_sent_idx = colon_token.i - colon_token.sent.start
        sent = colon_token.sent
        
        # Look for patterns in accepted/rejected colon usage
        if token_sent_idx > 0:
            prev_word = sent[token_sent_idx - 1].text.lower()
            
            # Words commonly accepted before colons
            if prev_word in feedback_patterns.get('accepted_preceding_words', set()):
                evidence_score -= 0.3
            
            # Words commonly flagged before colons
            elif prev_word in feedback_patterns.get('flagged_preceding_words', set()):
                evidence_score += 0.2
        
        # Context-specific patterns
        block_type = context.get('block_type', 'paragraph')
        block_patterns = feedback_patterns.get(f'{block_type}_colon_patterns', {})
        
        if 'accepted_rate' in block_patterns:
            acceptance_rate = block_patterns['accepted_rate']
            if acceptance_rate > 0.8:
                evidence_score -= 0.2  # High acceptance in this context
            elif acceptance_rate < 0.3:
                evidence_score += 0.1  # Low acceptance in this context
        
        return evidence_score

    def _get_cached_feedback_patterns_colon(self) -> Dict[str, Any]:
        """Load feedback patterns for colon usage from cache or feedback analysis."""
        return {
            'accepted_preceding_words': {'following', 'below', 'these', 'note', 'warning', 'example'},
            'flagged_preceding_words': {'the', 'a', 'an', 'to', 'for', 'with'},
            'paragraph_colon_patterns': {'accepted_rate': 0.6},
            'heading_colon_patterns': {'accepted_rate': 0.9},
            'list_item_colon_patterns': {'accepted_rate': 0.8},
        }

    # === SMART MESSAGING ===

    def _get_contextual_colon_message(self, colon_token: 'Token', evidence_score: float, context: Dict[str, Any]) -> str:
        """Generate context-aware error message for colon usage."""
        
        if evidence_score > 0.8:
            return "Incorrect colon usage: A colon must be preceded by a complete independent clause."
        elif evidence_score > 0.6:
            return "Consider revising colon usage: Ensure the text before the colon forms a complete thought."
        elif evidence_score > 0.4:
            return "Colon usage may be unclear: Check if the preceding text is a complete sentence."
        else:
            return "Review colon usage for clarity and grammatical correctness."

    def _generate_smart_colon_suggestions(self, colon_token: 'Token', evidence_score: float, sent: 'Span', context: Dict[str, Any]) -> List[str]:
        """Generate context-aware suggestions for colon usage."""
        
        suggestions = []
        block_type = context.get('block_type', 'paragraph')
        
        # High evidence suggestions
        if evidence_score > 0.7:
            suggestions.append("Rewrite the text before the colon to form a complete sentence.")
            suggestions.append("Remove the colon if it's not introducing a list, quote, or explanation.")
        
        # Medium evidence suggestions
        elif evidence_score > 0.4:
            suggestions.append("Ensure the clause before the colon can stand alone as a sentence.")
            if block_type in ['paragraph', 'list_item']:
                suggestions.append("Consider using a period and starting a new sentence instead.")
        
        # Context-specific suggestions
        if block_type == 'heading':
            suggestions.append("In headings, colons can separate main topics from subtopics.")
        elif context.get('next_block_type') in ['ordered_list', 'unordered_list']:
            suggestions.append("Colons can introduce lists when preceded by a complete statement.")
        elif block_type == 'admonition':
            suggestions.append("Admonition labels (Note:, Warning:) commonly use colons.")
        
        # General guidance
        if len(suggestions) < 2:
            suggestions.append("Use colons to introduce explanations, lists, or quotations.")
            suggestions.append("Ensure proper grammar in the clause preceding the colon.")
        
        return suggestions[:3]