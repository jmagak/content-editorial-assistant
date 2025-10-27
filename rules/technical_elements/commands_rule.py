"""
Commands Rule (Production-Grade)
Based on IBM Style Guide topic: "Commands"
Evidence-based analysis with surgical zero false positive guards for command usage.
Uses YAML-based configuration for maintainable pattern management.
"""
from typing import List, Dict, Any
from .base_technical_rule import BaseTechnicalRule
from .services.technical_config_service import TechnicalConfigServices, TechnicalContext
import re

try:
    from spacy.tokens import Doc
except ImportError:
    Doc = None

class CommandsRule(BaseTechnicalRule):
    """
    PRODUCTION-GRADE: Checks for the incorrect use of command names as verbs.
    
    Implements rule-specific evidence calculation for:
    - Command names used as verbs instead of proper command syntax
    - Context-aware detection of genuine command misuse vs. legitimate verb usage
    - Technical domain appropriateness checking
    
    Features:
    - YAML-based configuration for maintainable pattern management
    - Surgical zero false positive guards for command contexts
    - Dynamic base evidence scoring based on command specificity
    - Evidence-aware messaging and suggestions for command formatting
    """
    
    def __init__(self):
        """Initialize with YAML configuration service."""
        super().__init__()
        self.config_service = TechnicalConfigServices.commands()
    
    def _get_rule_type(self) -> str:
        return 'technical_commands'

    def analyze(self, text: str, sentences: List[str], nlp=None, context=None) -> List[Dict[str, Any]]:
        """
        PRODUCTION-GRADE: Evidence-based analysis for command usage violations.
        
        Implements the required production pattern:
        1. Find potential command misuse using rule-specific detection
        2. Calculate evidence using rule-specific _calculate_command_evidence()
        3. Apply zero false positive guards specific to command analysis
        4. Use evidence-aware messaging and suggestions
        """
        # === UNIVERSAL CODE CONTEXT GUARD ===
        # Skip analysis for code blocks, listings, and literal blocks (technical syntax, not prose)
        if context and context.get('block_type') in ['listing', 'literal', 'code_block', 'inline_code']:
            return []
        errors: List[Dict[str, Any]] = []
        if not nlp:
            return errors

        doc = nlp(text)
        context = context or {}

        # === STEP 1: Find potential command misuse ===
        potential_issues = self._find_potential_command_issues(doc, text, context)
        
        # === STEP 2: Process each potential issue with evidence calculation ===
        for issue in potential_issues:
            # Calculate rule-specific evidence score
            evidence_score = self._calculate_command_evidence(
                issue, doc, text, context
            )
            
            # Only create error if evidence suggests it's worth evaluating
            if evidence_score > 0.1:  # Low threshold - let enhanced validation decide
                error = self._create_error(
                    sentence=issue['sentence'],
                    sentence_index=issue['sentence_index'],
                    message=self._generate_command_specific_message(issue, evidence_score),
                    suggestions=self._generate_command_specific_suggestions(issue, evidence_score, context),
                    severity='low' if evidence_score < 0.7 else 'medium',
                    text=text,
                    context=context,
                    evidence_score=evidence_score,
                    span=issue.get('span', [0, 0]),
                    flagged_text=issue.get('command_word', issue.get('text', ''))
                )
                errors.append(error)
        
        return errors
    
    # === RULE-SPECIFIC METHODS ===
    
    def _find_potential_command_issues(self, doc, text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find potential command misuse for evidence assessment.
        Detects command words used as verbs in inappropriate contexts.
        Uses YAML-based configuration for maintainable pattern management.
        """
        issues = []
        
        # Load command patterns from YAML configuration
        command_patterns = self.config_service.get_patterns()
        
        # Build command words dictionary from YAML patterns
        command_words = {}
        for pattern_id, pattern_config in command_patterns.items():
            command_word = pattern_config.pattern
            command_words[command_word] = pattern_config.evidence
        
        for i, sent in enumerate(doc.sents):
            for token in sent:
                command_word = token.lemma_.lower()
                
                # Check if this is a known command word used as a verb
                if (command_word in command_words and 
                    hasattr(token, 'pos_') and token.pos_ == 'VERB'):
                    
                    # Find the pattern config for this command
                    pattern_config = None
                    for pid, pconfig in command_patterns.items():
                        if pconfig.pattern == command_word:
                            pattern_config = pconfig
                            break
                    
                    issues.append({
                        'type': 'command',
                        'subtype': 'command_as_verb',
                        'command_word': command_word,
                        'text': token.text,
                        'sentence': sent.text,
                        'sentence_index': i,
                        'span': [token.idx, token.idx + len(token.text)],
                        'base_evidence': command_words[command_word],
                        'token': token,
                        'sentence_obj': sent,
                        'pattern_config': pattern_config  # Include pattern config for legitimate patterns
                    })
        
        return issues
    
    def _calculate_command_evidence(self, issue: Dict[str, Any], doc, text: str, context: Dict[str, Any]) -> float:
        """
        PRODUCTION-GRADE: Calculate evidence score (0.0-1.0) for command violations.
        
        Implements rule-specific evidence calculation with:
        - Zero false positive guards for command analysis
        - Dynamic base evidence based on command specificity
        - Context-aware adjustments for technical documentation
        """
        
        # === SURGICAL ZERO FALSE POSITIVE GUARDS FOR COMMANDS ===
        # Apply ultra-precise command-specific guards that eliminate false positives
        # while preserving ALL legitimate command violations
        
        token = issue.get('token')
        if not token:
            return 0.0
            
        command_word = issue.get('command_word', '')
        sentence_obj = issue.get('sentence_obj')
        
        # === GUARD 1: LEGITIMATE VERB USAGE ===
        # Don't flag when command words are used as legitimate verbs
        if self._is_legitimate_verb_usage(command_word, token, sentence_obj, context):
            return 0.0  # Legitimate verb usage, not command misuse
            
        # === GUARD 2: QUOTED COMMAND EXAMPLES ===
        # Don't flag command words in direct quotes or examples
        if self._is_in_quoted_command_example(token, sentence_obj, context):
            return 0.0  # Quoted examples are not violations
            
        # === GUARD 3: PROPER COMMAND SYNTAX CONTEXT ===
        # Don't flag when proper command syntax is already present
        if self._has_proper_command_syntax(command_word, sentence_obj, context):
            return 0.0  # Proper command syntax already used
            
        # === GUARD 4: NON-TECHNICAL CONTEXT ===
        # Don't flag common verbs in non-technical contexts
        if self._is_non_technical_verb_context(command_word, sentence_obj, context):
            return 0.0  # Non-technical usage is acceptable
            
        # Apply selective technical guards (skip technical context guard for commands)
        # Commands are violations even in technical contexts when used as verbs
        
        # Check code blocks and literal blocks
        block_type = context.get('block_type', 'paragraph')
        if block_type in ['code_block', 'literal_block', 'inline_code']:
            return 0.0  # Code has its own formatting rules
            
        # Check entities (but not technical context)
        if hasattr(token, 'ent_type_') and token.ent_type_:
            if token.ent_type_ in ['ORG', 'PRODUCT', 'GPE', 'PERSON']:
                return 0.0  # Company names, product names, etc.
        
        # === DYNAMIC BASE EVIDENCE ASSESSMENT ===
        evidence_score = issue.get('base_evidence', 0.7)  # Command-specific base score
        
        # === LINGUISTIC CLUES (COMMAND-SPECIFIC) ===
        evidence_score = self._apply_command_linguistic_clues(evidence_score, issue, sentence_obj)
        
        # === STRUCTURAL CLUES ===
        evidence_score = self._apply_technical_structural_clues(evidence_score, context)
        
        # === SEMANTIC CLUES ===
        evidence_score = self._apply_command_semantic_clues(evidence_score, issue, text, context)
        
        return max(0.0, min(1.0, evidence_score))  # Clamp to valid range
    
    # === SURGICAL ZERO FALSE POSITIVE GUARD METHODS ===
    
    def _is_legitimate_verb_usage(self, command_word: str, token, sentence_obj, context: Dict[str, Any]) -> bool:
        """
        Surgical check: Is this command word being used as a legitimate verb?
        Only returns True for genuine verb usage, not command misuse.
        Uses YAML-based configuration for legitimate patterns.
        """
        
        # === LINGUISTIC GUARD: Selective protection for common English verbs ===
        # These guards only apply to common English verbs that happen to be command names.
        # Git-specific commands (fork, clone, commit, push, etc.) should ALWAYS be flagged as verbs.
        
        # List of commands that are ONLY technical (not common English verbs)
        git_specific_commands = {'fork', 'clone', 'commit', 'push', 'pull', 'merge', 'rebase', 
                                'checkout', 'stash', 'fetch', 'revert', 'cherry-pick'}
        sql_specific_commands = {'select', 'insert', 'update', 'delete'}
        unix_specific_commands = {'grep', 'awk', 'sed', 'chmod', 'chown', 'tar'}
        
        technical_only_commands = git_specific_commands | sql_specific_commands | unix_specific_commands
        
        # If this is a technical-only command, DO NOT apply the protective guards
        # These should ALWAYS be flagged when used as verbs per IBM Style Guide
        is_technical_command = command_word.lower() in technical_only_commands
        
        # Only apply protective guards for common English verbs (find, sort, run, etc.)
        if not is_technical_command:
            # GUARD 0: Check for gerunds in parallel lists (e.g., "creating, updating, and deploying")
            if token.tag_ == 'VBG':
                is_in_coordination = (
                    token.dep_ == 'conj' or
                    any(child.dep_ == 'conj' for child in token.children)
                )
                if is_in_coordination:
                    return True

            # GUARD A: Check for modal auxiliary verbs (e.g., "will find", "can run")
            if any(child.dep_ == 'aux' and child.pos_ == 'AUX' for child in token.children):
                return True

            # GUARD B: Check the subject of the verb for common verbs
            subject = None
            for child in token.children:
                if child.dep_ == 'nsubj':
                    subject = child
                    break
            
            if subject and subject.lemma_.lower() in ['you', 'we', 'they', 'i', 'he', 'she']:
                return True  # Example: "You run the script.", "We find the results."

        # ... existing YAML-based logic follows ...
        sent_text = sentence_obj.text.lower()
        
        # Get pattern config for this command word
        command_patterns = self.config_service.get_patterns()
        pattern_config = None
        for pattern_id, pconfig in command_patterns.items():
            if pconfig.pattern == command_word:
                pattern_config = pconfig
                break
        
        if not pattern_config:
            return False  # Unknown command, conservative approach
        
        # Check against legitimate patterns from YAML
        if pattern_config.legitimate_patterns:
            for legitimate_pattern in pattern_config.legitimate_patterns:
                if legitimate_pattern.lower() in sent_text:
                    return True
        
        # Use YAML-based context analysis for evidence adjustment
        tech_context = TechnicalContext(
            content_type=context.get('content_type', ''),
            audience=context.get('audience', ''),
            domain=context.get('domain', ''),
            block_type=context.get('block_type', '')
        )
        
        # Calculate context-adjusted evidence
        adjusted_evidence = self.config_service.calculate_context_evidence(
            pattern_config.evidence, tech_context
        )
        
        # If context adjustments significantly reduce evidence, likely legitimate usage
        if adjusted_evidence < 0.3:
            return True
        
        return False  # Conservative: flag if uncertain
    
    def _is_in_quoted_command_example(self, token, sentence_obj, context: Dict[str, Any]) -> bool:
        """
        Surgical check: Is this command word in a quoted example or code block?
        Only returns True for genuine quoted examples, not style violations.
        """
        sent_text = sentence_obj.text
        
        # Check for various quote patterns around the token
        quote_chars = ['"', "'", '`', '"', '"', ''', ''']
        
        # Look for the token within quotes
        for quote_char in quote_chars:
            if quote_char in sent_text:
                # Find quote pairs and check if token is within them
                quote_positions = [i for i, c in enumerate(sent_text) if c == quote_char]
                if len(quote_positions) >= 2:
                    token_start = token.idx - sentence_obj.start_char
                    for i in range(0, len(quote_positions) - 1, 2):
                        if quote_positions[i] < token_start < quote_positions[i + 1]:
                            return True
        
        # Check for code block indicators
        code_indicators = ['```', '`', '<code>', '</code>', '<pre>', '</pre>']
        if any(indicator in sent_text for indicator in code_indicators):
            return True
        
        return False
    
    def _has_proper_command_syntax(self, command_word: str, sentence_obj, context: Dict[str, Any]) -> bool:
        """
        Surgical check: Does the sentence already use proper command syntax?
        Only returns True when proper syntax is genuinely present.
        """
        sent_text = sentence_obj.text.lower()
        
        # Proper command syntax patterns
        proper_syntax_patterns = [
            f'use the {command_word} command',
            f'run the {command_word} command',
            f'execute the {command_word} command',
            f'the {command_word} command',
            f'{command_word} command',
            f'`{command_word}`',  # Inline code
            f'"{command_word}"',  # Quoted command
            f"'{command_word}'",  # Single quoted command
        ]
        
        # Check if proper syntax is already used
        for pattern in proper_syntax_patterns:
            if pattern in sent_text:
                return True
        
        # Check for command syntax indicators (only if they relate to this specific command)
        command_syntax_indicators = [
            f'run {command_word}', f'execute {command_word}', f'use {command_word}',
            'command line', 'command syntax', 'command usage'
        ]
        
        if any(indicator in sent_text for indicator in command_syntax_indicators):
            return True
        
        return False
    
    def _is_non_technical_verb_context(self, command_word: str, sentence_obj, context: Dict[str, Any]) -> bool:
        """
        Surgical check: Is this command word used in a non-technical context?
        Only returns True for genuine non-technical usage.
        """
        sent_text = sentence_obj.text.lower()
        content_type = context.get('content_type', '')
        domain = context.get('domain', '')
        
        # Non-technical content types
        if content_type in ['marketing', 'narrative', 'general', 'business']:
            # Common verbs that are acceptable in non-technical contexts
            non_technical_verbs = ['load', 'save', 'run', 'stop', 'start', 'update', 'remove', 'restore']
            if command_word in non_technical_verbs:
                return True
        
        # Non-technical domains
        if domain in ['business', 'marketing', 'general', 'finance', 'healthcare']:
            non_technical_verbs = ['load', 'save', 'run', 'stop', 'start', 'update', 'remove', 'restore']
            if command_word in non_technical_verbs:
                return True
        
        # Check for non-technical context indicators
        non_technical_indicators = [
            'business', 'company', 'organization', 'team', 'meeting',
            'project', 'client', 'customer', 'user experience', 'human resources'
        ]
        
        if any(indicator in sent_text for indicator in non_technical_indicators):
            # Only for common verbs, not technical commands
            common_verbs = ['save', 'run', 'start', 'stop', 'update', 'load']
            if command_word in common_verbs:
                return True
        
        return False
    
    # === CLUE METHODS ===

    def _apply_command_linguistic_clues(self, evidence_score: float, issue: Dict[str, Any], sentence_obj) -> float:
        """Apply linguistic clues specific to command analysis."""
        sent_text = sentence_obj.text
        command_word = issue.get('command_word', '')
        
        # Check for imperative mood (command-like usage)
        if sent_text.strip().startswith(command_word.capitalize()):
            evidence_score += 0.2  # Imperative mood suggests command usage
        
        # Check for direct object that suggests technical usage
        technical_objects = [
            'file', 'data', 'database', 'table', 'record', 'system',
            'application', 'program', 'script', 'module', 'package'
        ]
        
        if any(obj in sent_text.lower() for obj in technical_objects):
            evidence_score += 0.1  # Technical objects suggest command context
        
        # Check for lack of proper command formatting
        if '`' not in sent_text and '"' not in sent_text and "'" not in sent_text:
            evidence_score += 0.1  # No code formatting suggests misuse
        
        return evidence_score

    def _apply_command_semantic_clues(self, evidence_score: float, issue: Dict[str, Any], text: str, context: Dict[str, Any]) -> float:
        """Apply semantic clues specific to command usage."""
        content_type = context.get('content_type', 'general')
        domain = context.get('domain', 'general')
        audience = context.get('audience', 'general')
        command_word = issue.get('command_word', '')
        
        # Stricter standards for technical documentation
        if content_type in ['technical', 'api', 'developer', 'tutorial']:
            evidence_score += 0.2  # Technical docs should use proper command syntax
        elif content_type in ['procedural', 'guide']:
            evidence_score += 0.1  # Procedural docs benefit from proper syntax
        
        # Domain-specific adjustments
        if domain in ['software', 'programming', 'devops', 'engineering']:
            evidence_score += 0.15  # Technical domains require proper command syntax
        elif domain in ['database', 'system_administration']:
            evidence_score += 0.2  # System domains are very strict about command usage
        
        # Audience-specific adjustments
        if audience in ['developer', 'engineer', 'system_administrator']:
            evidence_score += 0.1  # Technical audiences expect proper syntax
        elif audience in ['beginner', 'general']:
            evidence_score += 0.15  # Beginners need clear command examples
        
        # Command-specific adjustments
        highly_technical_commands = ['import', 'export', 'install', 'uninstall', 'configure', 'deploy']
        if command_word in highly_technical_commands:
            evidence_score += 0.1  # These are clearly technical commands
        
        return evidence_score
    
    # === COMMAND-SPECIFIC MESSAGING ===
    
    def _generate_command_specific_message(self, issue: Dict[str, Any], evidence_score: float) -> str:
        """
        Generate IBM Style Guide compliant message for command-as-verb violations.
        
        IBM Style Guide: "Do not use a command name as a verb."
        This rule detects when commands are used as verbs and suggests rephrasing.
        """
        command_word = issue.get('command_word', '')
        flagged_text = issue.get('text', '')
        
        if evidence_score > 0.8:
            # High confidence - direct guidance
            return f"Avoid using command name '{command_word}' as a verb. IBM Style Guide: 'Do not use a command name as a verb.' Rephrase to use '{command_word}' as a noun instead."
        elif evidence_score > 0.5:
            # Medium confidence - balanced suggestion
            return f"Consider rephrasing to avoid using '{command_word}' as a verb. Use the command name as a noun for clearer technical writing."
        else:
            # Low confidence - gentle suggestion
            return f"'{flagged_text}' appears to use a command name as a verb. Consider rephrasing if this refers to the command."
    
    def _generate_command_specific_suggestions(self, issue: Dict[str, Any], evidence_score: float, context: Dict[str, Any]) -> List[str]:
        """
        Generate IBM Style Guide compliant suggestions for command-as-verb violations.
        
        Provides specific rephrasing examples that use the command as a noun.
        """
        command_word = issue.get('command_word', '')
        flagged_text = issue.get('text', '')
        suggestions = []
        
        # Mapping of common command verbs to noun-based alternatives
        command_rephrasings = {
            'clone': {
                'verb_examples': ['cloned', 'cloning', 'clone'],
                'noun_alternatives': [
                    f"Instead of 'you have {flagged_text}', use 'you have created a clone of'",
                    f"Instead of '{flagged_text} the repository', use 'create a clone of the repository'",
                    "Use 'Use the clone command to...' or 'Create a clone by...'",
                    "Example: 'You have already created a clone of the repository' (not 'cloned')"
                ]
            },
            'fork': {
                'verb_examples': ['forked', 'forking', 'fork'],
                'noun_alternatives': [
                    f"Instead of 'you have {flagged_text}', use 'you have created a fork of'",
                    f"Instead of '{flagged_text} the repository', use 'create a fork of the repository'",
                    "Example: 'You have already created a fork of the repository' (not 'forked')"
                ]
            },
            'commit': {
                'verb_examples': ['committed', 'committing', 'commit'],
                'noun_alternatives': [
                    f"Instead of '{flagged_text} the changes', use 'create a commit with the changes'",
                    f"Instead of 'you {flagged_text}', use 'you created a commit' or 'use the commit command'",
                    "Example: 'Use the commit command to save your changes'"
                ]
            },
            'push': {
                'verb_examples': ['pushed', 'pushing', 'push'],
                'noun_alternatives': [
                    f"Instead of '{flagged_text} the changes', use 'use the push command for the changes'",
                    f"Instead of 'you {flagged_text}', use 'you used the push command'",
                    "Example: 'Use the push command to upload your commits'"
                ]
            },
            'pull': {
                'verb_examples': ['pulled', 'pulling', 'pull'],
                'noun_alternatives': [
                    f"Instead of '{flagged_text} the changes', use 'use the pull command to get the changes'",
                    "Example: 'Use the pull command to update your local repository'"
                ]
            },
            'merge': {
                'verb_examples': ['merged', 'merging', 'merge'],
                'noun_alternatives': [
                    f"Instead of '{flagged_text} the branches', use 'perform a merge of the branches'",
                    "Example: 'Perform a merge using the merge command'"
                ]
            },
            'deploy': {
                'verb_examples': ['deployed', 'deploying', 'deploy'],
                'noun_alternatives': [
                    f"Instead of '{flagged_text} the application', use 'create a deployment of the application'",
                    "Example: 'Create a deployment using the deploy command'"
                ]
            }
        }
        
        # Get rephrasing suggestions for this specific command
        if command_word in command_rephrasings:
            alternatives = command_rephrasings[command_word]['noun_alternatives']
            suggestions.extend(alternatives[:3])  # Limit to 3 most relevant
        else:
            # Generic suggestions for commands not in the mapping
            suggestions.append(f"Rephrase to use '{command_word}' as a noun, not a verb")
            suggestions.append(f"Example: 'Use the {command_word} command' instead of '{flagged_text}'")
            suggestions.append(f"Example: 'Create a {command_word}' instead of '{flagged_text}'")
        
        # Add context-specific guidance
        if evidence_score > 0.7:
            suggestions.append("IBM Style Guide: Command names should be nouns, not verbs, in documentation")
        
        return suggestions[:5]  # Limit to 5 suggestions total
