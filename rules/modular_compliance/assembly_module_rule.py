"""Assembly Module Rule - validates assembly modules per Red Hat modular documentation guidelines."""
import re
import os
from typing import List, Optional, Dict, Any
from rules.base_rule import BaseRule
from .modular_structure_bridge import ModularStructureBridge
try:
    import yaml
except ImportError:
    yaml = None


class AssemblyModuleRule(BaseRule):
    
    def __init__(self):
        super().__init__()
        self.rule_type = "assembly_module"
        self.rule_subtype = "assembly_module"
        self.parser = ModularStructureBridge()
        self._load_config()
    
    def _get_rule_type(self) -> str:
        return "assembly_module"
    
    def _detect_content_type_from_metadata(self, text: str) -> Optional[str]:
        """
        Detect content type from document metadata attributes.
        
        Looks for:
        - :_mod-docs-content-type: PROCEDURE|CONCEPT|REFERENCE|ASSEMBLY
        - :_content-type: (deprecated but still supported)
        
        Returns lowercase type or None if not found.
        """
        # Check for new-style attribute
        new_style = re.search(r':_mod-docs-content-type:\s*(PROCEDURE|CONCEPT|REFERENCE|ASSEMBLY)', text, re.IGNORECASE)
        if new_style:
            return new_style.group(1).lower()
        
        # Check for old-style attribute (deprecated but still supported)
        old_style = re.search(r':_content-type:\s*(PROCEDURE|CONCEPT|REFERENCE|ASSEMBLY)', text, re.IGNORECASE)
        if old_style:
            return old_style.group(1).lower()
        
        return None
        
    def _load_config(self):
        config_path = os.path.join(
            os.path.dirname(__file__), 
            'config', 
            'modular_compliance_types.yaml'
        )
        
        if yaml and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    self.thresholds = config.get('thresholds', {})
                    self.thresholds.setdefault('concise_introduction_words', 100)
                    self.thresholds.setdefault('max_nesting_depth', 3)
            except Exception:
                self._set_fallback_config()
        else:
            self._set_fallback_config()
    
    def _set_fallback_config(self):
        self.thresholds = {
            'concise_introduction_words': 100,
            'max_nesting_depth': 3
        }
    
    def analyze(self, text: str, sentences: List[str] = None, nlp=None, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        # === UNIVERSAL CODE CONTEXT GUARD ===
        # Skip analysis for code blocks, listings, and literal blocks (technical syntax, not prose)
        if context and context.get('block_type') in ['listing', 'literal', 'code_block', 'inline_code']:
            return []
        errors = []
        context = context or {}
        
        # Detect content type from metadata if not provided
        detected_type = self._detect_content_type_from_metadata(text)
        content_type = context.get('content_type', detected_type)
        
        # Only analyze if this is an assembly module
        if content_type and content_type != 'assembly':
            return errors
            
        structure = self.parser.parse(text)
        compliance_issues = []
        
        compliance_issues.extend(self._validate_metadata(text))
        compliance_issues.extend(self._find_introduction_issues(structure, text))
        compliance_issues.extend(self._validate_include_directives(text))
        compliance_issues.extend(self._check_nesting_depth(text))
        compliance_issues.extend(self._validate_heading_case(structure))
        compliance_issues.extend(self._check_additional_resources_conflict(text))
        
        for issue in compliance_issues:
            error = self._create_error(
                sentence=issue.get('sentence', issue.get('flagged_text', '')),
                sentence_index=issue.get('line_number', 0),
                message=issue.get('message', ''),
                suggestions=issue.get('suggestions', []),
                severity=self._map_compliance_level_to_severity(issue.get('level')),
                text=text,
                context=context,
                flagged_text=issue.get('flagged_text', ''),
                span=issue.get('span', (0, 0))
            )
            errors.append(error)
        
        return errors
    
    def _map_compliance_level_to_severity(self, level: str) -> str:
        mapping = {
            'FAIL': 'high',
            'WARN': 'medium',
            'INFO': 'low'
        }
        return mapping.get(level, 'medium')
    
    def _validate_metadata(self, text: str) -> List[Dict[str, Any]]:
        issues = []
        
        has_new_attr = ':_mod-docs-content-type:' in text
        has_old_attr = ':_content-type:' in text and not has_new_attr
        has_deprecated_role = '[role="_abstract"]' in text
        
        if has_old_attr:
            issues.append({
                'type': 'metadata_format',
                'level': 'WARN',
                'message': "Using deprecated content type attribute ':_content-type:'",
                'flagged_text': ':_content-type:',
                'line_number': 0,
                'span': (0, 0),
                'suggestions': [
                    "Change ':_content-type:' to ':_mod-docs-content-type:' per Red Hat Aug 2023",
                    "Update: :_mod-docs-content-type: ASSEMBLY"
                ]
            })
        
        if has_deprecated_role:
            issues.append({
                'type': 'metadata_format',
                'level': 'WARN',
                'message': "Using deprecated [role=\"_abstract\"] tag",
                'flagged_text': '[role="_abstract"]',
                'line_number': 0,
                'span': (0, 0),
                'suggestions': [
                    "Remove [role=\"_abstract\"] tag per Red Hat Jan 2024",
                    "The role tag is no longer needed"
                ]
            })
        
        return issues
    
    def _find_introduction_issues(self, structure: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
        """
        Validate introduction per Red Hat guidelines for assemblies.
        
        Per Red Hat: Assemblies require a short introduction explaining the user story.
        If parser can't extract intro_paragraphs but document has content, assume it's present.
        """
        issues = []
        intro_paragraphs = structure.get('introduction_paragraphs', [])
        
        # If parser didn't extract intro, check if document has content
        if not intro_paragraphs:
            word_count = structure.get('word_count', 0)
            has_content = structure.get('has_content', False)
            
            # If document has meaningful content (>15 words), assume intro is present
            if has_content and word_count >= 15:
                return issues
        
        if not intro_paragraphs:
            issues.append({
                'type': 'critical_structural',
                'level': 'FAIL',
                'message': "Assembly lacks introduction paragraph providing context for the user story",
                'flagged_text': "Missing introduction",
                'line_number': 2,
                'span': (0, 0),
                'suggestions': [
                    "Add introductory paragraph explaining the user story this assembly addresses",
                    "Describe what the user will accomplish by following this assembly"
                ]
            })
        elif len(intro_paragraphs) > 1:
            issues.append({
                'type': 'improvement_suggestion',
                'level': 'WARN',
                'message': f"Introduction has {len(intro_paragraphs)} paragraphs, should be single paragraph",
                'flagged_text': "Multi-paragraph introduction",
                'line_number': 2,
                'span': (0, 0),
                'suggestions': [
                    "Combine into a single, concise paragraph",
                    "Move detailed context to the assembly body"
                ]
            })
        else:
            intro_text = intro_paragraphs[0]
            intro_word_count = len(intro_text.split())
            max_words = self.thresholds.get('concise_introduction_words', 100)
            
            if intro_word_count > max_words:
                issues.append({
                    'type': 'improvement_suggestion',
                    'level': 'INFO',
                    'message': f"Introduction is {intro_word_count} words (recommended: under {max_words})",
                    'flagged_text': intro_text[:100] + "...",
                    'line_number': 2,
                    'span': (0, 0),
                    'suggestions': [
                        "Shorten introduction to be more concise",
                        "Move detailed information to the assembly body"
                    ]
                })
        
        return issues
    
    def _validate_include_directives(self, text: str) -> List[Dict[str, Any]]:
        issues = []
        
        include_pattern = r'include::([^[\]]+\.adoc)'
        includes = re.findall(include_pattern, text, re.IGNORECASE)
        
        if not includes:
            issues.append({
                'type': 'critical_structural',
                'level': 'FAIL',
                'message': "Assembly contains no include directives for modules",
                'flagged_text': "No include directives",
                'line_number': 0,
                'span': (0, 0),
                'suggestions': [
                    "Add include directives: include::module-name.adoc[]",
                    "Assemblies should include modules (concept, procedure, reference)",
                    "Can also include other assemblies"
                ]
            })
        
        return issues
    
    def _validate_approved_sections(self, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        issues = []
        
        approved_sections = {
            'context', 'prerequisites', 'additional resources', 
            'related information', 'next steps'
        }
        
        for section in structure.get('sections', []):
            if section.get('level', 0) == 0:
                continue
            
            section_title_lower = section['title'].lower().strip()
            
            if not any(approved in section_title_lower for approved in approved_sections):
                issues.append({
                    'type': 'section_validation',
                    'level': 'INFO',
                    'message': f"Non-standard section in assembly: \"{section['title']}\"",
                    'flagged_text': section['title'],
                    'line_number': section.get('line_number', 0),
                    'span': section.get('span', (0, 0)),
                    'suggestions': [
                        f"Consider using standard sections: {', '.join(approved_sections)}",
                        "Ensure section adds value to the user story"
                    ]
                })
        
        return issues
    
    def _check_nesting_depth(self, text: str) -> List[Dict[str, Any]]:
        issues = []
        
        assembly_includes = re.findall(r'include::(assembly_[^[\]]+\.adoc)', text, re.IGNORECASE)
        
        if len(assembly_includes) > self.thresholds.get('max_nesting_depth', 3):
            issues.append({
                'type': 'structural_warning',
                'level': 'WARN',
                'message': f"Assembly includes {len(assembly_includes)} other assemblies (nesting may be too deep)",
                'flagged_text': f"{len(assembly_includes)} nested assemblies",
                'line_number': 0,
                'span': (0, 0),
                'suggestions': [
                    "Per Red Hat: Deep nesting creates complexity",
                    "Consider linking to assemblies instead of including them",
                    f"Limit to {self.thresholds.get('max_nesting_depth', 3)} levels of nesting"
                ]
            })
        
        return issues
    
    def _validate_heading_case(self, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        issues = []
        
        for section in structure.get('sections', []):
            heading = section.get('title', '')
            if not heading or section.get('level', 0) == 0:
                continue
            
            words = heading.split()
            if len(words) < 2:
                continue
            
            capitalized_count = 0
            short_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
            
            for i, word in enumerate(words[1:], start=1):
                clean_word = word.strip('.,!?:;()[]{}')
                if not clean_word:
                    continue
                
                if clean_word[0].isupper() and clean_word.lower() not in short_words:
                    if not (clean_word.isupper() or self._is_proper_noun(clean_word)):
                        capitalized_count += 1
            
            if capitalized_count >= 2:
                issues.append({
                    'type': 'heading_format',
                    'level': 'WARN',
                    'message': f'Heading uses title case instead of sentence case: "{heading}"',
                    'flagged_text': heading,
                    'line_number': section.get('line_number', 0),
                    'span': section.get('span', (0, 0)),
                    'suggestions': [
                        f'Convert to sentence case: "{self._to_sentence_case(heading)}"',
                        "Per Red Hat 2022: Use sentence case for headings"
                    ]
                })
        
        return issues
    
    def _check_additional_resources_conflict(self, text: str) -> List[Dict[str, Any]]:
        issues = []
        
        has_assembly_resources = bool(re.search(r'^==\s+Additional\s+resources', text, re.MULTILINE | re.IGNORECASE))
        
        if has_assembly_resources:
            last_include = None
            for match in re.finditer(r'include::([^[\]]+\.adoc)', text, re.IGNORECASE):
                last_include = match.group(1)
            
            if last_include:
                issues.append({
                    'type': 'structural_warning',
                    'level': 'INFO',
                    'message': "Assembly has 'Additional resources' section - check if last included module also has one",
                    'flagged_text': "Additional resources",
                    'line_number': 0,
                    'span': (0, 0),
                    'suggestions': [
                        "Per Red Hat Sept 2023: If both assembly and last module have 'Additional resources', combine them",
                        f"Check if {last_include} has 'Additional resources' section"
                    ]
                })
        
        return issues
    
    def _is_proper_noun(self, word: str) -> bool:
        proper_nouns = {
            'red', 'hat', 'linux', 'kubernetes', 'openshift', 'ansible', 'docker',
            'python', 'java', 'javascript', 'api', 'rest', 'http', 'https', 'ssh'
        }
        return word.lower() in proper_nouns
    
    def _to_sentence_case(self, heading: str) -> str:
        words = heading.split()
        if not words:
            return heading
        
        result = [words[0]]
        short_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        
        for word in words[1:]:
            clean_word = word.strip('.,!?:;()[]{}')
            if clean_word.isupper() or self._is_proper_noun(clean_word):
                result.append(word)
            elif clean_word.lower() in short_words:
                result.append(clean_word.lower())
            else:
                result.append(clean_word.lower())
        
        return ' '.join(result)

