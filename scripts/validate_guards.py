#!/usr/bin/env python3
"""
Guard Validation Tool

Enforces guard implementation standards:
1. Max 5 guards per rule
2. Each guard has required documentation
3. Each guard has associated tests

Usage:
    python scripts/validate_guards.py
    python scripts/validate_guards.py rules/language_and_grammar/pronouns_rule.py
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Guard:
    """Represents a single guard in a rule"""
    number: int
    description: str
    test_reference: str
    reason: str
    line_number: int
    
    def is_valid(self) -> Tuple[bool, List[str]]:
        """Validates guard has all required documentation"""
        issues = []
        
        if not self.description:
            issues.append(f"Guard {self.number}: Missing description")
        
        if not self.test_reference:
            issues.append(f"Guard {self.number}: Missing test reference")
        elif not self.test_reference.startswith('test_'):
            issues.append(f"Guard {self.number}: Test reference must start with 'test_'")
        
        if not self.reason:
            issues.append(f"Guard {self.number}: Missing reason")
        
        # Check for subjective language
        subjective_words = ['usually', 'often', 'sometimes', 'generally', 
                           'typically', 'mostly', 'probably']
        reason_lower = self.reason.lower()
        found_subjective = [w for w in subjective_words if w in reason_lower]
        if found_subjective:
            issues.append(
                f"Guard {self.number}: Reason contains subjective language: "
                f"{', '.join(found_subjective)}. Guards must be objective."
            )
        
        return len(issues) == 0, issues


@dataclass
class RuleGuardAnalysis:
    """Analysis results for a single rule file"""
    file_path: Path
    guard_count: int
    guards: List[Guard]
    issues: List[str]
    
    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0 and self.guard_count <= 5
    
    @property
    def has_guards(self) -> bool:
        return self.guard_count > 0


def extract_guards_from_file(file_path: Path) -> RuleGuardAnalysis:
    """
    Parses a rule file and extracts guard information
    
    Expected format:
    # === ZERO FALSE POSITIVE GUARDS ===
    # Current count: N/5
    
    # GUARD 1: Description
    # Test: test_function_name()
    # Reason: Why this is objective
    if condition:
        return 0.0
    """
    content = file_path.read_text()
    lines = content.split('\n')
    
    guards = []
    issues = []
    guard_section_found = False
    current_guard = None
    current_guard_desc = None
    current_guard_test = None
    current_guard_reason = None
    current_guard_line = None
    
    # Check for guard section marker
    guard_section_pattern = r'===\s*ZERO FALSE POSITIVE GUARDS\s*==='
    declared_count = None
    
    for i, line in enumerate(lines, 1):
        if re.search(guard_section_pattern, line):
            guard_section_found = True
            continue
        
        # Look for declared count
        if guard_section_found and not declared_count:
            count_match = re.search(r'Current count:\s*(\d+)/5', line)
            if count_match:
                declared_count = int(count_match.group(1))
        
        # Look for guard markers
        guard_match = re.match(r'\s*#\s*GUARD\s+(\d+):\s*(.+)', line)
        if guard_match:
            # Save previous guard if exists
            if current_guard is not None:
                guards.append(Guard(
                    number=current_guard,
                    description=current_guard_desc or "",
                    test_reference=current_guard_test or "",
                    reason=current_guard_reason or "",
                    line_number=current_guard_line
                ))
            
            current_guard = int(guard_match.group(1))
            current_guard_desc = guard_match.group(2).strip()
            current_guard_test = None
            current_guard_reason = None
            current_guard_line = i
            continue
        
        # Look for test reference
        if current_guard is not None:
            test_match = re.match(r'\s*#\s*Test:\s*(.+)', line)
            if test_match:
                current_guard_test = test_match.group(1).strip()
                continue
            
            # Look for reason
            reason_match = re.match(r'\s*#\s*Reason:\s*(.+)', line)
            if reason_match:
                current_guard_reason = reason_match.group(1).strip()
                continue
            
            # If we hit a non-comment line, guard definition is complete
            if line.strip() and not line.strip().startswith('#'):
                if current_guard_test is None or current_guard_reason is None:
                    # Guard is incomplete, save what we have
                    pass
    
    # Save last guard
    if current_guard is not None:
        guards.append(Guard(
            number=current_guard,
            description=current_guard_desc or "",
            test_reference=current_guard_test or "",
            reason=current_guard_reason or "",
            line_number=current_guard_line
        ))
    
    # Validate guards
    for guard in guards:
        valid, guard_issues = guard.is_valid()
        if not valid:
            issues.extend(guard_issues)
    
    # Check count consistency
    actual_count = len(guards)
    if guard_section_found:
        if declared_count is None:
            issues.append("Guard section found but missing 'Current count: N/5' declaration")
        elif declared_count != actual_count:
            issues.append(
                f"Declared count ({declared_count}) doesn't match actual guards ({actual_count})"
            )
        
        if actual_count > 5:
            issues.append(f"TOO MANY GUARDS: {actual_count}/5. Maximum is 5 per rule.")
    
    # Check guard numbering
    expected_numbers = list(range(1, actual_count + 1))
    actual_numbers = sorted([g.number for g in guards])
    if actual_numbers != expected_numbers:
        issues.append(
            f"Guard numbering is incorrect. Expected {expected_numbers}, got {actual_numbers}"
        )
    
    return RuleGuardAnalysis(
        file_path=file_path,
        guard_count=actual_count,
        guards=guards,
        issues=issues
    )


def check_guard_tests_exist(analysis: RuleGuardAnalysis) -> List[str]:
    """Verifies that test files exist for each guard"""
    issues = []
    
    rule_name = analysis.file_path.stem  # e.g., "pronouns_rule"
    test_file_guards = analysis.file_path.parent.parent.parent / "tests" / "guards" / f"test_guards_{rule_name}.py"
    test_file_root = analysis.file_path.parent.parent.parent / "tests" / f"test_guards_{rule_name}.py"
    
    if not analysis.guards:
        return issues
    
    # Check both locations
    if test_file_guards.exists():
        test_file = test_file_guards
    elif test_file_root.exists():
        test_file = test_file_root
    else:
        issues.append(f"Missing test file: {test_file_guards} or {test_file_root}")
        return issues
    
    # Check if tests are referenced in test file
    test_content = test_file.read_text()
    for guard in analysis.guards:
        test_func_name = guard.test_reference.replace('()', '')
        if test_func_name not in test_content:
            issues.append(
                f"Guard {guard.number} references test '{guard.test_reference}' "
                f"but it's not found in {test_file.name}"
            )
    
    return issues


def validate_rule_file(file_path: Path, check_tests: bool = True) -> RuleGuardAnalysis:
    """Validates a single rule file"""
    analysis = extract_guards_from_file(file_path)
    
    if check_tests and analysis.guards:
        test_issues = check_guard_tests_exist(analysis)
        analysis.issues.extend(test_issues)
    
    return analysis


def validate_all_rules(rules_dir: Path, check_tests: bool = True) -> Dict[str, RuleGuardAnalysis]:
    """Validates all rule files in the rules directory"""
    results = {}
    
    # Find all Python files in rules directory
    for rule_file in rules_dir.rglob("*_rule.py"):
        if rule_file.name.startswith('base_'):
            continue
        
        analysis = validate_rule_file(rule_file, check_tests)
        if analysis.has_guards:  # Only report files with guards
            results[str(rule_file.relative_to(rules_dir))] = analysis
    
    return results


def print_report(results: Dict[str, RuleGuardAnalysis]):
    """Prints validation report"""
    print("=" * 80)
    print("GUARD VALIDATION REPORT")
    print("=" * 80)
    print()
    
    total_guards = 0
    total_issues = 0
    rules_with_guards = 0
    
    for rule_path, analysis in sorted(results.items()):
        rules_with_guards += 1
        total_guards += analysis.guard_count
        total_issues += len(analysis.issues)
        
        # Status indicator
        status = "✓" if analysis.is_valid else "✗"
        count_indicator = f"{analysis.guard_count}/5"
        if analysis.guard_count > 5:
            count_indicator = f"❌ {count_indicator}"
        elif analysis.guard_count >= 4:
            count_indicator = f"⚠️  {count_indicator}"
        
        print(f"{status} {rule_path}")
        print(f"   Guards: {count_indicator}")
        
        if analysis.issues:
            print(f"   Issues:")
            for issue in analysis.issues:
                print(f"     - {issue}")
        
        if analysis.guards and not analysis.issues:
            print(f"   Guards:")
            for guard in analysis.guards:
                print(f"     {guard.number}. {guard.description}")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Rules with guards: {rules_with_guards}")
    print(f"Total guards: {total_guards}")
    print(f"Total issues: {total_issues}")
    print()
    
    if total_issues == 0:
        print("✓ All guards are properly documented and validated!")
        return 0
    else:
        print(f"✗ Found {total_issues} issue(s) that need attention.")
        return 1


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Validate specific file
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return 1
        
        analysis = validate_rule_file(file_path)
        results = {file_path.name: analysis}
        return print_report(results)
    else:
        # Validate all rules
        project_root = Path(__file__).parent.parent
        rules_dir = project_root / "rules"
        
        if not rules_dir.exists():
            print(f"Error: Rules directory not found: {rules_dir}")
            return 1
        
        results = validate_all_rules(rules_dir)
        return print_report(results)


if __name__ == "__main__":
    sys.exit(main())

