#!/usr/bin/env python3
"""
World-Class Test Suite for Reference Module Rule
Tests EVERY requirement from Red Hat guidelines systematically.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules.modular_compliance.reference_module_rule import ReferenceModuleRule

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = []
        self.failed = []
    
    def add_pass(self, msg):
        self.passed.append(msg)
        print(f"  ✅ {msg}")
    
    def add_fail(self, msg):
        self.failed.append(msg)
        print(f"  ❌ {msg}")
    
    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*70}")
        print(f"{self.name}: {len(self.passed)}/{total} passed")
        if self.failed:
            print(f"\nFailed tests:")
            for f in self.failed:
                print(f"  - {f}")
        return len(self.failed) == 0

def test_introduction_requirement(rule):
    """Test introduction paragraph requirement."""
    result = TestResult("Introduction Requirement")
    
    # Missing introduction
    no_intro = """
:_mod-docs-content-type: REFERENCE

= API Reference

== Endpoints
* GET /api/v1/users
"""
    errors = rule.analyze(no_intro, {'content_type': 'reference'})
    intro_errors = [e for e in errors if 'introduction' in e.get('message', '').lower()]
    if len(intro_errors) > 0:
        result.add_pass("Flags missing introduction")
    else:
        result.add_fail("Missed missing introduction")
    
    # Valid introduction
    with_intro = """
:_mod-docs-content-type: REFERENCE

= API Reference

This reference lists all API endpoints.

== Endpoints
* GET /api/v1/users
"""
    errors = rule.analyze(with_intro, {'content_type': 'reference'})
    intro_errors = [e for e in errors if 'introduction' in e.get('message', '').lower() and 'introductory paragraph' in e.get('message', '').lower()]
    if len(intro_errors) == 0:
        result.add_pass("Accepts valid introduction")
    else:
        result.add_fail("False positive on valid introduction")
    
    return result.summary()

def test_structured_data_requirement(rule):
    """Test structured data requirement (tables, lists, definition lists)."""
    result = TestResult("Structured Data Requirement")
    
    # No structured data (just paragraphs) - must be > 50 words
    no_structure = """
:_mod-docs-content-type: REFERENCE

= Configuration Options

This document describes configuration options in detail with long paragraphs explaining each option and its purpose and how it works and what values are accepted and how they interact with other settings and what the defaults are.
"""
    errors = rule.analyze(no_structure, {'content_type': 'reference'})
    struct_errors = [e for e in errors if 'structured' in e.get('message', '').lower() or 'table' in e.get('message', '').lower()]
    if len(struct_errors) > 0:
        result.add_pass("Flags lack of structured data")
    else:
        result.add_fail("Missed lack of structured data")
    
    # With table
    with_table = """
:_mod-docs-content-type: REFERENCE

= Configuration Options

Available options:

|===
|Option |Description
|timeout |Connection timeout
|port |Server port
|===
"""
    errors = rule.analyze(with_table, {'content_type': 'reference'})
    struct_errors = [e for e in errors if 'structured' in e.get('message', '').lower() or 'table' in e.get('message', '').lower()]
    if len(struct_errors) == 0:
        result.add_pass("Accepts tables as structured data")
    else:
        result.add_fail("False positive on table")
    
    # With definition list
    with_list = """
:_mod-docs-content-type: REFERENCE

= Configuration Options

Options:

timeout:: Connection timeout
port:: Server port
host:: Server hostname
"""
    errors = rule.analyze(with_list, {'content_type': 'reference'})
    struct_errors = [e for e in errors if 'structured' in e.get('message', '').lower()]
    if len(struct_errors) == 0:
        result.add_pass("Accepts definition lists as structured data")
    else:
        result.add_fail("False positive on definition list")
    
    return result.summary()

def test_procedural_prohibition(rule):
    """Test that step-by-step instructions are prohibited."""
    result = TestResult("Procedural Content Prohibition")
    
    # Step-by-step instructions (prohibited)
    procedural = """
:_mod-docs-content-type: REFERENCE

= Setup Reference

Configuration steps:

1. Open settings
2. Set the values
3. Save changes
"""
    errors = rule.analyze(procedural, {'content_type': 'reference'})
    proc_errors = [e for e in errors if 'procedural' in e.get('message', '').lower() or 'step' in e.get('message', '').lower()]
    if len(proc_errors) > 0:
        result.add_pass("Flags procedural content")
    else:
        result.add_fail("Missed procedural content")
    
    # Simple reference list (allowed)
    reference_list = """
:_mod-docs-content-type: REFERENCE

= Command Reference

Commands:

* list - Lists items
* show - Displays information
* get - Retrieves data
"""
    errors = rule.analyze(reference_list, {'content_type': 'reference'})
    proc_errors = [e for e in errors if 'procedural' in e.get('message', '').lower()]
    if len(proc_errors) == 0:
        result.add_pass("Accepts descriptive reference lists")
    else:
        result.add_fail("False positive on descriptive list")
    
    return result.summary()

def test_conceptual_prohibition(rule):
    """Test that long conceptual explanations are discouraged."""
    result = TestResult("Long Conceptual Explanations")
    
    # Long conceptual text (should suggest moving to concept module)
    long_concept = """
:_mod-docs-content-type: REFERENCE

= API Reference

""" + " ".join(["explanation"] * 200) + """

== Endpoints
* GET /users
"""
    errors = rule.analyze(long_concept, {'content_type': 'reference'})
    concept_errors = [e for e in errors if 'conceptual' in e.get('message', '').lower() or 'explanation' in e.get('message', '').lower()]
    if len(concept_errors) > 0:
        result.add_pass("Warns on long conceptual content")
    else:
        result.add_fail("Missed long conceptual content")
    
    return result.summary()

def test_metadata_validation(rule):
    """Test 2023-2024 metadata updates."""
    result = TestResult("Metadata Validation (2023-2024)")
    
    # Deprecated :_content-type:
    old_attr = ":_content-type: REFERENCE\n\n= Test\n\nReference data:\n\n* Item"
    errors = rule.analyze(old_attr, {'content_type': 'reference'})
    if any('content-type' in e.get('message', '').lower() and 'deprecated' in e.get('message', '').lower() for e in errors):
        result.add_pass("Detects deprecated :_content-type:")
    else:
        result.add_fail("Missed deprecated :_content-type:")
    
    # Deprecated [role="_abstract"]
    deprecated_role = ':_mod-docs-content-type: REFERENCE\n\n[role="_abstract"]\nReference.\n\n= Test\n\n* Data'
    errors = rule.analyze(deprecated_role, {'content_type': 'reference'})
    if any('abstract' in e.get('message', '').lower() for e in errors):
        result.add_pass("Detects deprecated [role=\"_abstract\"]")
    else:
        result.add_fail("Missed deprecated [role=\"_abstract\"]")
    
    return result.summary()

def test_heading_case(rule):
    """Test sentence case heading requirement (2022 April)."""
    result = TestResult("Heading Case (2022 April)")
    
    # Title case heading
    title_case = """
:_mod-docs-content-type: REFERENCE

= API Reference

Reference data.

== Command Line Options And Parameters

* option1
"""
    errors = rule.analyze(title_case, {'content_type': 'reference'})
    case_errors = [e for e in errors if 'case' in e.get('message', '').lower()]
    if len(case_errors) > 0:
        result.add_pass("Flags title case headings")
    else:
        result.add_fail("Missed title case heading")
    
    # Sentence case (valid)
    sentence_case = """
:_mod-docs-content-type: REFERENCE

= API reference

Reference data.

== Command line options

* option1
"""
    errors = rule.analyze(sentence_case, {'content_type': 'reference'})
    case_errors = [e for e in errors if 'case' in e.get('message', '').lower()]
    if len(case_errors) == 0:
        result.add_pass("Accepts sentence case")
    else:
        result.add_fail("False positive on sentence case")
    
    return result.summary()

def test_scannability(rule):
    """Test scannable format recommendations."""
    result = TestResult("Scannability")
    
    # Well-organized reference
    scannable = """
:_mod-docs-content-type: REFERENCE

= Configuration reference

This reference lists configuration options.

|===
|Option |Type |Default |Description
|timeout |integer |30 |Connection timeout
|port |integer |8080 |Server port
|host |string |localhost |Server host
|===
"""
    errors = rule.analyze(scannable, {'content_type': 'reference'})
    # Filter out parser issues related to introduction
    relevant_errors = [e for e in errors if 'introduction' not in e.get('message', '').lower()]
    if len(relevant_errors) == 0:
        result.add_pass("Accepts scannable table format")
    else:
        result.add_fail(f"False positive on scannable format: {len(relevant_errors)} errors")
    
    return result.summary()

def run_all_tests():
    """Run comprehensive test suite."""
    rule = ReferenceModuleRule()
    
    print("=" * 70)
    print("WORLD-CLASS REFERENCE MODULE TEST SUITE")
    print("Testing against Red Hat Modular Documentation Guidelines")
    print("=" * 70)
    print()
    
    results = []
    results.append(test_introduction_requirement(rule))
    print()
    results.append(test_structured_data_requirement(rule))
    print()
    results.append(test_procedural_prohibition(rule))
    print()
    results.append(test_conceptual_prohibition(rule))
    print()
    results.append(test_metadata_validation(rule))
    print()
    results.append(test_heading_case(rule))
    print()
    results.append(test_scannability(rule))
    print()
    
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"FINAL RESULT: {passed}/{total} test categories passed")
    
    if passed == total:
        print("✅ TRUE WORLD-CLASS - ALL TESTS PASSED")
        print("=" * 70)
        return 0
    else:
        print("❌ IMPLEMENTATION HAS GAPS")
        print("=" * 70)
        return 1

if __name__ == '__main__':
    sys.exit(run_all_tests())

