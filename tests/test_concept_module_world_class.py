#!/usr/bin/env python3
"""
World-Class Test Suite for Concept Module Rule
Tests EVERY requirement from Red Hat guidelines systematically.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules.modular_compliance.concept_module_rule import ConceptModuleRule

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

def test_metadata_validation(rule):
    """Test 2023-2024 Red Hat metadata updates."""
    result = TestResult("Metadata Validation (2023-2024)")
    
    # Test 1: Deprecated :_content-type: attribute
    old_attr = ":_content-type: CONCEPT\n\n= Test\nContent"
    errors = rule.analyze(old_attr, {'content_type': 'concept'})
    if any('content-type' in e.get('message', '').lower() for e in errors):
        result.add_pass("Detects deprecated :_content-type:")
    else:
        result.add_fail("Missed deprecated :_content-type:")
    
    # Test 2: New format should not be flagged
    new_attr = ":_mod-docs-content-type: CONCEPT\n\n= Test\nContent"
    errors = rule.analyze(new_attr, {'content_type': 'concept'})
    if not any('content-type' in e.get('message', '').lower() for e in errors):
        result.add_pass("Accepts :_mod-docs-content-type:")
    else:
        result.add_fail("False positive on :_mod-docs-content-type:")
    
    # Test 3: Deprecated [role="_abstract"] tag
    deprecated_role = ':_mod-docs-content-type: CONCEPT\n\n[role="_abstract"]\n= Test\nContent'
    errors = rule.analyze(deprecated_role, {'content_type': 'concept'})
    if any('abstract' in e.get('message', '').lower() for e in errors):
        result.add_pass("Detects deprecated [role=\"_abstract\"]")
    else:
        result.add_fail("Missed deprecated [role=\"_abstract\"]")
    
    return result.summary()

def test_simple_action_exception(rule):
    """Test 2021 August #150: Simple actions should be allowed."""
    result = TestResult("Simple Action Exception (2021 Aug)")
    
    # Test 1: Simple actions should be ALLOWED
    simple_actions = """
:_mod-docs-content-type: CONCEPT

= API Concepts

The API provides endpoints.

* Use the REST endpoint for queries
* View logs in the console  
* Check the health status
* Call the API with credentials
* Monitor response times
"""
    errors = rule.analyze(simple_actions, {'content_type': 'concept'})
    proc_errors = [e for e in errors if 'procedural' in e.get('message', '').lower()]
    if len(proc_errors) == 0:
        result.add_pass("Allows simple actions (Use, View, Check, Call, Monitor)")
    else:
        result.add_fail(f"False positive: Flagged {len(proc_errors)} simple actions")
    
    # Test 2: "See" references should be allowed
    see_refs = """
:_mod-docs-content-type: CONCEPT

= Documentation

For more details:

* See the API documentation
* Refer to the installation guide
* Consider the security implications
"""
    errors = rule.analyze(see_refs, {'content_type': 'concept'})
    proc_errors = [e for e in errors if 'procedural' in e.get('message', '').lower()]
    if len(proc_errors) == 0:
        result.add_pass("Allows 'See', 'Refer to', 'Consider' references")
    else:
        result.add_fail("False positive: Flagged reference phrases")
    
    # Test 3: Descriptive lists should be allowed
    descriptive = """
:_mod-docs-content-type: CONCEPT

= Authentication Types

Available methods:

* Token authentication validates API keys
* OAuth provides third-party authorization
* LDAP integrates with directory services
"""
    errors = rule.analyze(descriptive, {'content_type': 'concept'})
    proc_errors = [e for e in errors if 'procedural' in e.get('message', '').lower()]
    if len(proc_errors) == 0:
        result.add_pass("Allows descriptive lists (no imperatives)")
    else:
        result.add_fail("False positive: Flagged descriptive content")
    
    return result.summary()

def test_procedural_detection(rule):
    """Test that real procedures are caught."""
    result = TestResult("Procedural Content Detection")
    
    # Test 1: Numbered list with imperatives
    numbered_proc = """
:_mod-docs-content-type: CONCEPT

= Setup

To configure:

1. Open the settings
2. Navigate to security
3. Enable encryption  
4. Save the changes
"""
    errors = rule.analyze(numbered_proc, {'content_type': 'concept'})
    proc_errors = [e for e in errors if 'procedural' in e.get('message', '').lower()]
    if len(proc_errors) > 0:
        result.add_pass("Flags numbered lists with imperatives")
    else:
        result.add_fail("Missed numbered procedural list")
    
    # Test 2: Sequential pattern with Then/Next/Finally
    sequential_list = """
:_mod-docs-content-type: CONCEPT

= Installation

Steps:

1. Download the package
2. Then extract the files
3. Next, run the installer
4. Finally, restart the system
"""
    errors = rule.analyze(sequential_list, {'content_type': 'concept'})
    proc_errors = [e for e in errors if 'procedural' in e.get('message', '').lower()]
    if len(proc_errors) > 0:
        result.add_pass("Flags sequential patterns (Then, Next, Finally)")
    else:
        result.add_fail("Missed sequential pattern in list")
    
    # Test 3: Sequential pattern in paragraphs
    sequential_para = """
:_mod-docs-content-type: CONCEPT

= Installation

First, download the package. Then extract the files. Next, run the installer.
"""
    errors = rule.analyze(sequential_para, {'content_type': 'concept'})
    proc_errors = [e for e in errors if 'procedural' in e.get('message', '').lower()]
    if len(proc_errors) > 0:
        result.add_pass("Flags sequential patterns in paragraphs")
    else:
        result.add_fail("Missed sequential pattern in paragraphs")
    
    return result.summary()

def test_introduction_requirements(rule):
    """Test introduction paragraph requirements."""
    result = TestResult("Introduction Requirements")
    
    # Test 1: Missing introduction
    no_intro = ":_mod-docs-content-type: CONCEPT\n\n= Test"
    errors = rule.analyze(no_intro, {'content_type': 'concept'})
    intro_errors = [e for e in errors if 'introduction' in e.get('message', '').lower()]
    if len(intro_errors) > 0:
        result.add_pass("Flags missing introduction")
    else:
        result.add_fail("Missed missing introduction")
    
    # Test 2: Multi-paragraph introduction
    multi_para = """
:_mod-docs-content-type: CONCEPT

= Test

First paragraph.

Second paragraph.

== Body
Content
"""
    errors = rule.analyze(multi_para, {'content_type': 'concept'})
    multi_errors = [e for e in errors if 'paragraph' in e.get('message', '').lower()]
    if len(multi_errors) > 0:
        result.add_pass("Flags multi-paragraph introduction")
    else:
        result.add_fail("Missed multi-paragraph introduction")
    
    # Test 3: Long introduction (>100 words)
    long_intro = """
:_mod-docs-content-type: CONCEPT

= Test

""" + " ".join(["word"] * 120) + """

== Body
Content
"""
    errors = rule.analyze(long_intro, {'content_type': 'concept'})
    concise_errors = [e for e in errors if '120' in e.get('message', '')]
    if len(concise_errors) > 0:
        result.add_pass("Flags long introduction (>100 words)")
    else:
        result.add_fail("Missed long introduction")
    
    # Test 4: Valid single paragraph introduction
    valid_intro = """
:_mod-docs-content-type: CONCEPT

= Test

This is a valid single paragraph introduction under 100 words.

== Body
More content here.
"""
    errors = rule.analyze(valid_intro, {'content_type': 'concept'})
    intro_errors = [e for e in errors if 'introduction' in e.get('message', '').lower() and 'paragraph' in e.get('message', '').lower()]
    if len(intro_errors) == 0:
        result.add_pass("Accepts valid single paragraph introduction")
    else:
        result.add_fail("False positive on valid introduction")
    
    return result.summary()

def test_title_format(rule):
    """Test title format validation."""
    result = TestResult("Title Format")
    
    # Test 1: Gerund title (procedure-like) should be flagged
    gerund = """
:_mod-docs-content-type: CONCEPT

= Configuring Network Settings

This explains configuration.
"""
    errors = rule.analyze(gerund, {'content_type': 'concept'})
    title_errors = [e for e in errors if 'procedural' in e.get('message', '').lower() and 'title' in e.get('message', '').lower()]
    if len(title_errors) > 0:
        result.add_pass("Flags gerund titles (Configuring)")
    else:
        result.add_fail("Missed gerund title")
    
    # Test 2: Noun-based title should be accepted
    noun_title = """
:_mod-docs-content-type: CONCEPT

= Network Security

This explains security concepts.
"""
    errors = rule.analyze(noun_title, {'content_type': 'concept'})
    title_errors = [e for e in errors if 'title' in e.get('message', '').lower()]
    if len(title_errors) == 0:
        result.add_pass("Accepts noun-based titles")
    else:
        result.add_fail("False positive on noun-based title")
    
    return result.summary()

def test_heading_case(rule):
    """Test sentence case heading requirement (2022 April)."""
    result = TestResult("Heading Case (2022 April)")
    
    # Test 1: Title case heading should be flagged
    title_case = """
:_mod-docs-content-type: CONCEPT

= Network Concepts

This is the introduction.

== Understanding Network Security Settings

Content here.
"""
    errors = rule.analyze(title_case, {'content_type': 'concept'})
    case_errors = [e for e in errors if 'case' in e.get('message', '').lower()]
    if len(case_errors) > 0:
        result.add_pass("Flags title case headings")
    else:
        result.add_fail("Missed title case heading")
    
    # Test 2: Sentence case should be accepted
    sentence_case = """
:_mod-docs-content-type: CONCEPT

= Network concepts

This is the introduction.

== Understanding network security

Content here.
"""
    errors = rule.analyze(sentence_case, {'content_type': 'concept'})
    case_errors = [e for e in errors if 'case' in e.get('message', '').lower()]
    if len(case_errors) == 0:
        result.add_pass("Accepts sentence case headings")
    else:
        result.add_fail("False positive on sentence case")
    
    return result.summary()

def test_module_nesting(rule):
    """Test module nesting detection (2021 June)."""
    result = TestResult("Module Nesting Detection (2021 June)")
    
    # Test 1: Module includes should be flagged
    nested = """
:_mod-docs-content-type: CONCEPT

= Main Concept

include::con_sub-concept.adoc[]
"""
    errors = rule.analyze(nested, {'content_type': 'concept'})
    nest_errors = [e for e in errors if 'module' in e.get('message', '').lower() and 'include' in e.get('message', '').lower()]
    if len(nest_errors) > 0:
        result.add_pass("Flags module nesting")
    else:
        result.add_fail("Missed module nesting")
    
    # Test 2: Snippet includes should be allowed
    snippet = """
:_mod-docs-content-type: CONCEPT

= Main Concept

include::snippets/code-example.adoc[]
"""
    errors = rule.analyze(snippet, {'content_type': 'concept'})
    nest_errors = [e for e in errors if 'module' in e.get('message', '').lower() and 'include' in e.get('message', '').lower()]
    if len(nest_errors) == 0:
        result.add_pass("Allows snippet includes")
    else:
        result.add_fail("False positive on snippet include")
    
    return result.summary()

def run_all_tests():
    """Run comprehensive test suite."""
    rule = ConceptModuleRule()
    
    print("=" * 70)
    print("WORLD-CLASS CONCEPT MODULE TEST SUITE")
    print("Testing against Red Hat Modular Documentation Guidelines")
    print("=" * 70)
    print()
    
    results = []
    results.append(test_metadata_validation(rule))
    print()
    results.append(test_simple_action_exception(rule))
    print()
    results.append(test_procedural_detection(rule))
    print()
    results.append(test_introduction_requirements(rule))
    print()
    results.append(test_title_format(rule))
    print()
    results.append(test_heading_case(rule))
    print()
    results.append(test_module_nesting(rule))
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

