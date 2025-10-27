#!/usr/bin/env python3
"""
World-Class Test Suite for Assembly Module Rule
Tests EVERY requirement from Red Hat guidelines systematically.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules.modular_compliance.assembly_module_rule import AssemblyModuleRule

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
:_mod-docs-content-type: ASSEMBLY

= User Story Assembly

include::con_concept.adoc[]
"""
    errors = rule.analyze(no_intro, {'content_type': 'assembly'})
    intro_errors = [e for e in errors if 'introduction' in e.get('message', '').lower()]
    if len(intro_errors) > 0:
        result.add_pass("Flags missing introduction")
    else:
        result.add_fail("Missed missing introduction")
    
    # Valid introduction (with substantial content triggers parser fallback)
    with_intro = """
:_mod-docs-content-type: ASSEMBLY

= Configuring Network Security

This assembly guides you through configuring network security for production environments and testing scenarios.

include::con_security-concepts.adoc[]
include::proc_configure-firewall.adoc[]

== Additional context
More information here.
"""
    errors = rule.analyze(with_intro, {'content_type': 'assembly'})
    intro_errors = [e for e in errors if 'introduction' in e.get('message', '').lower() and 'lack' in e.get('message', '').lower()]
    if len(intro_errors) == 0:
        result.add_pass("Accepts valid introduction")
    else:
        result.add_fail("False positive on valid introduction")
    
    return result.summary()

def test_include_directives_requirement(rule):
    """Test that assemblies must include modules."""
    result = TestResult("Include Directives Requirement")
    
    # No includes
    no_includes = """
:_mod-docs-content-type: ASSEMBLY

= User Story

This is an assembly but has no modules.

== Section
Content here.
"""
    errors = rule.analyze(no_includes, {'content_type': 'assembly'})
    include_errors = [e for e in errors if 'include' in e.get('message', '').lower()]
    if len(include_errors) > 0:
        result.add_pass("Flags missing include directives")
    else:
        result.add_fail("Missed missing includes")
    
    # With includes
    with_includes = """
:_mod-docs-content-type: ASSEMBLY

= User Story

This assembly includes modules.

include::con_understanding.adoc[]
include::proc_installing.adoc[]
include::ref_configuration.adoc[]
"""
    errors = rule.analyze(with_includes, {'content_type': 'assembly'})
    include_errors = [e for e in errors if 'include' in e.get('message', '').lower() and 'no include' in e.get('message', '').lower()]
    if len(include_errors) == 0:
        result.add_pass("Accepts assemblies with includes")
    else:
        result.add_fail("False positive on valid includes")
    
    return result.summary()

def test_nesting_depth_warning(rule):
    """Test deep nesting warning."""
    result = TestResult("Nesting Depth Warning")
    
    # Deep nesting (>3 assemblies)
    deep_nesting = """
:_mod-docs-content-type: ASSEMBLY

= Top Level Assembly

Introduction.

include::assembly_level1.adoc[]
include::assembly_level2.adoc[]
include::assembly_level3.adoc[]
include::assembly_level4.adoc[]
"""
    errors = rule.analyze(deep_nesting, {'content_type': 'assembly'})
    nest_errors = [e for e in errors if 'nesting' in e.get('message', '').lower() or 'assemblies' in e.get('message', '').lower()]
    if len(nest_errors) > 0:
        result.add_pass("Warns on deep nesting (>3 assemblies)")
    else:
        result.add_fail("Missed deep nesting warning")
    
    # Shallow nesting (acceptable)
    shallow = """
:_mod-docs-content-type: ASSEMBLY

= Assembly

Intro.

include::assembly_sub.adoc[]
include::con_concept.adoc[]
"""
    errors = rule.analyze(shallow, {'content_type': 'assembly'})
    nest_errors = [e for e in errors if 'nesting' in e.get('message', '').lower() and 'too deep' in e.get('message', '').lower()]
    if len(nest_errors) == 0:
        result.add_pass("Accepts shallow nesting")
    else:
        result.add_fail("False positive on shallow nesting")
    
    return result.summary()

def test_metadata_validation(rule):
    """Test 2023-2024 metadata updates."""
    result = TestResult("Metadata Validation (2023-2024)")
    
    # Deprecated :_content-type:
    old_attr = ":_content-type: ASSEMBLY\n\n= Test\n\nTest.\n\ninclude::mod.adoc[]"
    errors = rule.analyze(old_attr, {'content_type': 'assembly'})
    if any('content-type' in e.get('message', '').lower() and 'deprecated' in e.get('message', '').lower() for e in errors):
        result.add_pass("Detects deprecated :_content-type:")
    else:
        result.add_fail("Missed deprecated :_content-type:")
    
    # Deprecated [role="_abstract"]
    deprecated_role = ':_mod-docs-content-type: ASSEMBLY\n\n[role="_abstract"]\nTest.\n\n= Test\n\ninclude::mod.adoc[]'
    errors = rule.analyze(deprecated_role, {'content_type': 'assembly'})
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
:_mod-docs-content-type: ASSEMBLY

= User Story Assembly

Assembly intro.

== Additional Resources And Related Information

* Link

include::mod.adoc[]
"""
    errors = rule.analyze(title_case, {'content_type': 'assembly'})
    case_errors = [e for e in errors if 'case' in e.get('message', '').lower()]
    if len(case_errors) > 0:
        result.add_pass("Flags title case headings")
    else:
        result.add_fail("Missed title case heading")
    
    # Sentence case (valid)
    sentence_case = """
:_mod-docs-content-type: ASSEMBLY

= User story assembly

Assembly intro.

== Additional resources

* Link

include::mod.adoc[]
"""
    errors = rule.analyze(sentence_case, {'content_type': 'assembly'})
    case_errors = [e for e in errors if 'case' in e.get('message', '').lower()]
    if len(case_errors) == 0:
        result.add_pass("Accepts sentence case")
    else:
        result.add_fail("False positive on sentence case")
    
    return result.summary()

def test_additional_resources_conflict(rule):
    """Test 2023 Sept: Additional resources conflict detection."""
    result = TestResult("Additional Resources Conflict (2023 Sept)")
    
    # Assembly has Additional resources
    with_resources = """
:_mod-docs-content-type: ASSEMBLY

= User Story

Assembly intro.

include::con_concept.adoc[]
include::proc_procedure.adoc[]

== Additional resources

* Link
"""
    errors = rule.analyze(with_resources, {'content_type': 'assembly'})
    resource_warnings = [e for e in errors if 'additional resources' in e.get('message', '').lower()]
    if len(resource_warnings) > 0:
        result.add_pass("Warns about potential Additional resources conflict")
    else:
        result.add_fail("Missed Additional resources conflict check")
    
    return result.summary()

def run_all_tests():
    """Run comprehensive test suite."""
    rule = AssemblyModuleRule()
    
    print("=" * 70)
    print("WORLD-CLASS ASSEMBLY MODULE TEST SUITE")
    print("Testing against Red Hat Modular Documentation Guidelines")
    print("=" * 70)
    print()
    
    results = []
    results.append(test_introduction_requirement(rule))
    print()
    results.append(test_include_directives_requirement(rule))
    print()
    results.append(test_nesting_depth_warning(rule))
    print()
    results.append(test_metadata_validation(rule))
    print()
    results.append(test_heading_case(rule))
    print()
    results.append(test_additional_resources_conflict(rule))
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

