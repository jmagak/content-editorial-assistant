"""
Guards Test Template

Copy this template when creating guard tests for a new rule.
Rename to: test_guards_[your_rule_name].py

Each guard requires TWO tests:
1. test_guard_N_prevents_false_positive - Proves the guard fixes a real issue
2. test_guard_N_no_false_negatives - Proves the guard doesn't suppress real errors
"""

import pytest
from rules.your_category.your_rule import YourRule  # REPLACE THIS


class TestGuard1Description:
    """
    GUARD 1: [Short description of what this guard protects]
    
    Context: [Explain the false positive this guard prevents]
    """
    
    def test_guard_1_prevents_false_positive(self):
        """
        Verifies guard prevents false positive
        
        Original Issue: [Describe what was incorrectly flagged]
        Expected: No error (explain why this is objectively correct)
        """
        rule = YourRule()
        
        # Text that was incorrectly flagged before guard
        text = """
        [INSERT EXAMPLE TEXT THAT TRIGGERED FALSE POSITIVE]
        """
        
        context = {
            # Include any context that triggers the guard
        }
        
        results = rule.analyze(text, context)
        
        # Assert no errors in the guarded context
        assert len(results) == 0, (
            f"Guard should prevent error in [context type], "
            f"but found: {results}"
        )
    
    def test_guard_1_no_false_negatives(self):
        """
        Verifies guard doesn't create false negatives
        
        Expected: Real errors outside guarded context are still detected
        """
        rule = YourRule()
        
        # Text with ACTUAL error that should still be caught
        text = """
        [INSERT EXAMPLE WITH LEGITIMATE ERROR]
        """
        
        context = {
            # Context without guard trigger
        }
        
        results = rule.analyze(text, context)
        
        # Assert error IS detected
        assert len(results) > 0, (
            "Guard should not suppress legitimate errors outside guarded context"
        )
        
        # Verify it's the error we expect
        assert results[0].matched_text == "[expected error text]"


class TestGuard2Description:
    """
    GUARD 2: [Short description]
    
    Context: [Explain the false positive this guard prevents]
    """
    
    def test_guard_2_prevents_false_positive(self):
        """[Document the false positive]"""
        # ... implementation ...
        pass
    
    def test_guard_2_no_false_negatives(self):
        """[Document that real errors still work]"""
        # ... implementation ...
        pass


# Add more guard test classes as needed (max 5)


class TestGuardInteractions:
    """
    Tests interactions between multiple guards
    
    Ensures guards don't conflict or create unexpected behavior
    """
    
    def test_multiple_guards_dont_conflict(self):
        """
        Verifies that multiple guards can coexist
        
        If Guard 1 and Guard 2 both could apply, ensure behavior is correct
        """
        rule = YourRule()
        
        # Text that could trigger multiple guards
        text = """
        [EXAMPLE TEXT]
        """
        
        results = rule.analyze(text)
        
        # Assert expected behavior
        assert len(results) == 0, "Multiple guards should not conflict"
    
    def test_guard_priority_correct(self):
        """
        If guards have priority, verify they execute in correct order
        """
        # Only needed if guards have dependencies
        pass


class TestGuardEdgeCases:
    """
    Tests edge cases and boundary conditions
    
    What happens when guard logic is at its limits?
    """
    
    def test_guard_boundary_case_1(self):
        """
        Test behavior at guard boundaries
        
        Example: If guard checks for "4+ items", test with exactly 4
        """
        pass

