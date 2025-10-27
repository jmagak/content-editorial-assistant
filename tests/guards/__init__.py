"""
Guard Test Suite

This package contains all Zero False Positive Guard tests for the Content Editorial Assistant.

Each guard test file follows the Three-Part Analysis methodology:
1. Objective Truth Test - Validates that the flagged pattern is actually correct
2. False Negative Risk Assessment - Ensures real errors are still caught
3. Inversion Test - Confirms the guard doesn't suppress legitimate errors

Guards implemented:
- Guard 1: Verbs - Prerequisites Present Perfect
- Guard 2: Plurals - GitOps/ACS Proper Nouns
- Guard 3: Colons - Contextual Labels
- Guard 4: Capitalization - Technical Commands
- Guard 5: Ambiguity - Technical Commands as Promises
- Guard 6: Ambiguity - Cross-Sentence Subject Reference
- Guard 7: Periods - URLs, Commands, Technical Content
- Guard 8: Passive Voice - Imperative Mood
"""

