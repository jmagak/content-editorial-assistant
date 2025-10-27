#!/usr/bin/env python3
"""
Script to apply Universal Code Context Guard to all rule files.
This guard prevents prose rules from analyzing technical code blocks.

Block types to exclude:
- AsciiDoc: 'listing', 'literal' 
- Markdown: 'code_block', 'inline_code'

Usage:
    python scripts/apply_universal_code_guard.py [--dry-run]
"""
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Guard clause to insert (with proper indentation preserved)
GUARD_CLAUSE = '''# === UNIVERSAL CODE CONTEXT GUARD ===
# Skip analysis for code blocks, listings, and literal blocks (technical syntax, not prose)
if context and context.get('block_type') in ['listing', 'literal', 'code_block', 'inline_code']:
    return []
'''

def find_rule_files(base_path: Path) -> List[Path]:
    """Find all Python rule files that have an analyze method."""
    rule_files = []
    
    # Search in rules directory
    rules_dir = base_path / 'rules'
    
    for py_file in rules_dir.rglob('*.py'):
        # Skip __init__.py and base files (they don't have concrete implementations)
        if py_file.name in ['__init__.py']:
            continue
            
        # Read file and check if it has an analyze method
        try:
            content = py_file.read_text()
            if 'def analyze(' in content and 'return errors' in content:
                rule_files.append(py_file)
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")
    
    return sorted(rule_files)

def check_if_guard_exists(content: str) -> bool:
    """Check if the file already has the universal code guard."""
    # Check for various patterns of the guard
    patterns = [
        r"context\.get\('block_type'\)\s+in\s+\['listing',\s*'literal',\s*'code_block',\s*'inline_code'\]",
        r"UNIVERSAL CODE CONTEXT GUARD",
        r"Skip analysis for code blocks, listings, and literal blocks"
    ]
    
    for pattern in patterns:
        if re.search(pattern, content):
            return True
    return False

def find_analyze_method_start(content: str) -> Tuple[int, int]:
    """
    Find the line number where analyze method starts and where to insert the guard.
    Returns (analyze_line, insert_line) or (None, None) if not found.
    """
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        # Look for the analyze method definition
        if re.match(r'\s*def analyze\(', line):
            # Find where to insert the guard (after docstring if present)
            insert_line = i + 1
            
            # Skip past the docstring if present
            if insert_line < len(lines):
                next_line = lines[insert_line].strip()
                if next_line.startswith('"""') or next_line.startswith("'''"):
                    # Find the end of docstring
                    quote = '"""' if next_line.startswith('"""') else "'''"
                    in_docstring = True
                    insert_line += 1
                    
                    while insert_line < len(lines) and in_docstring:
                        if quote in lines[insert_line]:
                            in_docstring = False
                            insert_line += 1
                            break
                        insert_line += 1
            
            # Skip any existing context checks or variable initialization
            while insert_line < len(lines):
                line_stripped = lines[insert_line].strip()
                # Stop if we hit actual logic (errors =, doc =, etc.)
                if (line_stripped.startswith('errors') or 
                    line_stripped.startswith('doc =') or
                    line_stripped.startswith('if not nlp') or
                    line_stripped.startswith('for ') or
                    line_stripped.startswith('return ')):
                    break
                # Skip blank lines, comments, and context = context or {}
                if (not line_stripped or 
                    line_stripped.startswith('#') or
                    'context = context or' in line_stripped or
                    'context = context if context' in line_stripped):
                    insert_line += 1
                    continue
                # If we see an existing guard, don't insert
                if 'block_type' in line_stripped and 'return' in lines[insert_line:insert_line+3]:
                    return (i, None)  # Guard already exists
                break
            
            return (i, insert_line)
    
    return (None, None)

def insert_guard(content: str, insert_line: int) -> str:
    """Insert the guard clause at the specified line."""
    lines = content.split('\n')
    
    # Get indentation from the surrounding lines
    indent = '        '  # Default 8 spaces for method body
    if insert_line < len(lines) and lines[insert_line]:
        # Try to match indentation of next line
        match = re.match(r'^(\s+)', lines[insert_line])
        if match:
            indent = match.group(1)
    
    # Insert the guard with proper indentation
    guard_lines = GUARD_CLAUSE.strip().split('\n')
    indented_lines = []
    for line in guard_lines:
        if line.strip():  # If line has content
            # For the return statement, add extra indentation
            if line.strip().startswith('return'):
                indented_lines.append(indent + '    ' + line.strip())
            else:
                indented_lines.append(indent + line.strip())
        else:
            indented_lines.append('')  # Keep blank lines blank
    
    # Insert all guard lines at once
    for i, guard_line in enumerate(reversed(indented_lines)):
        lines.insert(insert_line, guard_line)
    
    return '\n'.join(lines)

def apply_guard_to_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Apply the guard to a single file.
    Returns (success, message).
    """
    try:
        content = file_path.read_text()
        
        # Check if guard already exists
        if check_if_guard_exists(content):
            return (False, "Guard already exists")
        
        # Find where to insert
        analyze_line, insert_line = find_analyze_method_start(content)
        
        if analyze_line is None:
            return (False, "No analyze method found")
        
        if insert_line is None:
            return (False, "Guard already exists (detected during insertion)")
        
        # Insert the guard
        new_content = insert_guard(content, insert_line)
        
        if dry_run:
            return (True, f"Would insert guard at line {insert_line + 1}")
        else:
            file_path.write_text(new_content)
            return (True, f"✅ Inserted guard at line {insert_line + 1}")
    
    except Exception as e:
        return (False, f"Error: {e}")

def main():
    """Main execution function."""
    dry_run = '--dry-run' in sys.argv
    
    # Get the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print("=" * 80)
    print("Universal Code Context Guard Application")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()
    
    # Find all rule files
    print("Finding rule files...")
    rule_files = find_rule_files(project_root)
    print(f"Found {len(rule_files)} rule files with analyze methods\n")
    
    # Apply guard to each file
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    for rule_file in rule_files:
        rel_path = rule_file.relative_to(project_root)
        success, message = apply_guard_to_file(rule_file, dry_run)
        
        if success:
            print(f"✅ {rel_path}")
            print(f"   {message}")
            success_count += 1
        elif "already exists" in message:
            print(f"⏭️  {rel_path}")
            print(f"   {message}")
            skipped_count += 1
        else:
            print(f"❌ {rel_path}")
            print(f"   {message}")
            error_count += 1
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(rule_files)}")
    print(f"✅ Successfully {'would apply' if dry_run else 'applied'} guard: {success_count}")
    print(f"⏭️  Skipped (guard exists): {skipped_count}")
    print(f"❌ Errors: {error_count}")
    print()
    
    if dry_run:
        print("🔍 This was a DRY RUN. No files were modified.")
        print("   Run without --dry-run to apply changes.")
    else:
        print("✨ Changes applied successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()

