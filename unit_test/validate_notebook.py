#!/usr/bin/env python3
"""
Notebook Validation Script
Validates Jupyter notebook structure and Python syntax in code cells.

Usage:
    python validate_notebook.py <notebook_path>
    python validate_notebook.py comparative_study/03a_CITA_Baseline/Llama3_BF16.ipynb
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Tuple


def validate_notebook(notebook_path: str, verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate a Jupyter notebook for structure and Python syntax.

    Args:
        notebook_path: Path to the .ipynb file
        verbose: If True, print detailed progress

    Returns:
        Tuple of (success: bool, errors: List[str])
    """
    errors = []
    notebook_path = Path(notebook_path)

    # Check if file exists
    if not notebook_path.exists():
        errors.append(f"File not found: {notebook_path}")
        return False, errors

    # Try to read as JSON
    try:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
        if verbose:
            print(f'✅ Notebook JSON structure valid: {notebook_path.name}')
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return False, errors
    except Exception as e:
        errors.append(f"Error reading file: {e}")
        return False, errors

    # Validate notebook structure
    if 'cells' not in nb:
        errors.append("Missing 'cells' key in notebook")
        return False, errors

    cells = nb['cells']
    if verbose:
        print(f'📊 Total cells: {len(cells)}')

    # Analyze cells
    code_cells = 0
    markdown_cells = 0
    syntax_errors = []

    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type', 'unknown')

        if cell_type == 'code':
            code_cells += 1

            # Get cell source
            source = cell.get('source', [])
            if isinstance(source, list):
                source = ''.join(source)

            # Skip empty cells
            if not source.strip():
                if verbose:
                    print(f'  ⚪ Cell {i} (code): empty, skipped')
                continue

            # Try to compile the code
            try:
                compile(source, f'<cell-{i}>', 'exec')
                if verbose:
                    print(f'  ✅ Cell {i} (code): syntax OK ({len(source)} chars)')
            except SyntaxError as e:
                error_msg = f"Cell {i}: {e.msg} (line {e.lineno})"
                syntax_errors.append((i, error_msg))
                errors.append(error_msg)
                if verbose:
                    print(f'  ❌ Cell {i} (code): SYNTAX ERROR')
                    print(f'     {e}')

        elif cell_type == 'markdown':
            markdown_cells += 1
        else:
            if verbose:
                print(f'  ⚠️  Cell {i}: unknown type "{cell_type}"')

    # Print summary
    if verbose:
        print(f'\n📈 Summary:')
        print(f'  Total cells: {len(cells)}')
        print(f'  Code cells: {code_cells}')
        print(f'  Markdown cells: {markdown_cells}')
        print(f'  Syntax errors: {len(syntax_errors)}')

    # Determine success
    success = len(errors) == 0

    if verbose:
        if success:
            print(f'\n✅ SUCCESS: All code cells have valid Python syntax!')
        else:
            print(f'\n❌ FAILED: Found {len(syntax_errors)} syntax error(s)')
            for i, error in syntax_errors:
                print(f'  - {error}')

    return success, errors


def main():
    parser = argparse.ArgumentParser(
        description='Validate Jupyter notebook structure and Python syntax',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single notebook
  python validate_notebook.py comparative_study/03a_CITA_Baseline/Llama3_BF16.ipynb

  # Validate multiple notebooks
  python validate_notebook.py notebook1.ipynb notebook2.ipynb

  # Quiet mode (only show errors)
  python validate_notebook.py -q notebook.ipynb
        """
    )
    parser.add_argument(
        'notebooks',
        nargs='+',
        help='Path(s) to Jupyter notebook file(s)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode (only show errors)'
    )

    args = parser.parse_args()

    # Validate all notebooks
    all_success = True
    results = []

    for notebook in args.notebooks:
        if not args.quiet:
            print(f'\n{"="*80}')
            print(f'Validating: {notebook}')
            print(f'{"="*80}\n')

        success, errors = validate_notebook(notebook, verbose=not args.quiet)
        results.append((notebook, success, errors))

        if not success:
            all_success = False

    # Print final summary if multiple notebooks
    if len(args.notebooks) > 1:
        print(f'\n{"="*80}')
        print(f'FINAL SUMMARY ({len(args.notebooks)} notebooks)')
        print(f'{"="*80}')

        for notebook, success, errors in results:
            status = '✅ PASS' if success else '❌ FAIL'
            print(f'{status}: {notebook}')
            if errors and not args.quiet:
                for error in errors:
                    print(f'  - {error}')

    # Exit with appropriate code
    sys.exit(0 if all_success else 1)


if __name__ == '__main__':
    main()
