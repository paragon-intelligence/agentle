#!/usr/bin/env python3
import os
from pathlib import Path


def create_test_files(source_dir: str = "agentle", target_dir: str = "tests") -> None:
    """
    Creates test files for every Python module in source_dir.
    Maintains the same directory structure in the target_dir.

    Args:
        source_dir: Directory containing the source modules
        target_dir: Directory where test files will be created
    """
    # Convert to Path objects
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Ensure target directory exists
    target_path.mkdir(exist_ok=True)

    # Make sure there's an __init__.py in the tests directory
    init_file = target_path / "__init__.py"
    if not init_file.exists():
        init_file.touch()

    # Get the absolute path to correctly identify the source directory
    source_absolute = source_path.absolute()

    # Walk through the source directory
    for root, _, files in os.walk(source_path):
        # Convert to Path objects
        root_path = Path(root)

        # Compute the relative path from the source_dir
        rel_path = root_path.relative_to(
            source_path.parent
            if root == str(source_absolute)
            else source_absolute.parent
        )

        # Create the corresponding directory in the target directory
        if root != str(source_path):
            target_subdir = target_path / root_path.relative_to(source_path)
            target_subdir.mkdir(exist_ok=True)

            # Create __init__.py in the test subdirectory
            test_init_file = target_subdir / "__init__.py"
            if not test_init_file.exists():
                test_init_file.touch()

        # Process Python files
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Determine the corresponding test file path
                if root == str(source_path):
                    test_file_path = target_path / f"test_{file}"
                else:
                    relative_source_path = root_path.relative_to(source_path)
                    test_file_path = target_path / relative_source_path / f"test_{file}"

                # Check if test file already exists
                if not test_file_path.exists():
                    print(f"Creating test file: {test_file_path}")

                    # Create a basic test template
                    source_module_path = (
                        str(rel_path / file)[:-3].replace("/", ".").replace("\\", ".")
                    )
                    if source_module_path.startswith("."):
                        source_module_path = source_module_path[1:]

                    with open(str(test_file_path), "w") as f:
                        f.write(f"""import pytest
from {source_module_path} import *

def test_{file[:-3]}_placeholder():
    \"\"\"
    Placeholder test for {source_module_path}
    Replace with actual tests
    \"\"\"
    assert True  # Replace with actual test assertions
""")
                else:
                    print(f"Test file already exists: {test_file_path}")


if __name__ == "__main__":
    create_test_files()
    print("Test structure generation complete!")
