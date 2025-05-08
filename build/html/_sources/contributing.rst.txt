Contributing
===========

Thank you for your interest in contributing to Agentle! This guide will help you get started.

Development Setup
----------------

To set up Agentle for local development:

1. Clone the repository from GitHub:

   .. code-block:: bash

      git clone https://github.com/paragon-intelligence/agentle.git
      cd agentle

2. Create and activate a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev,test]"

Code Standards
-------------

Agentle uses:

- Black for code formatting
- Isort for import sorting
- Flake8 for linting
- Mypy for type checking

Testing
-------

Tests are written using pytest. Run the test suite with:

.. code-block:: bash

   pytest tests/

We encourage test-driven development for new features.

Documentation
------------

Documentation is built using Sphinx:

.. code-block:: bash

   cd docs
   make html

Preview the documentation by opening `build/html/index.html` in a web browser.

Please document all public modules, functions, classes, and methods with docstrings following the Google style guide.

Submitting Changes
-----------------

1. Create a new branch for your feature or bugfix:

   .. code-block:: bash

      git checkout -b feature-or-fix-name

2. Make your changes and commit them with clear messages.

3. Push to your fork and submit a pull request.

4. Ensure CI passes on your PR.

Thank you for contributing!