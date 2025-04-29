Contributing
============

We welcome contributions to Agentle! This document outlines the process for contributing to the project.

Development Setup
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

4. Create a branch for your feature:

   .. code-block:: bash

      git checkout -b feature-name

Code Standards
------------

- We use mypy for type checking
- Ruff for linting
- All code should be properly typed

Testing
------

Before submitting a pull request, make sure all tests pass:

.. code-block:: bash

   # Run tests command here

Documentation
------------

Please update the documentation when adding or modifying features.

To build the documentation locally:

.. code-block:: bash

   cd docs
   make html

The built documentation will be available in the `docs/build/html` directory.

Submitting Changes
----------------

1. Push your changes to your fork
2. Submit a pull request to the main repository
3. Ensure CI checks pass
4. Update your PR based on reviewer feedback 