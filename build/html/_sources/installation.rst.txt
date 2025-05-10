============
Installation
============

This guide will help you install Agentle and its dependencies.

Basic Installation
----------------

You can install Agentle using pip:

.. code-block:: bash

    pip install agentle

This will install the core Agentle package with minimal dependencies.

Installing with Optional Dependencies
----------------------------------

Agentle has several optional dependencies for different features:

.. code-block:: bash

    # Install with all optional dependencies
    pip install "agentle[all]"
    
    # Install with specific dependency groups
    pip install "agentle[google]"     # Google AI integration
    pip install "agentle[langfuse]"   # Langfuse observability
    pip install "agentle[http]"       # BlackSheep ASGI server
    pip install "agentle[streamlit]"  # Streamlit UI integration
    pip install "agentle[cache]"      # Caching support (aiocache)

You can also combine dependency groups:

.. code-block:: bash

    # Install with Google AI and Langfuse
    pip install "agentle[google,langfuse]"
    
    # Install for deployment with Google AI
    pip install "agentle[google,asgi,langfuse]"

Installation from Source
----------------------

To install the latest development version from source:

.. code-block:: bash

    git clone https://github.com/paragon-intelligence/agentle.git
    cd agentle
    pip install -e .

Setting Up API Keys
-----------------

Agentle requires API keys for the language model providers you plan to use:

Google AI
~~~~~~~~

For Google AI models (Gemini), you'll need a Google AI API key:

1. Visit the `Google AI Studio <https://aistudio.google.com/app/apikey/>`_
2. Click on "Get API key"
3. Follow the instructions to create a project and API key
4. Set the environment variable (or place it in your .env):

.. code-block:: bash

    # Linux/macOS
    export GOOGLE_API_KEY="your-google-api-key"
    
    # Windows
    set GOOGLE_API_KEY=your-google-api-key

OpenAI
~~~~~

For OpenAI models, you'll need an OpenAI API key:

1. Visit `OpenAI API Keys <https://platform.openai.com/account/api-keys>`_
2. Create a new API key
3. Set the environment variable:

.. code-block:: bash

    # Linux/macOS
    export OPENAI_API_KEY="your-openai-api-key"
    
    # Windows
    set OPENAI_API_KEY=your-openai-api-key


Langfuse (for Observability)
~~~~~~~~~~~~~~~~~~~~~~~~~~

To use Langfuse for tracing and observability:

1. Sign up at `Langfuse <https://cloud.langfuse.com>`_
2. Create a new project
3. Go to Settings → API Keys to get your Public and Secret keys
4. Set the environment variables:

.. code-block:: bash

    # Linux/macOS
    export LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
    export LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
    export LANGFUSE_HOST="https://cloud.langfuse.com"  # Optional
    
    # Windows
    set LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
    set LANGFUSE_SECRET_KEY=your-langfuse-secret-key
    set LANGFUSE_HOST=https://cloud.langfuse.com

System Requirements
----------------

Agentle requires Python 3.13 or higher, with the following specifications:

- Python 3.13+
- 2GB RAM minimum (4GB+ recommended)
- 500MB disk space

Compatibility
-----------

Agentle has been tested on the following platforms:

- Ubuntu 20.04+ (Linux)
- macOS 11+ (Big Sur or later)
- Windows 10/11
- Python 3.13+

Troubleshooting
-------------

If you encounter any installation issues:

1. Ensure you have the latest pip version:

   .. code-block:: bash

       pip install --upgrade pip

2. Check for Python version compatibility:

   .. code-block:: bash

       python --version

3. For SSL certificate errors, you may need to:

   .. code-block:: bash

       pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org agentle

4. If you see errors about aiocache when using knowledge integration with caching, install the optional dependency:

   .. code-block:: bash

       pip install aiocache

Next Steps
---------

Once Agentle is installed, you can:

1. Check out the :doc:`quickstart` guide to create your first agent
2. Explore the :doc:`core concepts <agents>` of the framework
3. Try the examples in the GitHub repository