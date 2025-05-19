======================
Agentle Documentation
======================

Welcome to Agentle's documentation. Agentle is a powerful yet elegant framework for building the next generation of AI agents.

.. image:: /../../docs/logo.png
   :alt: Agentle Logo
   :width: 200
   :align: center

Author's Note
------------

I created Agentle out of frustration with the direction of other agent frameworks. Many frameworks have lost sight of clean design principles by adding numerous configuration flags to their Agent constructors (like ``enable_whatever=True``, ``add_memory=True``, etc.). This approach creates countless possible combinations, making debugging and development unnecessarily complex.

Agentle strives to maintain a careful balance between simplicity and practicality. For example, I've wrestled with questions like whether document parsing functionality belongs in the Agent constructor. While not "simple" in the purest sense, such features can be practical for users. Finding this balance is central to Agentle's design philosophy.

Core principles of Agentle:

* Avoiding configuration flags in constructors whenever possible
* Organizing each class and function in separate modules by design
* Following the Single Responsibility Principle rather than strictly Pythonic conventions
* Creating a codebase that's not only easy to use but also easy to maintain and extend

Through this thoughtful approach to architecture, Agentle aims to provide a framework that's both powerful and elegant for building the next generation of AI agents.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   usage
   agents
   inputs
   tools
   structured_outputs
   prompt_management
   knowledge_integration
   agent_composition
   a2a
   mcp
   observability
   deployment
   examples
   contributing
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

