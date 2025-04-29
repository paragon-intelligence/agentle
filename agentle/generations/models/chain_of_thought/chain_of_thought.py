"""
Chain of Thought reasoning implementation for the Agentle framework.

This module provides the primary model for representing and working with
chain-of-thought reasoning processes in AI generations. The Chain of Thought
technique makes the reasoning process explicit by breaking down complex problem-solving
into discrete, observable steps, leading to a final answer.

This structured approach offers several benefits:
- Improved reasoning transparency for complex tasks
- Better error detection and debugging of model reasoning
- Support for step-by-step verification of logical processes
- Enhanced explainability for regulatory or user trust requirements

The module implements a Pydantic model for structured Chain of Thought representation
with multilingual output formatting capabilities.
"""

from typing import Sequence

from pydantic import BaseModel, Field

from agentle.generations.models.chain_of_thought.step import Step


class ChainOfThought[T](BaseModel):
    """
    Structured reasoning process with final answer.

    This class represents a complete chain-of-thought reasoning process,
    breaking down complex problem-solving into a sequence of explicit steps
    with detailed explanations, culminating in a final answer or conclusion.

    Chain of Thought is particularly useful for:
    - Complex reasoning tasks requiring step-by-step thinking
    - Making model reasoning transparent and verifiable
    - Debugging or explaining how a model arrived at a conclusion
    - Implementing research techniques like Chain of Thought prompting

    The generic type parameter T allows the final_answer to be of any type,
    such as a string, number, boolean, or complex structured data.

    Attributes:
        general_title: High-level description of reasoning goal
        steps: Logical steps in reasoning process
        final_answer: Conclusion of the reasoning chain, of type T

    Example:
        >>> ChainOfThought(
        ...     general_title="Math problem solution",
        ...     steps=[step1, step2],
        ...     final_answer=42
        ... )
    """

    general_title: str = Field(
        description="A brief label or description that identifies the purpose of the reasoning.",
        # examples=["Sum of two numbers", "Logical problem solving"],
    )

    steps: Sequence[Step] = Field(
        description="The sequence of steps that make up the full reasoning process.",
    )

    final_answer: T = Field(
        description="The conclusion or result after all the reasoning steps."
    )

    def as_string(self, lang: str = "en") -> str:
        """Return a localized string representation of the ChainOfThought.

        Args:
            lang: ISO language code for the output format (default: "en" for English)

        Returns:
            str: A formatted string showing the reasoning process and final answer in the specified language

        Example:
            >>> print(chain_of_thought.as_string())  # Default English
            MATH PROBLEM SOLUTION

            Step 1: Analyze input data
            - Data: 234 and 567
            - Check if they are integers

            Step 2: Perform the calculation
            - 234 + 567 = 801

            Final Answer: 801

            >>> print(chain_of_thought.as_string("es"))  # Spanish
            SOLUCIÓN DEL PROBLEMA MATEMÁTICO

            Paso 1: Analizar datos de entrada
            - Datos: 234 y 567
            - Comprobar si son números enteros

            Paso 2: Realizar el cálculo
            - 234 + 567 = 801

            Respuesta Final: 801

            >>> print(chain_of_thought.as_string("pt"))  # Portuguese
            SOLUÇÃO DO PROBLEMA MATEMÁTICO

            Passo 1: Analisar dados de entrada
            - Dados: 234 e 567
            - Verificar se são números inteiros

            Passo 2: Realizar o cálculo
            - 234 + 567 = 801

            Resposta Final: 801
        """
        # Define language-specific terms
        translations = {
            "en": {"step": "Step", "final_answer": "Final Answer"},
            "es": {"step": "Paso", "final_answer": "Respuesta Final"},
            "fr": {"step": "Étape", "final_answer": "Réponse Finale"},
            "de": {"step": "Schritt", "final_answer": "Endgültige Antwort"},
            "pt": {"step": "Passo", "final_answer": "Resposta Final"},
        }

        # Default to English if requested language is not available
        if lang not in translations:
            lang = "en"

        # Get translation dictionary for the specified language
        t = translations[lang]

        # Start with the title
        result = f"{self.general_title.upper()}\n\n"

        # Add each step with its details
        for step in self.steps:
            result += f"{t['step']} {step.step_number}: {step.explanation}\n"
            for detail in step.details:
                result += f"- {detail.detail}\n"
            result += "\n"

        # Add the final answer
        result += f"{t['final_answer']}: {self.final_answer}"

        return result
