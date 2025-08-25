"""
Interface abstrata para clientes de telemetria OpenTelemetry.

Este módulo define a interface base que todos os clientes de telemetria devem implementar
para funcionar com o decorador @observe genérico. A interface é agnóstica ao provedor
específico, permitindo extensibilidade fácil para diferentes serviços de observabilidade.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, AsyncGenerator, Optional, Protocol, runtime_checkable

from rsb.models.base_model import BaseModel


@runtime_checkable
class TraceContext(Protocol):
    """Protocolo para contextos de trace que podem ser utilizados entre diferentes clientes."""

    def get_trace_id(self) -> Optional[str]:
        """Retorna o ID do trace atual."""
        ...


@runtime_checkable
class GenerationContext(Protocol):
    """Protocolo para contextos de geração que podem ser utilizados entre diferentes clientes."""

    def get_generation_id(self) -> Optional[str]:
        """Retorna o ID da geração atual."""
        ...

    def get_trace_id(self) -> Optional[str]:
        """Retorna o ID do trace pai."""
        ...


class OtelClient(BaseModel, ABC):
    """
    Interface abstrata para clientes de telemetria OpenTelemetry.

    Esta classe define o contrato que todos os clientes de telemetria devem seguir
    para serem compatíveis com o sistema de observabilidade do Agentle. Implementações
    específicas (como Langfuse, DataDog, etc.) devem herdar desta classe.

    A interface foi projetada para ser:
    - Agnóstica ao provedor: funciona com qualquer sistema de telemetria
    - Assíncrona: suporta operações não-bloqueantes
    - Robusta: falhas não devem interromper a execução principal
    - Extensível: novos provedores podem ser adicionados facilmente
    """

    @abstractmethod
    async def trace_context(
        self,
        *,
        name: str,
        input_data: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> AsyncGenerator[Optional[TraceContext], None]:
        """
        Cria um contexto de trace para rastrear uma operação completa.

        Args:
            name: Nome identificador do trace
            input_data: Dados de entrada da operação
            metadata: Metadados adicionais
            user_id: Identificador do usuário
            session_id: Identificador da sessão
            tags: Tags para categorização

        Yields:
            TraceContext: Contexto do trace criado
        """
        yield None  # Implementação padrão retorna None

    @abstractmethod
    async def generation_context(
        self,
        *,
        trace_context: Optional[TraceContext] = None,
        name: str,
        model: str,
        provider: str,
        input_data: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AsyncGenerator[Optional[GenerationContext], None]:
        """
        Cria um contexto de geração para rastrear uma invocação de modelo de IA.

        Args:
            trace_context: Contexto do trace pai
            name: Nome da geração
            model: Identificador do modelo utilizado
            provider: Nome do provedor (google, openai, etc.)
            input_data: Dados de entrada para o modelo
            metadata: Metadados adicionais

        Yields:
            GenerationContext: Contexto da geração criada
        """
        yield None  # Implementação padrão retorna None

    @abstractmethod
    async def update_generation(
        self,
        generation_context: GenerationContext,
        *,
        output_data: Mapping[str, Any],
        usage_details: Optional[Mapping[str, Any]] = None,
        cost_details: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        """
        Atualiza uma geração com dados de saída e métricas.

        Args:
            generation_context: Contexto da geração a ser atualizada
            output_data: Dados de saída do modelo
            usage_details: Detalhes de uso (tokens, etc.)
            cost_details: Detalhes de custo
            metadata: Metadados adicionais
            end_time: Timestamp de finalização
        """
        ...

    @abstractmethod
    async def update_trace(
        self,
        trace_context: TraceContext,
        *,
        output_data: Any,
        success: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        """
        Atualiza um trace com dados finais.

        Args:
            trace_context: Contexto do trace a ser atualizado
            output_data: Dados de saída da operação
            success: Se a operação foi bem-sucedida
            metadata: Metadados adicionais
            end_time: Timestamp de finalização
        """
        ...

    @abstractmethod
    async def add_trace_score(
        self,
        trace_context: TraceContext,
        *,
        name: str,
        value: float | str,
        comment: Optional[str] = None,
    ) -> None:
        """
        Adiciona uma pontuação ao trace.

        Args:
            trace_context: Contexto do trace
            name: Nome da métrica
            value: Valor da pontuação
            comment: Comentário opcional
        """
        ...

    @abstractmethod
    async def handle_error(
        self,
        trace_context: Optional[TraceContext],
        generation_context: Optional[GenerationContext],
        error: Exception,
        start_time: datetime,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Trata erros durante a execução.

        Args:
            trace_context: Contexto do trace
            generation_context: Contexto da geração
            error: Exceção ocorrida
            start_time: Timestamp do início da operação
            metadata: Metadados adicionais
        """
        ...

    @abstractmethod
    async def flush(self) -> None:
        """
        Força o envio imediato de todos os eventos pendentes.

        Este método garante que todos os dados de telemetria sejam enviados
        ao backend antes do término da aplicação.
        """
        ...
