"""
Implementação específica do cliente Langfuse usando SDK V3 baseado em OpenTelemetry.

Este módulo contém toda a lógica específica do Langfuse, isolada da interface genérica.
Utiliza o novo SDK V3 que é baseado em OpenTelemetry para melhor performance e
compatibilidade com o ecossistema OTEL.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncGenerator, Literal, Optional, cast

from rsb.models.field import Field

from .otel_client import GenerationContext, OtelClient, TraceContext

if TYPE_CHECKING:
    from langfuse._client.span import LangfuseGeneration, LangfuseSpan


class _LangfuseTraceContext:
    """Contexto de trace específico do Langfuse."""

    def __init__(self, span: LangfuseSpan):
        self._span = span

    def get_trace_id(self) -> Optional[str]:
        """Retorna o ID do trace."""
        return self._span.trace_id if self._span else None

    @property
    def span(self) -> LangfuseSpan:
        """Acesso ao span interno do Langfuse."""
        return self._span


class _LangfuseGenerationContext:
    """Contexto de geração específico do Langfuse."""

    def __init__(self, generation: LangfuseGeneration):
        self._generation = generation

    def get_generation_id(self) -> Optional[str]:
        """Retorna o ID da geração."""
        return self._generation.id if self._generation else None

    def get_trace_id(self) -> Optional[str]:
        """Retorna o ID do trace pai."""
        return self._generation.trace_id if self._generation else None

    @property
    def generation(self) -> LangfuseGeneration:
        """Acesso à geração interna do Langfuse."""
        return self._generation


class LangfuseOtelClient(OtelClient):
    """
    Cliente de telemetria específico para Langfuse usando SDK V3.

    Esta implementação utiliza o novo SDK V3 do Langfuse baseado em OpenTelemetry
    para fornecer observabilidade otimizada para aplicações de IA. O cliente
    mantém a precisão crucial de contagens de tokens e cálculos de custo.

    Características principais:
    - Baseado no SDK V3 com OpenTelemetry
    - Suporte completo a traces, spans e gerações
    - Cálculo preciso de custos e tokens
    - Pontuação automática de traces
    - Tratamento robusto de erros
    """

    type: Literal["langfuse"] = Field(default="langfuse")

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        environment: Optional[str] = None,
        release: Optional[str] = None,
    ):
        """
        Inicializa o cliente Langfuse.

        Args:
            public_key: Chave pública do Langfuse
            secret_key: Chave secreta do Langfuse
            host: URL do host Langfuse
            environment: Ambiente de execução
            release: Versão/release da aplicação
        """
        from langfuse._client.client import Langfuse

        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            environment=environment,
            release=release,
        )
        self._logger = logging.getLogger(__name__)

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
        Cria um contexto de trace no Langfuse.

        Utiliza o método start_as_current_span do SDK V3 para criar um trace
        que será automaticamente gerenciado pelo contexto.
        """
        try:
            # Preparar metadados do trace
            trace_metadata = dict(metadata) if metadata else {}

            # Usar start_as_current_span do SDK V3
            with self._client.start_as_current_span(
                name=name,
                input=dict(input_data),
                metadata=trace_metadata,
            ) as span:
                # Atualizar informações do trace se fornecidas
                if user_id or session_id or tags:
                    span.update_trace(
                        user_id=user_id,
                        session_id=session_id,
                        tags=list(tags) if tags else None,
                    )

                yield _LangfuseTraceContext(span)

        except Exception as e:
            self._logger.error(f"Erro ao criar trace context: {e}")
            yield None

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
        Cria um contexto de geração no Langfuse.

        Utiliza o método start_as_current_generation do SDK V3 para criar
        uma geração dentro do trace atual.
        """
        try:
            # Preparar metadados da geração
            generation_metadata = dict(metadata) if metadata else {}
            generation_metadata.update(
                {
                    "provider": provider,
                    "model": model,
                }
            )

            # Usar start_as_current_generation do SDK V3
            with self._client.start_as_current_generation(
                name=name,
                model=model,
                input=dict(input_data),
                metadata=generation_metadata,
            ) as generation:
                yield _LangfuseGenerationContext(generation)

        except Exception as e:
            self._logger.error(f"Erro ao criar generation context: {e}")
            yield None

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

        Esta implementação garante que as contagens de tokens e custos
        sejam registrados com precisão no Langfuse UI.
        """
        if not isinstance(generation_context, _LangfuseGenerationContext):
            return

        try:
            generation = generation_context.generation

            # Preparar dados para atualização em uma única chamada
            update_params = {"output": dict(output_data)}

            # Adicionar metadados se fornecidos
            if metadata:
                update_params["metadata"] = dict(metadata)

            # ✅ FIX: Mapear para o formato correto esperado pelo Langfuse V3
            if usage_details:
                langfuse_usage = {
                    "prompt_tokens": usage_details.get("input", 0),
                    "completion_tokens": usage_details.get("output", 0),
                    "total_tokens": usage_details.get("total", 0),
                }
                update_params["usage_details"] = langfuse_usage

            # ✅ FIX: Try multiple cost field name formats to ensure compatibility
            if cost_details:
                input_cost = float(cost_details.get("input", 0.0))
                output_cost = float(cost_details.get("output", 0.0))
                total_cost = float(cost_details.get("total", 0.0))

                # Ensure we have meaningful cost values (not zero)
                if total_cost > 0 or input_cost > 0 or output_cost > 0:
                    # Try the format that Langfuse V3 documentation suggests
                    langfuse_cost = {
                        "total": total_cost,
                    }

                    # Also include breakdown if available
                    if input_cost > 0:
                        langfuse_cost["input"] = input_cost
                    if output_cost > 0:
                        langfuse_cost["output"] = output_cost

                    update_params["cost_details"] = langfuse_cost

                    # Also add cost information to metadata for better visibility
                    cost_metadata = {
                        "cost_usd_input": input_cost,
                        "cost_usd_output": output_cost,
                        "cost_usd_total": total_cost,
                        "currency": "USD",
                    }

                    if "metadata" not in update_params:
                        update_params["metadata"] = {}
                    update_params["metadata"].update(cost_metadata)

            # ✅ FIX: Fazer uma única chamada de update com todos os parâmetros
            generation.update(**update_params)  # type: ignore

            # ✅ FIX: Also try setting cost details as generation attributes
            if cost_details and hasattr(generation, "_otel_span"):
                try:
                    # Set cost as span attributes for better visibility
                    span = generation._otel_span  # type: ignore
                    span.set_attribute(
                        "cost.total", float(cost_details.get("total", 0.0))
                    )
                    span.set_attribute(
                        "cost.input", float(cost_details.get("input", 0.0))
                    )
                    span.set_attribute(
                        "cost.output", float(cost_details.get("output", 0.0))
                    )
                    span.set_attribute("cost.currency", "USD")
                except Exception as e:
                    self._logger.debug(f"Could not set cost attributes: {e}")

        except Exception as e:
            self._logger.error(f"Erro ao atualizar geração: {e}")

    async def update_trace(
        self,
        trace_context: TraceContext,
        *,
        output_data: Any,
        success: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        """Atualiza um trace com dados finais."""
        if not isinstance(trace_context, _LangfuseTraceContext):
            return

        try:
            span = trace_context.span

            # Preparar metadados do trace
            trace_metadata = dict(metadata) if metadata else {}
            trace_metadata["success"] = success

            # ✅ FIX: Extract cost information from output_data and add to trace
            if isinstance(output_data, dict):
                # Check if cost summary is in output data
                if "cost_summary" in output_data:
                    cost_summary = output_data["cost_summary"]
                    trace_metadata.update(
                        {
                            "cost_details": {
                                "input": cost_summary.get("input_cost", 0.0),
                                "output": cost_summary.get("output_cost", 0.0),
                            },
                        }
                    )

                # Check if usage summary is in output data
                if "usage_summary" in output_data:
                    usage_summary = output_data["usage_summary"]
                    trace_metadata.update(
                        {
                            "tokens_total": usage_summary.get("total_tokens", 0),
                            "tokens_input": usage_summary.get("input_tokens", 0),
                            "tokens_output": usage_summary.get("output_tokens", 0),
                        }
                    )

            # ✅ FIX: Update trace with cost information using the span's update_trace method
            span.update_trace(
                output=cast(dict[str, Any], output_data),
                metadata=trace_metadata,
            )

            # ✅ FIX: Also try to set cost directly on the trace if possible
            if hasattr(span, "_otel_span"):
                try:
                    otel_span = span._otel_span  # type: ignore[reportPrivateUsage]
                    if hasattr(otel_span, "set_attribute"):
                        # Set cost attributes on the span
                        if "cost_total" in trace_metadata:
                            otel_span.set_attribute(
                                "cost.total", float(trace_metadata["cost_total"])
                            )
                        if "cost_input" in trace_metadata:
                            otel_span.set_attribute(
                                "cost.input", float(trace_metadata["cost_input"])
                            )
                        if "cost_output" in trace_metadata:
                            otel_span.set_attribute(
                                "cost.output", float(trace_metadata["cost_output"])
                            )
                        if "tokens_total" in trace_metadata:
                            otel_span.set_attribute(
                                "usage.total_tokens",
                                int(trace_metadata["tokens_total"]),
                            )
                except Exception as e:
                    self._logger.debug(f"Could not set trace attributes: {e}")

        except Exception as e:
            self._logger.error(f"Erro ao atualizar trace: {e}")

    async def add_trace_score(
        self,
        trace_context: TraceContext,
        *,
        name: str,
        value: float | str,
        comment: Optional[str] = None,
    ) -> None:
        """Adiciona uma pontuação ao trace."""
        if not isinstance(trace_context, _LangfuseTraceContext):
            return

        try:
            # Usar o método score_current_trace do SDK V3
            self._client.score_current_trace(
                name=name,
                value=value,
                comment=comment,
            )

        except Exception as e:
            self._logger.error(f"Erro ao adicionar pontuação: {e}")

    async def handle_error(
        self,
        trace_context: Optional[TraceContext],
        generation_context: Optional[GenerationContext],
        error: Exception,
        start_time: datetime,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Trata erros durante a execução."""
        try:
            error_metadata = dict(metadata) if metadata else {}
            error_metadata.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "latency_until_error": (
                        datetime.now() - start_time
                    ).total_seconds(),
                }
            )

            # Atualizar geração com erro se existir
            if generation_context and isinstance(
                generation_context, _LangfuseGenerationContext
            ):
                generation = generation_context.generation
                generation.update(
                    output={"error": str(error)},
                    metadata=error_metadata,
                    level="ERROR",
                )

            # Adicionar pontuações de erro ao trace
            if trace_context and isinstance(trace_context, _LangfuseTraceContext):
                # Pontuação de sucesso do trace
                await self.add_trace_score(
                    trace_context,
                    name="trace_success",
                    value=0.0,
                    comment=f"Error: {type(error).__name__} - {str(error)[:100]}",
                )

                # Categorizar erro
                error_category = self._categorize_error(error)
                await self.add_trace_score(
                    trace_context,
                    name="error_category",
                    value=error_category,
                    comment=f"Error classified as: {error_category}",
                )

                # Atualizar trace com erro
                await self.update_trace(
                    trace_context,
                    output_data={
                        "error": str(error),
                        "error_type": type(error).__name__,
                    },
                    success=False,
                    metadata=error_metadata,
                )

        except Exception as e:
            self._logger.error(f"Erro ao tratar exceção: {e}")

    def _categorize_error(self, error: Exception) -> str:
        """Categoriza o tipo de erro para análise."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        if "timeout" in error_str or "time" in error_type:
            return "timeout"
        elif "connection" in error_str or "network" in error_str:
            return "network"
        elif "auth" in error_str or "key" in error_str or "credential" in error_str:
            return "authentication"
        elif "limit" in error_str or "quota" in error_str or "rate" in error_str:
            return "rate_limit"
        elif "value" in error_type or "type" in error_type or "attribute" in error_type:
            return "validation"
        elif "memory" in error_str or "resource" in error_str:
            return "resource"
        else:
            return "other"

    async def flush(self) -> None:
        """Força o envio imediato de todos os eventos pendentes."""
        try:
            self._client.flush()
            self._logger.debug("Eventos enviados com sucesso para Langfuse")
        except Exception as e:
            self._logger.error(f"Erro ao enviar eventos para Langfuse: {e}")
