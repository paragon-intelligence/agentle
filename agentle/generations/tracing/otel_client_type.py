from agentle.generations.tracing.langfuse_otel_client import LangfuseOtelClient
from agentle.generations.tracing.no_op_otel_client import NoOpOtelClient


type OtelClientType = LangfuseOtelClient | NoOpOtelClient
