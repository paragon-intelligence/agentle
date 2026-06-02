"""
Complete enhanced API endpoint integration for Agentle framework.

This module provides comprehensive HTTP API endpoint support with:
- Multiple authentication methods
- Advanced retry strategies
- Circuit breaker pattern
- Rate limiting
- Response caching
- File uploads (multipart/form-data)
- Streaming responses
- Request/response hooks
- SSL/Proxy configuration
- and more...
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, MutableMapping, Sequence
from typing import Any, Literal

import aiohttp
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.apis.authentication import (
    AuthenticationBase,
    AuthenticationConfig,
    NoAuthentication,
)
from agentle.agents.apis.endpoint_parameter import EndpointParameter
from agentle.agents.apis.file_upload import FileUpload
from agentle.agents.apis.http_method import HTTPMethod
from agentle.agents.apis.parameter_location import ParameterLocation
from agentle.agents.apis.circuit_breaker import CircuitBreaker
from agentle.agents.apis.circuit_breaker_error import CircuitBreakerError
from agentle.agents.apis.rate_limiter import RateLimiter
from agentle.agents.apis.request_config import RequestConfig
from agentle.agents.apis.response_cache import ResponseCache
from agentle.agents.apis.retry_strategy import RetryStrategy
from agentle.generations.tools.tool import ApprovalPolicy, RiskLevel, Tool

logger = logging.getLogger(__name__)


class Endpoint(BaseModel):
    """
    Enhanced HTTP API endpoint with comprehensive features.

    Supports:
    - Multiple authentication methods
    - Advanced retry strategies with circuit breakers
    - Rate limiting and quota management
    - Response caching
    - File uploads
    - Streaming responses
    - Request/response hooks
    - SSL and proxy configuration
    """

    name: str = Field(description="Unique name for this endpoint")

    description: str = Field(
        description="Human-readable description of what this endpoint does"
    )

    call_condition: str | None = Field(
        description="Condition or context when the agent should call this endpoint",
        default=None,
    )

    url: str | None = Field(
        description="Complete URL for the endpoint (if not using base_url + path)",
        default=None,
    )

    path: str | None = Field(
        description="Path to append to base URL (when used with API class)",
        default=None,
    )

    method: HTTPMethod = Field(
        description="HTTP method for this endpoint", default=HTTPMethod.GET
    )

    parameters: Sequence[EndpointParameter] = Field(
        description="Parameters that can be passed to this endpoint",
        default_factory=list,
    )

    headers: MutableMapping[str, str] = Field(
        description="Additional headers for this endpoint", default_factory=dict
    )

    request_config: RequestConfig = Field(
        description="Request configuration for this endpoint",
        default_factory=RequestConfig,
    )

    auth_config: AuthenticationConfig | None = Field(
        description="Authentication configuration",
        default=None,
    )

    response_format: Literal["json", "text", "bytes", "stream", "xml"] = Field(
        description="Expected response format", default="json"
    )

    # File upload support
    supports_file_upload: bool = Field(
        description="Whether this endpoint supports file uploads", default=False
    )

    # Pagination support
    supports_pagination: bool = Field(
        description="Whether this endpoint supports pagination", default=False
    )
    pagination_param_name: str = Field(
        description="Name of pagination parameter", default="page"
    )
    pagination_style: Literal["page", "offset", "cursor"] = Field(
        description="Style of pagination", default="page"
    )

    # Response validation
    validate_response_schema: bool = Field(
        description="Whether to validate response against schema", default=False
    )
    response_schema: dict[str, Any] | None = Field(
        description="JSON schema for response validation", default=None
    )

    # Advanced features
    enable_hooks: bool = Field(
        description="Enable request/response hooks", default=False
    )

    approval_required: bool = Field(
        description="Whether this endpoint must be approved by a human before dispatch.",
        default=False,
    )

    approval_policy: ApprovalPolicy | None = Field(
        description="Approval policy name for human approval workflows.",
        default=None,
    )

    risk_level: RiskLevel | None = Field(
        description="Human-readable risk level for approval UIs and policy engines.",
        default=None,
    )

    approval_timeout_seconds: int | None = Field(
        description="How long an approval request may remain pending before expiring.",
        default=None,
    )

    approver_scope: dict[str, Any] | None = Field(
        description="Structured scope describing who may approve this endpoint call.",
        default=None,
    )

    display_template: str | None = Field(
        description="Optional template used by applications to display approval requests.",
        default=None,
    )

    redaction_schema: dict[str, Any] | None = Field(
        description="Optional schema describing how applications should redact arguments.",
        default=None,
    )

    approval_metadata: dict[str, Any] = Field(
        description="Application-specific metadata for human approval workflows.",
        default_factory=dict,
    )

    # Internal state (not serialized)
    _auth_handler: AuthenticationBase | None = None
    _circuit_breaker: CircuitBreaker | None = None
    _rate_limiter: RateLimiter | None = None
    _response_cache: ResponseCache | None = None

    def model_post_init(self, __context: Any) -> None:
        """Initialize internal components."""
        super().model_post_init(__context)

        # Initialize authentication handler
        if self.auth_config:
            self._auth_handler = self.auth_config.create_handler()
        else:
            self._auth_handler = NoAuthentication()

        # Initialize circuit breaker if enabled
        if self.request_config.enable_circuit_breaker:
            self._circuit_breaker = CircuitBreaker(self.request_config)

        # Initialize rate limiter if enabled
        if self.request_config.enable_rate_limiting:
            self._rate_limiter = RateLimiter(self.request_config)

        # Initialize cache if enabled
        if self.request_config.enable_caching:
            self._response_cache = ResponseCache(self.request_config)

    def get_full_url(self, base_url: str | None = None) -> str:
        """Get the complete URL for this endpoint."""
        if self.url:
            return self.url

        if base_url and self.path:
            return f"{base_url.rstrip('/')}/{self.path.lstrip('/')}"

        raise ValueError(
            f"Endpoint '{self.name}' must have either 'url' or 'path' (with base_url)"
        )

    def get_enhanced_description(self) -> str:
        """Get description enhanced with call condition."""
        base_desc = self.description

        if self.call_condition:
            return f"{base_desc}\n\nCall this endpoint: {self.call_condition}"

        return base_desc

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy."""
        base_delay = self.request_config.retry_delay

        if self.request_config.retry_strategy == RetryStrategy.CONSTANT:
            delay = base_delay

        elif self.request_config.retry_strategy == RetryStrategy.LINEAR:
            delay = base_delay * (attempt + 1)

        elif self.request_config.retry_strategy == RetryStrategy.EXPONENTIAL:
            delay = base_delay * (2**attempt)

        elif self.request_config.retry_strategy == RetryStrategy.FIBONACCI:
            # Calculate Fibonacci number for attempt
            fib = [1, 1]
            for _ in range(attempt):
                fib.append(fib[-1] + fib[-2])
            delay = base_delay * fib[-1]

        else:
            delay = base_delay

        # Add jitter (±20%)
        jitter = delay * 0.2 * (random.random() - 0.5) * 2
        delay = delay + jitter

        # Cap at 60 seconds
        return min(delay, 60.0)

    def _should_retry(
        self, response: aiohttp.ClientResponse | None, exception: Exception | None
    ) -> bool:
        """Determine if request should be retried."""
        # Retry on configured status codes
        if response and response.status in self.request_config.retry_on_status_codes:
            return True

        # Retry on exceptions if configured
        if exception and self.request_config.retry_on_exceptions:
            # Don't retry on certain exceptions
            if isinstance(exception, (asyncio.CancelledError, KeyboardInterrupt)):
                return False
            return True

        return False

    async def _parse_response(self, response: aiohttp.ClientResponse) -> Any:
        """Parse response based on format."""
        if self.response_format == "json":
            return await response.json()
        elif self.response_format == "text":
            return await response.text()
        elif self.response_format == "bytes":
            return await response.read()
        elif self.response_format == "xml":
            # Try to parse XML
            try:
                import xml.etree.ElementTree as ET

                text = await response.text()
                return ET.fromstring(text)
            except Exception:
                return await response.text()
        else:
            return await response.text()

    async def _handle_streaming_response(
        self, response: aiohttp.ClientResponse
    ) -> AsyncIterator[bytes]:
        """Handle streaming response."""
        async for chunk in response.content.iter_chunked(8192):
            yield chunk

    async def _validate_response(self, data: Any) -> Any:
        """Validate response against schema if configured."""
        if not self.validate_response_schema or not self.response_schema:
            return data

        try:
            import jsonschema

            jsonschema.validate(instance=data, schema=self.response_schema)
            return data
        except Exception as e:
            logger.warning(f"Response validation failed: {e}")
            return data

    async def make_request(
        self,
        base_url: str | None = None,
        global_headers: MutableMapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Internal method to make the HTTP request with all enhancements."""
        url = self.get_full_url(base_url)

        # Check rate limiter
        if self._rate_limiter:
            await self._rate_limiter.acquire()

        # Check cache (only for GET requests if configured)
        if (
            self._response_cache
            and self.method == HTTPMethod.GET
            and self.request_config.cache_only_get
        ):
            cached = await self._response_cache.get(url, kwargs)
            if cached is not None:
                logger.debug(f"Cache hit for {url}")
                return cached

        # Separate parameters by location
        query_params: dict[str, Any] = {}
        body_params: dict[str, Any] = {}
        header_params: dict[str, str] = {}
        path_params: dict[str, Any] = {}
        files: dict[str, FileUpload] = {}

        for param in self.parameters:
            param_name = param.name

            # Use provided value or default
            if param_name in kwargs:
                value = kwargs[param_name]
            elif param.default is not None:
                value = param.default
            elif param.required:
                raise ValueError(f"Required parameter '{param_name}' not provided")
            else:
                continue

            # Handle file uploads
            if isinstance(value, FileUpload):
                files[param_name] = value
                continue

            # Place parameter in appropriate location with proper type handling
            if param.location == ParameterLocation.QUERY:
                # Handle boolean conversion for query params
                if isinstance(value, bool):
                    # Convert Python bool to lowercase string for URL compatibility
                    query_params[param_name] = str(value).lower()
                else:
                    query_params[param_name] = value
            elif param.location == ParameterLocation.BODY:
                body_params[param_name] = value
            elif param.location == ParameterLocation.HEADER:
                # Convert to string for headers
                if isinstance(value, bool):
                    header_params[param_name] = str(value).lower()
                else:
                    header_params[param_name] = str(value)
            elif param.location == ParameterLocation.PATH:
                path_params[param_name] = value

        # Replace path parameters in URL
        for param_name, value in path_params.items():
            url = url.replace(f"{{{param_name}}}", str(value))

        # Merge headers
        headers = {}
        if global_headers:
            headers.update(global_headers)
        headers.update(self.headers)
        headers.update(header_params)

        # Apply authentication
        if self._auth_handler:
            await self._auth_handler.refresh_if_needed()
            await self._auth_handler.apply_auth(None, url, headers, query_params)  # type: ignore

        # Prepare connector kwargs (will be used to create fresh connector for each attempt)
        connector_kwargs: dict[str, Any] = {
            "limit": 10,
            "limit_per_host": 5,
            "ttl_dns_cache": 300,
        }

        if not self.request_config.verify_ssl:
            connector_kwargs["ssl"] = False

        # Prepare timeout
        timeout = aiohttp.ClientTimeout(
            total=self.request_config.timeout,
            connect=self.request_config.connect_timeout,
            sock_read=self.request_config.read_timeout,
        )

        # Define the request function for circuit breaker
        async def make_single_request() -> Any:
            """Make a single request attempt."""
            # Create a fresh connector for each request attempt to avoid "Session is closed" errors on retries
            connector = aiohttp.TCPConnector(**connector_kwargs)
            session = None
            try:
                session = aiohttp.ClientSession(connector=connector, timeout=timeout)
                # Prepare request kwargs
                request_kwargs: dict[str, Any] = {
                    "headers": headers,
                    "allow_redirects": self.request_config.follow_redirects,
                    "max_redirects": self.request_config.max_redirects,
                }

                if query_params:
                    request_kwargs["params"] = query_params

                # Handle different content types
                if files and self.supports_file_upload:
                    # Multipart form-data
                    form_data = aiohttp.FormData()
                    for key, file in files.items():
                        form_data.add_field(
                            key,
                            file.content,
                            filename=file.filename,
                            content_type=file.mime_type or "application/octet-stream",
                        )
                    for key, value in body_params.items():
                        form_data.add_field(key, str(value))
                    request_kwargs["data"] = form_data

                elif body_params and self.method in [
                    HTTPMethod.POST,
                    HTTPMethod.PUT,
                    HTTPMethod.PATCH,
                ]:
                    # JSON body
                    request_kwargs["json"] = body_params
                    if "Content-Type" not in headers:
                        headers["Content-Type"] = "application/json"

                # Proxy configuration
                if self.request_config.proxy_url:
                    request_kwargs["proxy"] = self.request_config.proxy_url
                    if self.request_config.proxy_auth:
                        request_kwargs["proxy_auth"] = aiohttp.BasicAuth(
                            *self.request_config.proxy_auth
                        )

                # Log request if enabled
                if self.request_config.enable_request_logging:
                    logger.info(f"Request: {self.method} {url}")
                    logger.debug(f"Headers: {headers}")
                    logger.debug(f"Params: {query_params}")

                # Make request
                async with session.request(
                    method=self.method.value, url=url, **request_kwargs
                ) as response:
                    # Log response if enabled
                    if self.request_config.enable_response_logging:
                        logger.info(f"Response: {response.status} from {url}")

                    # Handle HTTP errors
                    if response.status >= 400:
                        error_text = await response.text()

                        # Check for Retry-After header
                        if (
                            response.status == 429
                            and self.request_config.respect_retry_after
                        ):
                            retry_after = response.headers.get("Retry-After")
                            if retry_after:
                                try:
                                    wait_time = int(retry_after)
                                    logger.warning(
                                        f"Rate limited. Waiting {wait_time}s as per Retry-After header"
                                    )
                                    await asyncio.sleep(wait_time)
                                except ValueError:
                                    pass

                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"HTTP {response.status}: {error_text}",
                        )

                    # Handle streaming responses
                    if self.response_format == "stream":
                        chunks: list[bytes] = []
                        async for chunk in self._handle_streaming_response(response):
                            chunks.append(chunk)
                        return b"".join(chunks)

                    # Parse response
                    result = await self._parse_response(response)

                    # Validate response
                    result = await self._validate_response(result)

                    # Cache response if configured
                    if self._response_cache and self.method == HTTPMethod.GET:
                        await self._response_cache.set(url, kwargs, result)

                    return result
            finally:
                # Always close the session to prevent "Session is closed" errors on retries
                if session is not None:
                    await session.close()
                    # Give the connector time to close properly
                    await asyncio.sleep(0.01)

        # Execute with retries
        last_exception = None

        for attempt in range(self.request_config.max_retries + 1):
            try:
                # Execute with circuit breaker if enabled
                if self._circuit_breaker:
                    result = await self._circuit_breaker.call(make_single_request)
                else:
                    result = await make_single_request()

                return result

            except asyncio.CancelledError:
                logger.debug(f"Request to {url} was cancelled")
                raise

            except CircuitBreakerError:
                # Don't retry if circuit is open
                raise

            except Exception as e:
                last_exception = e

                # Check if we should retry
                should_retry = self._should_retry(None, e)

                if not should_retry or attempt >= self.request_config.max_retries:
                    break

                # Calculate delay and wait
                delay = self._calculate_retry_delay(attempt)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.request_config.max_retries + 1}). "
                    + f"Retrying in {delay:.2f}s: {str(e)}"
                )

                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    logger.debug("Sleep interrupted by cancellation")
                    raise

        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("All request attempts failed")

    def to_tool(
        self,
        base_url: str | None = None,
        global_headers: MutableMapping[str, str] | None = None,
    ) -> Tool[Any]:
        """Convert this endpoint to a Tool instance."""

        async def endpoint_callable(**kwargs: Any) -> Any:
            """Callable function for the tool."""
            return await self.make_request(
                base_url=base_url, global_headers=global_headers, **kwargs
            )

        # Create tool parameters from endpoint parameters
        tool_parameters: dict[str, object] = {}

        for param in self.parameters:
            if hasattr(param, "to_tool_parameter_schema"):
                # Use the parameter's own schema conversion method
                tool_parameters[param.name] = param.to_tool_parameter_schema()
            else:
                # Fallback for parameters without schema method
                param_info: dict[str, object] = {
                    "type": getattr(param, "param_type", "string") or "string",
                    "description": param.description,
                    "required": param.required,
                }

                if param.default is not None:
                    param_info["default"] = param.default

                if hasattr(param, "enum") and param.enum:
                    param_info["enum"] = list(param.enum)

                # Add constraints for number/primitive types
                if hasattr(param, "parameter_schema") and param.parameter_schema:
                    from agentle.agents.apis.primitive_schema import PrimitiveSchema

                    schema = param.parameter_schema
                    # Only PrimitiveSchema has minimum, maximum, format
                    if isinstance(schema, PrimitiveSchema):
                        if schema.minimum is not None:
                            param_info["minimum"] = schema.minimum
                        if schema.maximum is not None:
                            param_info["maximum"] = schema.maximum
                        if schema.format:
                            param_info["format"] = schema.format

                tool_parameters[param.name] = param_info

        tool_name = "_".join(self.name.lower().split())

        # Create the tool
        tool = Tool(
            name=tool_name,
            description=self.get_enhanced_description(),
            parameters=tool_parameters,
            approval_required=self.approval_required,
            approval_policy=self.approval_policy,
            risk_level=self.risk_level,
            approval_timeout_seconds=self.approval_timeout_seconds,
            approver_scope=self.approver_scope,
            display_template=self.display_template,
            redaction_schema=self.redaction_schema,
            approval_metadata=self.approval_metadata,
        )

        tool.set_callable_ref(endpoint_callable)
        try:
            tool._serialize_all_callables()
        except Exception:
            logger.debug(
                "Endpoint tool '%s' could not be serialized for suspension resume",
                tool_name,
                exc_info=True,
            )

        logger.debug(
            f"Created tool '{self.name}' with {len(tool_parameters)} parameters"
        )

        return tool
