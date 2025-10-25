# Changelog

## v0.9.13
fix(openrouter-adapter): Add debug logging for OpenRouter response

- Add debug logging to track raw OpenRouter API response
- Helps with troubleshooting and understanding API interactions
- Provides visibility into response data before parsing

## v0.9.12
fix(openrouter-adapter): Improve JSON response parsing with robust error handling

- Add more detailed error logging for JSON parsing failures
- Read response text before attempting JSON parsing to prevent body consumption
- Include response length in error log for better diagnostics
- Use `json.loads()` instead of `.json()` method for more explicit parsing
- Enhance error message with additional context about response parsing

## v0.9.11
fix(openrouter-adapter): Improve JSON response parsing with robust error handling

- Add comprehensive error handling for JSON parsing in OpenRouter response
- Capture and log detailed error information when JSON parsing fails
- Include response status code, content type, and preview in error logging
- Raise a descriptive ValueError with context when JSON parsing fails
- Enhance error traceability for debugging OpenRouter API interactions

## v0.9.10
feat(docx-parser, pdf-parser): Implement max_output_tokens for response generation

- Add max_output_tokens attribute to DocxFileParser and PDFFileParser for controlling token generation in responses
- Update relevant methods to utilize max_output_tokens in generation configuration

## v0.9.9
fix(openrouter-adapter): Improve tool call argument parsing with robust error handling

- Add comprehensive JSON parsing error handling for tool call arguments
- Implement fallback mechanism to extract first valid JSON object when parsing fails
- Log detailed warnings and errors for malformed JSON input
- Prevent adapter from breaking when encountering invalid JSON in tool call arguments
- Ensure graceful degradation by using empty dict if parsing completely fails

## v0.9.8
fix(whatsapp): Improve phone number ID extraction for WhatsApp webhook payloads

- Update phone number ID extraction logic for linked IDs
- Set remoteJid to remoteJidAlt when "@lid" is present in key
- Ensure consistent phone number ID handling for different WhatsApp message types

## v0.9.7
fix(whatsapp): not returning early when "lid" is present.

## v0.9.6
fix(stdio-mcp-server): Improve JSON message parsing and connection state handling

- Enhance message parsing to skip non-JSON lines in stdout
- Add debug logging for skipped lines and invalid JSON
- Modify connection state checks to allow requests/notifications during initialization
- Prevent potential errors from processing non-JSON output from MCP servers
- Improve error handling and logging for message decoding

## v0.9.5
fix(whatsapp): Improve WhatsApp webhook payload phone number ID extraction

- Update phone number ID extraction logic for WhatsApp webhook payloads
- Add fallback extraction method using remoteJid when remoteJidAlt is not available
- Ensure consistent phone number ID extraction across different webhook scenarios
- Prevent potential null or incomplete phone number ID assignments

## v0.9.4
feat(whatsapp): Improve WhatsApp webhook payload and key handling

- Add remoteJidAlt field to Key model for enhanced JID parsing
- Update phone number ID extraction logic in webhook payload processing
- Modify message handling to use remoteJidAlt for linked WhatsApp accounts
- Enhance robustness of JID parsing for different WhatsApp account types


## v0.9.3

### 🚀 **New Responses API (Experimental)**
- **feat(responses)**: Brand new experimental Responses API with comprehensive OpenRouter integration
  - Complete OpenRouter API compatibility with streaming support and structured outputs
  - Advanced function tool management with `FunctionCallStore` for tracking and executing function calls
  - Enhanced `FunctionTool` class with robust type safety and parameter conversion
  - `ToolPair` system for managing tool definitions and call history
  - Support for both synchronous and asynchronous tool execution
  - Comprehensive response streaming with `AsyncStream` implementation
  - Integration with `orjson` for improved JSON handling performance
  - Support for reasoning, background processing, and conversation management
  - Factory methods for easy instantiation (`openrouter()`, `openai()`)

### 🌐 **Web Extraction & Automation**
- **feat(web)**: Complete web extraction and automation system
  - **Extractor class**: Intelligent web content extraction with structured output support
  - **Browser automation**: Full Playwright integration with configurable options
  - **Action system**: Comprehensive set of web actions including:
    - Click, scroll, write text, take screenshots
    - Generate PDFs, execute JavaScript, press keys
    - Wait for conditions, viewport management
  - **Extraction preferences**: Advanced configuration options:
    - Geolocation support, mobile/desktop modes
    - Ad blocking, TLS verification control
    - Custom headers, timeouts, and proxy settings
    - Content filtering (include/exclude tags)
  - **Structured prompts**: AI-powered content analysis with markdown conversion
  - **Error handling**: Graceful handling of invalid URLs and extraction failures

### 🔧 **Enhanced Assistant & Tracing**
- **feat(assistant)**: Enhanced Assistant model with new fields and Responder integration
- **feat(tracing)**: Improved tracing system with BaseModel inheritance
  - Refactored `LangfuseOtelClient` and `NoOpOtelClient` to inherit from BaseModel
  - Enhanced type safety and error handling across tracing components
  - Better parameter conversion and callable management

### 🛠️ **Function Call Management**
- **feat(responses)**: Advanced function call management system
  - Methods for retrieving, removing, and clearing function calls
  - Enhanced tool presence checking and call count retrieval
  - Support for multiple function calls with improved access patterns
  - Example usage documentation for better developer experience

### 🐛 **Bug Fixes & Improvements**
- **fix(responses)**: Multiple fixes for structured outputs and JSON schema handling
  - Proper JSON schema "additionalProperties" setting and retrieval
  - Improved handling of incomplete structured outputs with clearer error messaging
  - Fixed tool calls and response object text format handling
  - Streamlined response object creation and event handling
- **refactor(file)**: Commented out MIME type validation for broader file type acceptance
- **fix(responses)**: Enhanced OpenRouter example with updated max output tokens

### 📈 **Performance & Developer Experience**
- **feat(responses)**: AsyncStream implementation for efficient streaming responses
- **refactor(responses, web)**: Simplified JSON response handling and improved output parsing
- **feat(utils)**: Added `needs` utility for dependency management
- Enhanced error messaging and debugging capabilities across all new features

---

## v0.9.2

- feat(agent): cache instructions in Agent class and Developer message for the providers that supports it (cache ephemeral)


## v0.9.1

### 🎯 **Pricing Integration**
- **feat(generations)**: Added `Pricing` model to `Generation` class with `input_pricing`, `output_pricing`, and `total_pricing` fields
- **feat(providers)**: All major providers now automatically populate pricing information in `Generation` objects
  - OpenRouter: Dynamic pricing from API with automatic per-token to per-million conversion
  - OpenAI: Static pricing table with comprehensive model coverage
  - Google (Gemini): Static pricing for all Gemini models
  - Cerebras: Static pricing for Llama models
  - Amazon Bedrock: Static pricing for Claude and other models
- **feat(adapters)**: Added async `adapt_async()` methods to all provider adapters for pricing calculation
- **fix(openrouter)**: Corrected pricing calculation - OpenRouter returns per-token prices, now properly converted to per-million-tokens
- Pricing is calculated using the abstract methods `price_per_million_tokens_input()` and `price_per_million_tokens_output()`
- Costs are rounded to 8 decimal places for precision
- Graceful fallback to empty `Pricing` object if calculation fails

### 🐛 **Bug Fixes**
- fix(openrouter): when fallback models are used, use "models" in the requests, not "model"
- fix(openrouter): accept single string in `with_fallback_models()` method (was treating string as iterable)

### 🚨 **Error Handling**
- feat(openrouter): comprehensive custom exception hierarchy with detailed error messages
  - Created 16+ custom exception classes for all documented OpenRouter errors
  - Each exception includes clear descriptions, possible causes, and actionable solutions
  - Automatic error parsing from response status codes and error messages
  - Examples: `ModelNotFoundError`, `InsufficientCreditsError`, `ContextLengthExceededError`, etc.
  - See `examples/openrouter_error_handling.py` for usage examples

## v0.9.0 🚀

**Release Date:** October 18, 2024

We're thrilled to announce **Agentle v0.9.0**, a major milestone packed with powerful new features, critical fixes, and performance improvements. This release represents months of refinement across our parsing, provider, and agent ecosystems.

---

### ✨ **Highlights**

#### 🛡️ **Guardrails System**
- **NEW:** Comprehensive guardrails integration for safer AI interactions
- **Tool Leakage Validator:** Prevent sensitive tool information from leaking in responses

#### 🔌 **OpenRouter Provider**
- **Full OpenRouter integration** with streaming support and structured outputs
- **Dynamic price fetching** for cost-aware model selection
- **Model fallbacks** for improved reliability
- **Factory methods** for easier instantiation
- Enhanced tool adapter with complex type expansion in JSON Schema conversion
- Configurable httpx client timeout for generation tasks

#### 📄 **Document Parsing Revolution**
- **Native PDF processing** with PyMuPDF for superior performance
- **RTF Document Parser** - brand new format support
- **Native DOCX processing** option for faster document handling
- Enhanced PDF parsing with encrypted file support and structured exception handling
- PPTX parsing improvements with legacy conversion support
- Legacy `.doc` file conversion with improved error handling
- Optimized archive processing (ZIP, RAR) with better error messaging
- Page screenshot optimization to reduce per-image analysis costs
- Configurable `render_scale` for PDF screenshot quality control
- `max_concurrent_pages` field for PDF parsing concurrency control

#### 🧠 **Embeddings & Vector Stores**
- **DeepInfra Embedding Provider** - OpenAI-compatible API integration
- Batch embedding generation methods for async processing
- Enhanced async embedding methods with None value handling
- Load balancer implementation (DuckDB and in-memory) with provider quotas and ranking

#### 💬 **WhatsApp Enhancements**
- Enhanced markdown formatting (headers, tables, blockquotes, horizontal rules)
- Improved message splitting with list item grouping
- `send_message` method for independent message sending
- Multiple callback support with semantic type aliases
- Better table formatting (vertical lists for improved readability)

#### 🔧 **Developer Experience**
- **Circuit breaker** implementation with DuckDB backend
- Static knowledge `from_text` class method for easier content creation
- `append_content` method for dynamic ParsedFile content addition
- Provider ID implementation across GenerationProvider subclasses
- Fallback models parameter for provider resilience
- Enhanced file validation with MIME type and content checks

---

### 🐛 **Bug Fixes**

#### WhatsApp
- Fixed Brazilian phone number formatting after Meta's privacy update
- Proper handling of Evolution API typed dicts (converted to BaseModels)
- Fixed `from_number` extraction from sender
- Corrected buffer handling for Evolution API fields
- Improved thinking tag removal logic
- Enhanced race condition protection and batch timing
- Rate limit checks now happen before message processing

#### Agents & Tools
- Fixed assistant message context handling before tool processing
- Removed unnecessary cast in `_stream_with_tools` function
- Corrected property "text" update logic
- Fixed duplicate tools in vector stores
- Resolved duplicated parameter issues

#### Parsing & Generation
- Fixed PDF extraction error handling
- Handled cases where no extraction is returned from PDF provider
- Updated PyMuPDF import for better compatibility
- Normalized model names (removed 'google/' prefix)
- Fixed pickling errors in RAG search by avoiding callable serialization
- Improved logging calls

---

### ⚡ **Performance Improvements**

- **Single page screenshot per page** when images are present (avoiding duplicate OCR)
- Optimized PDF page processing with PyMuPDF
- Eliminated redundant temp file creation in archive processing
- Better markdown parsing performance
- Fail-fast on DOCX→PDF conversion errors

---

### 🔄 **Refactoring & Code Quality**

- Converted Evolution API typed dicts to BaseModels for better type safety
- Renamed 'agent' parameters to 'provider' for consistency
- Renamed 'chunk_tokens' to 'output_tokens' with new 'input_tokens' property
- Protocol classes follow single underscore convention
- Cleaned up whitespace and improved exception handling
- Streamlined parser class retrieval
- Made Langfuse a parameter for better flexibility
- Enhanced CallbackWithContext with token propagation

---

### 📚 **Documentation & Testing**

- Added test for uppercase file extension handling
- Testing failover Generation Provider
- Updated examples with lightweight implementations
- Comprehensive changelog updates

---

### 🙏 **Thank You**

This release wouldn't be possible without the dedication of our contributors and the valuable feedback from our community. We're committed to making Agentle the most powerful and reliable agentic AI framework.

**Upgrade now and experience the future of AI agents!**

```bash
pip install --upgrade agentle
```

---

## v0.8.68

- feat(openrouter) streaming with structured outputs
