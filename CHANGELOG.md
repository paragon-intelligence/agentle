# Changelog

## v0.2.0

### ✨ New Features

- **DbPromptProvider** - Retrieve prompts from PostgreSQL databases
- **Enhanced Langfuse Scoring** - Added new scores for improved filtering in the Langfuse UI
- **New MCP Server Types**
  - `StreamableHTTPMCPServer` for HTTP-based streaming
  - `StdioMCPServer` for standard I/O operations
- **Prompt Concatenation** - Added `__add__` method to `Prompt` class for easy prompt combination

### 🔧 Performance Improvements

- **Asyncio Task Scheduling** - Implemented to deliver generations faster, saving approximately 0.5 seconds per request

### 🐛 Bug Fixes

- Fixed `FailoverGenerationProvider` instantiation issue caused by missing method implementation in superclass
- Resolved price display bug in Langfuse interface for values greater than $0.0

### 🧹 Maintenance

- Fixed various linter errors throughout the codebase