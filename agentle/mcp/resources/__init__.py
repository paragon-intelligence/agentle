"""
# Resources Module

## Overview

The `resources` module is a core component that implements the Model Context Protocol (MCP) resources primitive. Resources allow servers to expose data and content that can be read by clients and used as context for LLM interactions.

Resources are designed to be **application-controlled**, meaning that the client application can decide how and when they should be used. Different MCP clients may handle resources differently:

- Some clients require users to explicitly select resources before they can be used
- Others might automatically select resources based on heuristics
- Some implementations may allow the AI model itself to determine which resources to use

## Resource Types

The module supports two primary resource types:

### Text Resources

```python
class TextResource:
    \"\"\"
    Text resources contain UTF-8 encoded text data.

    Suitable for:
    - Source code
    - Configuration files
    - Log files
    - JSON/XML data
    - Plain text
    \"\"\"
```

### Binary Resources

```python
class BinaryResource:
    \"\"\"
    Binary resources contain raw binary data encoded in base64.

    Suitable for:
    - Images
    - PDFs
    - Audio files
    - Video files
    - Other non-text formats
    \"\"\"
```

## Resource Identification

Resources are identified using URIs that follow this format:

```
[protocol]://[host]/[path]
```

Examples:
- `file:///home/user/documents/report.pdf`
- `postgres://database/customers/schema`
- `screen://localhost/display1`

The protocol and path structure is defined by the MCP server implementation. Servers can define their own custom URI schemes.

## Resource Discovery

The module provides two main methods for resource discovery:

### Direct Resources

```python
def list_resources():
    \"\"\"
    Returns a list of concrete resources available from the server.

    Returns:
        List[Dict]: List of resource objects with the following structure:
            {
                "uri": str,           # Unique identifier for the resource
                "name": str,          # Human-readable name
                "description": str,   # Optional description
                "mimeType": str       # Optional MIME type
            }
    \"\"\"
```

### Resource Templates

```python
def list_resource_templates():
    \"\"\"
    Returns a list of URI templates for dynamic resources.

    Templates follow RFC 6570 for URI template specification.

    Returns:
        List[Dict]: List of template objects with the following structure:
            {
                "uriTemplate": str,   # URI template following RFC 6570
                "name": str,          # Human-readable name for this type
                "description": str,   # Optional description
                "mimeType": str       # Optional MIME type for all matching resources
            }
    \"\"\"
```

## Reading Resources

```python
def read_resource(uri):
    \"\"\"
    Reads the content of a resource identified by the given URI.

    Args:
        uri (str): The URI of the resource to read

    Returns:
        Dict: Response with resource contents:
            {
                "contents": [
                    {
                        "uri": str,        # The URI of the resource
                        "mimeType": str,   # Optional MIME type

                        # One of:
                        "text": str,       # For text resources
                        "blob": str        # For binary resources (base64 encoded)
                    }
                ]
            }

    Notes:
        Servers may return multiple resources in response to one request.
        This could be used, for example, to return a list of files inside
        a directory when the directory is read.
    \"\"\"
```

## Resource Updates

The module supports real-time updates for resources through two mechanisms:

### List Changes

```python
def register_list_change_handler(callback):
    \"\"\"
    Registers a callback to be notified when the list of available resources changes.

    Args:
        callback (callable): Function to call when resource list changes
    \"\"\"
```

### Content Changes

```python
def subscribe_to_resource(uri):
    \"\"\"
    Subscribes to updates for a specific resource.

    Args:
        uri (str): The URI of the resource to subscribe to

    Returns:
        bool: True if subscription was successful
    \"\"\"

def unsubscribe_from_resource(uri):
    \"\"\"
    Unsubscribes from updates for a specific resource.

    Args:
        uri (str): The URI of the resource to unsubscribe from

    Returns:
        bool: True if unsubscription was successful
    \"\"\"

def register_resource_update_handler(callback):
    \"\"\"
    Registers a callback to be notified when a subscribed resource changes.

    Args:
        callback (callable): Function to call when resource content changes.
            The callback will receive the URI of the changed resource.
    \"\"\"
```

## Example Usage

```python
from mcp import Server
from mcp.resources import TextResource, BinaryResource

# Create server with resources capability
server = Server(
    name="example-server",
    version="1.0.0",
    capabilities={"resources": {}}
)

# Implement resource handlers
@server.handler("resources/list")
async def handle_list_resources(request):
    return {
        "resources": [
            {
                "uri": "file:///logs/app.log",
                "name": "Application Logs",
                "mimeType": "text/plain"
            }
        ]
    }

@server.handler("resources/read")
async def handle_read_resource(request):
    uri = request.params.get("uri")

    if uri == "file:///logs/app.log":
        log_contents = await read_log_file()
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": log_contents
                }
            ]
        }

    raise ValueError("Resource not found")
```

## Best Practices

When implementing resource support:

1. Use clear, descriptive resource names and URIs
2. Include helpful descriptions to guide LLM understanding
3. Set appropriate MIME types when known
4. Implement resource templates for dynamic content
5. Use subscriptions for frequently changing resources
6. Handle errors gracefully with clear error messages
7. Consider pagination for large resource lists
8. Cache resource contents when appropriate
9. Validate URIs before processing
10. Document your custom URI schemes

## Security Considerations

When exposing resources:

- Validate all resource URIs
- Implement appropriate access controls
- Sanitize file paths to prevent directory traversal
- Be cautious with binary data handling
- Consider rate limiting for resource reads
- Audit resource access
- Encrypt sensitive data in transit
- Validate MIME types
- Implement timeouts for long-running reads
- Handle resource cleanup appropriately
"""
