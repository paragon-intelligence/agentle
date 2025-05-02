import json
from collections.abc import Callable
from typing import Any, Literal
import uuid

from rsb.adapters.adapter import Adapter

from agentle.agents.agent import Agent
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.user_message import UserMessage


class AgentToStreamlit(Adapter[Agent, "Callable[[], None]"]):
    title: str | None
    description: str | None
    initial_mode: Literal["dev", "presentation"]

    def __init__(
        self,
        title: str | None = None,
        description: str | None = None,
        initial_mode: Literal["dev", "presentation"] = "presentation",
    ):
        self.title = title
        self.description = description
        self.initial_mode = initial_mode

    def adapt(self, _f: Agent) -> Callable[[], None]:
        """
        Creates a Streamlit app that provides a chat interface to interact with the agent.

        This method returns a function that can be executed as a Streamlit app.
        The returned app provides a chat interface where users can interact with the agent,
        view the conversation history, and switch between development and presentation modes.

        Dev mode shows detailed information useful for developers, including raw response data,
        token usage, and parsed outputs. Presentation mode provides a clean interface for
        demonstrating the agent's capabilities.

        The interface supports both text and file inputs, including images.

        Args:
            title: Optional title for the Streamlit app (defaults to agent name)
            description: Optional description for the Streamlit app (defaults to agent description)
            initial_mode: The initial display mode ("dev" or "presentation", defaults to "presentation")

        Returns:
            Callable[[], None]: A function that can be executed as a Streamlit app

        Example:
            ```python
            from agentle.agents.agent import Agent
            from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

            # Create an agent
            agent = Agent(
                generation_provider=GoogleGenerationProvider(),
                model="gemini-2.0-flash",
                instructions="You are a helpful assistant."
            )

            # Get the Streamlit app function
            app = agent.to_streamlit_app(title="My Assistant")

            # Save this as app.py and run with: streamlit run app.py
            app()
            ```
        """
        agent = _f

        app_title = self.title or f"{agent.name} Agent"
        app_description = self.description or agent.description

        def _format_tool_call(tool_call: ToolExecutionSuggestion) -> str:
            """Format a tool call for display in the UI."""
            args_str = json.dumps(tool_call.args, indent=2)
            return f"**Tool Call:** `{tool_call.tool_name}`\n```json\n{args_str}\n```"

        def _streamlit_app() -> None:
            from typing import cast

            try:
                import streamlit as st
            except ImportError:
                print(
                    "Error: Streamlit is required for this feature. "
                    + "Please install it with: pip install streamlit"
                )
                return

            # Setup page config
            st.set_page_config(
                page_title=app_title,
                page_icon="🤖",
                layout="wide",
                initial_sidebar_state="expanded",
            )

            # Initialize session state
            if "messages" not in st.session_state:
                st.session_state.messages = []  # List of message dictionaries
            if "display_mode" not in st.session_state:
                st.session_state.display_mode = self.initial_mode  # str
            if "token_usage" not in st.session_state:
                st.session_state.token_usage = []  # List of tuples (prompt_tokens, completion_tokens)
            if "uploaded_files" not in st.session_state:
                st.session_state.uploaded_files = []  # List of file dictionaries

            # Sidebar for settings and developer tools
            with st.sidebar:
                st.title("⚙️ Settings")

                # Display mode selector
                st.session_state.display_mode = st.selectbox(
                    "Display Mode",
                    ["presentation", "dev"],
                    index=0 if st.session_state.display_mode == "presentation" else 1,
                )

                # Agent information
                st.subheader("Agent Info")
                st.write(f"**Name:** {agent.name}")
                st.write(f"**Model:** {agent.model or 'Not specified'}")

                if agent.has_tools():
                    st.subheader("Available Tools")
                    for tool in agent.tools:
                        tool_name = getattr(tool, "name", str(tool))
                        st.write(f"- {tool_name}")

                # In dev mode, show more details
                if st.session_state.display_mode == "dev":
                    st.subheader("Usage Statistics")
                    if st.session_state.token_usage:
                        # Extract token usage from session state with proper type safety
                        token_usage_list: list[tuple[int, int]] = []
                        for raw_item in st.session_state.token_usage:
                            # Ensure we are working with tuples coming from the usage list
                            if not isinstance(raw_item, tuple):
                                continue

                            # Safely cast to a tuple of unknown objects for further checks
                            item_tuple = cast(tuple[Any, ...], raw_item)

                            if len(item_tuple) < 2:
                                continue

                            # Convert the first two positions to integers (prompt and completion tokens)
                            try:
                                prompt_tok = (
                                    int(item_tuple[0])
                                    if isinstance(item_tuple[0], (int, float))
                                    else 0
                                )
                                completion_tok = (
                                    int(item_tuple[1])
                                    if isinstance(item_tuple[1], (int, float))
                                    else 0
                                )
                                token_usage_list.append((prompt_tok, completion_tok))
                            except (IndexError, TypeError):
                                # Skip malformed usage tuples
                                continue

                        # Calculate totals with explicit typing
                        total_prompt = 0
                        total_completion = 0
                        for prompt_tok, completion_tok in token_usage_list:
                            total_prompt += prompt_tok
                            total_completion += completion_tok

                        total = total_prompt + total_completion

                        st.write(f"**Total Prompt Tokens:** {total_prompt}")
                        st.write(f"**Total Completion Tokens:** {total_completion}")
                        st.write(f"**Total Tokens:** {total}")
                    else:
                        st.write("No usage data yet.")

                    if hasattr(agent, "config"):
                        st.subheader("Configuration")
                        try:
                            if hasattr(agent.config, "model_dump") and callable(
                                getattr(agent.config, "model_dump")
                            ):
                                config_data = getattr(agent.config, "model_dump")()
                                st.json(json.dumps(config_data))
                            elif hasattr(agent.config, "__dict__"):
                                st.json(json.dumps(agent.config.__dict__, default=str))
                            else:
                                st.json(
                                    json.dumps(
                                        {"config": str(agent.config)}, default=str
                                    )
                                )
                        except Exception:
                            st.write(f"Config: {str(agent.config)}")

                # Add a clear button
                if st.button("Clear Conversation"):
                    st.session_state.messages = []
                    st.session_state.token_usage = []
                    st.session_state.uploaded_files = []
                    st.rerun()

            # Main chat interface
            st.title(app_title)
            st.write(app_description)

            # Display chat messages
            message_container = st.container()
            with message_container:
                for message in st.session_state.messages:
                    if not isinstance(message, dict):
                        continue

                    role = message.get("role", "")
                    content = message.get("content", "")
                    metadata = message.get("metadata", {})

                    if role == "user":
                        with st.chat_message("user", avatar="👤"):
                            st.write(content)
                            # Display file attachments if any
                            if (
                                isinstance(metadata, dict)
                                and "files" in metadata
                                and isinstance(metadata["files"], list)
                            ):
                                for _file_data_any in metadata["files"]:
                                    if not isinstance(_file_data_any, dict):
                                        continue

                                    file_data: dict[str, Any] = cast(
                                        dict[str, Any], _file_data_any
                                    )

                                    mime_type = str(file_data.get("mime_type", ""))
                                    if mime_type.startswith("image/"):
                                        # Handle image data safely
                                        image_data = file_data.get("data", b"")
                                        if isinstance(image_data, bytes) and image_data:
                                            st.image(image_data)
                                    else:
                                        # Get file data with proper type checking
                                        file_bytes = file_data.get("data", b"")
                                        if not isinstance(file_bytes, bytes):
                                            file_bytes = b""

                                        # Use explicit type checking for strings
                                        name_value = file_data.get("name", "file")
                                        file_name = (
                                            name_value
                                            if isinstance(name_value, str)
                                            else "file"
                                        )

                                        mime_value = file_data.get(
                                            "mime_type", "application/octet-stream"
                                        )
                                        file_mime = (
                                            mime_value
                                            if isinstance(mime_value, str)
                                            else "application/octet-stream"
                                        )

                                        st.download_button(
                                            label=f"📎 {file_name}",
                                            data=file_bytes,
                                            file_name=file_name,
                                            mime=file_mime,
                                        )
                    elif role == "assistant":
                        with st.chat_message("assistant", avatar="🤖"):
                            st.write(content)

                            # Show tool calls if available and in dev mode
                            if (
                                st.session_state.display_mode == "dev"
                                and isinstance(metadata, dict)
                                and "tool_calls" in metadata
                                and isinstance(metadata["tool_calls"], list)
                            ):
                                for tool_call in metadata["tool_calls"]:
                                    if not isinstance(tool_call, dict):
                                        continue

                                    if "tool_name" in tool_call and "args" in tool_call:
                                        # Create safe versions of all parameters with proper type checking
                                        tool_name_value = tool_call.get("tool_name", "")
                                        tool_name = (
                                            tool_name_value
                                            if isinstance(tool_name_value, str)
                                            else ""
                                        )

                                        # Get ID with type safety
                                        id_value = tool_call.get("id", None)
                                        if not isinstance(id_value, str):
                                            id_value = str(uuid.uuid4())
                                        call_id = id_value

                                        # Convert args to a safe dictionary with explicit type checking
                                        args_dict: dict[str, object] = {}
                                        args_value = tool_call.get("args")
                                        if isinstance(args_value, dict):
                                            for k, v in args_value.items():
                                                if isinstance(k, str):
                                                    args_dict[k] = cast(object, v)
                                                else:
                                                    # Convert non-string keys to strings
                                                    try:
                                                        args_dict[
                                                            str(cast(object, k))
                                                        ] = v
                                                    except (ValueError, TypeError):
                                                        pass

                                        # Create a properly typed tool execution suggestion
                                        try:
                                            tc_obj = ToolExecutionSuggestion(
                                                tool_name=tool_name,
                                                args=args_dict,
                                                id=call_id,
                                            )
                                            st.markdown(_format_tool_call(tc_obj))
                                        except (ValueError, TypeError) as e:
                                            st.warning(
                                                f"Could not format tool call: {e}"
                                            )

                                        # If we have a result, display it
                                        if "result" in tool_call:
                                            st.write("**Result:**")
                                            result_value: object = tool_call.get(
                                                "result", ""
                                            )
                                            result_str: str = (
                                                str(cast(object, result_value))
                                                if result_value is not None
                                                else ""
                                            )
                                            st.code(result_str, language="python")

                            # Show parsed data if available and in dev mode
                            if (
                                st.session_state.display_mode == "dev"
                                and isinstance(metadata, dict)
                                and "parsed" in metadata
                                and metadata["parsed"] is not None
                            ):
                                st.write("**Parsed Output:**")
                                try:
                                    parsed_json = json.dumps(
                                        cast(object, metadata.get("parsed", {})),
                                        default=str,
                                    )
                                    st.json(parsed_json)
                                except Exception:
                                    st.code(
                                        str(cast(object, metadata.get("parsed", "")))
                                    )

            # File upload area
            uploaded_file = st.file_uploader(
                "Upload a file or image",
                type=None,
                key="file_uploader",
                accept_multiple_files=False,
            )
            if uploaded_file is not None:
                file_bytes = uploaded_file.getvalue()
                mime_type = uploaded_file.type or "application/octet-stream"

                # Only add if it's valid bytes
                if not isinstance(file_bytes, (bytearray, memoryview)):
                    st.session_state.uploaded_files.append(
                        {
                            "name": uploaded_file.name,
                            "data": file_bytes,
                            "mime_type": mime_type,
                        }
                    )

                    # Preview file if it's an image
                    if mime_type.startswith("image/"):
                        st.image(file_bytes, caption=f"Uploaded: {uploaded_file.name}")
                    else:
                        st.success(
                            f"File uploaded: {uploaded_file.name} ({len(file_bytes)} bytes)"
                        )

                # Clear the uploader
                st.session_state["file_uploader"] = None

            # Input area
            user_input = st.chat_input("Type your message here...")

            if user_input:
                # Prepare message metadata
                metadata = {}

                # Include uploaded files if any
                if st.session_state.uploaded_files:
                    # Make a deep copy to avoid reference issues
                    metadata["files"] = []

                    for _upload_file_any in st.session_state.uploaded_files:
                        if not isinstance(_upload_file_any, dict):
                            continue

                        upload_file: dict[str, Any] = cast(
                            dict[str, Any], _upload_file_any
                        )

                        # Create a new dictionary with valid types
                        safe_file_data = {
                            "name": str(upload_file.get("name", "file")),
                            "data": upload_file.get("data", b"")
                            if isinstance(upload_file.get("data"), bytes)
                            else b"",
                            "mime_type": str(
                                upload_file.get("mime_type", "application/octet-stream")
                            ),
                        }
                        metadata["files"].append(safe_file_data)
                    # Clear the files after adding them to the message
                    st.session_state.uploaded_files = []

                # Add user message to chat
                st.session_state.messages.append(
                    {"role": "user", "content": user_input, "metadata": metadata}
                )

                # Display user message immediately
                with st.chat_message("user", avatar="👤"):
                    st.write(user_input)
                    # Display files if present
                    if "files" in metadata and isinstance(metadata["files"], list):
                        for _meta_file_any in metadata["files"]:
                            if not isinstance(_meta_file_any, dict):
                                continue

                            meta_file: dict[str, Any] = cast(
                                dict[str, Any], _meta_file_any
                            )

                            # Get data with type checking
                            file_bytes = meta_file.get("data", b"")
                            if not isinstance(file_bytes, bytes):
                                continue

                            mime_type = str(
                                meta_file.get("mime_type", "application/octet-stream")
                            )

                            # Use explicit type checking for strings
                            name_value = meta_file.get("name", "file")
                            file_name = (
                                name_value if isinstance(name_value, str) else "file"
                            )

                            mime_value = meta_file.get(
                                "mime_type", "application/octet-stream"
                            )
                            file_mime = (
                                mime_value
                                if isinstance(mime_value, str)
                                else "application/octet-stream"
                            )

                            st.download_button(
                                label=f"📎 {file_name}",
                                data=file_bytes,
                                file_name=file_name,
                                mime=file_mime,
                            )

                # Display a spinner while waiting for the agent's response
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking..."):
                        # Prepare input for the agent
                        agent_input: UserMessage | str
                        if (
                            "files" in metadata
                            and isinstance(metadata["files"], list)
                            and metadata["files"]
                        ):
                            # Create parts for the user message
                            parts = [TextPart(text=user_input)]

                            # Add file parts
                            for file_data in metadata["files"]:
                                if not isinstance(file_data, dict):
                                    continue

                                # Get data with type checking
                                file_bytes = file_data.get("data", b"")
                                if not isinstance(file_bytes, bytes):
                                    continue

                                mime_type = str(
                                    file_data.get(
                                        "mime_type", "application/octet-stream"
                                    )
                                )

                                try:
                                    file_part = FilePart(
                                        data=file_bytes, mime_type=mime_type
                                    )
                                    # Create a new list with the correct types
                                    # This is safer than using append which can cause type issues
                                    parts = parts + [file_part]  # type: ignore
                                except ValueError:
                                    # If mime type is not valid, skip this file
                                    st.warning(
                                        f"Skipping file with invalid MIME type: {mime_type}"
                                    )
                                    continue

                            agent_input = UserMessage(parts=parts)
                        else:
                            # Simple text input
                            agent_input = user_input

                        # Run the agent
                        try:
                            with agent.with_mcp_servers():
                                result = agent.run(agent_input)

                                # Extract information from the result
                                generation = result.generation
                                response_text = generation.text
                                tool_calls = (
                                    generation.tool_calls
                                    if hasattr(generation, "tool_calls")
                                    else []
                                )
                                parsed = result.parsed

                                # Update token usage stats
                                if hasattr(generation, "usage"):
                                    st.session_state.token_usage.append(
                                        (
                                            generation.usage.prompt_tokens,
                                            generation.usage.completion_tokens,
                                        )
                                    )

                                # Prepare metadata for storage
                                response_metadata = {}
                                if tool_calls:
                                    tool_call_data = []
                                    for tc in tool_calls:
                                        tc_data = {
                                            "tool_name": tc.tool_name,
                                            "args": tc.args,
                                            "id": tc.id,
                                        }
                                        tool_call_data.append(tc_data)
                                    response_metadata["tool_calls"] = tool_call_data

                                if parsed is not None:
                                    try:
                                        # Try to convert to a dict for storage
                                        if hasattr(parsed, "model_dump") and callable(
                                            getattr(parsed, "model_dump")
                                        ):
                                            response_metadata["parsed"] = getattr(
                                                parsed, "model_dump"
                                            )()
                                        elif hasattr(parsed, "__dict__"):
                                            response_metadata["parsed"] = (
                                                parsed.__dict__
                                            )
                                        else:
                                            response_metadata["parsed"] = str(parsed)
                                    except Exception:
                                        response_metadata["parsed"] = str(parsed)

                                # Store the message and metadata
                                st.session_state.messages.append(
                                    {
                                        "role": "assistant",
                                        "content": response_text,
                                        "metadata": response_metadata,
                                    }
                                )

                                # Display the response
                                st.write(response_text)

                                # In dev mode, show tool calls and parsed data
                                if st.session_state.display_mode == "dev":
                                    for tc in tool_calls:
                                        # Ensure tc is a ToolExecutionSuggestion before formatting
                                        if isinstance(tc, ToolExecutionSuggestion):
                                            st.markdown(_format_tool_call(tc))

                                    if parsed is not None:
                                        st.write("**Parsed Output:**")
                                        try:
                                            parsed_json = json.dumps(
                                                response_metadata.get("parsed", {}),
                                                default=str,
                                            )
                                            st.json(parsed_json)
                                        except Exception:
                                            st.code(str(parsed))

                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": error_msg,
                                    "metadata": {},
                                }
                            )

        return _streamlit_app
