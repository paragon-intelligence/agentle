import json
from collections.abc import Callable
from typing import Any, Literal, cast, List, Tuple, Dict, Union
import uuid

from rsb.adapters.adapter import Adapter

from agentle.agents.agent import Agent
from agentle.agents.knowledge.static_knowledge import StaticKnowledge
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.models.messages.message import Message
from agentle.generations.models.messages.developer_message import DeveloperMessage

# Define a type for session-added knowledge items for clarity
SessionKnowledgeItem = Dict[
    str, Any
]  # Keys: "type", "name", "content", "data_bytes", "mime_type"


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
        token usage, static knowledge, and parsed outputs. Presentation mode provides a clean
        interface for demonstrating the agent's capabilities.

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
            from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

            # Create an agent
            agent = Agent(
                generation_provider=GoogleGenaiGenerationProvider(),
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
        app_description = self.description or (
            agent.description
            if agent.description and agent.description != "An AI agent"
            else None
        )

        def _format_tool_call_display(tool_call: ToolExecutionSuggestion) -> str:
            args_str = json.dumps(tool_call.args, indent=2, default=str)
            return f"**Tool Executed:** `{tool_call.tool_name}`\n**Arguments:**\n```json\n{args_str}\n```"

        def _streamlit_app() -> None:
            try:
                import streamlit as st
            except ImportError:
                print(
                    "CRITICAL ERROR: Streamlit is not installed or cannot be imported. "
                    + "Please install it with: pip install streamlit"
                )
                return

            st.set_page_config(
                page_title=app_title,
                page_icon="🤖",
                layout="wide",
                initial_sidebar_state="expanded",
            )

            # Initialize session state variables (plain assignment, types managed via casting or context)
            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "display_mode" not in st.session_state:
                st.session_state.display_mode = self.initial_mode
            if "token_usage" not in st.session_state:
                st.session_state.token_usage = []
            if (
                "uploaded_file_for_next_message" not in st.session_state
            ):  # Simplified for single file
                st.session_state.uploaded_file_for_next_message = (
                    None  # Stores a single file dict or None
                )
            if "session_added_knowledge" not in st.session_state:
                st.session_state.session_added_knowledge = []

            # --- Sidebar ---
            with st.sidebar:
                st.title("⚙️ Settings & Info")
                st.divider()

                st.selectbox(
                    "Display Mode",
                    ["presentation", "dev"],
                    key="display_mode",  # Direct binding
                    help="Switch between presentation view and developer view with more details.",
                )
                st.divider()

                st.header("Agent Details")
                st.write(f"**Name:** {agent.name}")
                if agent.description and agent.description != "An AI agent":
                    st.caption(f"{agent.description}")
                st.write(f"**Model:** `{agent.model or 'Not specified'}`")

                if agent.has_tools():
                    with st.expander("Available Tools", expanded=False):
                        tools_list: List[Any] = list(agent.tools)
                        if not tools_list:
                            st.caption("No tools configured.")
                        for _, tool_item in enumerate(tools_list):
                            tool_name = getattr(tool_item, "name", str(tool_item))
                            st.markdown(f"- `{tool_name}`")

                # Display Agent's Original Static Knowledge
                if agent.static_knowledge:
                    with st.expander("Initial Knowledge Base (Agent)", expanded=False):
                        knowledge_list_agent: List[Union[StaticKnowledge, str]] = list(
                            agent.static_knowledge
                        )
                        if not knowledge_list_agent:
                            st.caption("No initial knowledge items.")
                        for i, item in enumerate(knowledge_list_agent):
                            source_text, cache_text = "", "Cache: N/A"
                            if isinstance(item, StaticKnowledge):
                                source_text = (
                                    f"**Source {i + 1} (Static):** `{item.content}`"
                                )
                                cache_info = item.cache
                                if cache_info == "infinite":
                                    cache_text = "Cache: Infinite"
                                elif isinstance(cache_info, int) and cache_info > 0:
                                    cache_text = f"Cache: {cache_info}s"
                                elif cache_info == 0 or cache_info is None:
                                    cache_text = "Cache: Disabled/Default"
                                else:
                                    cache_text = f"Cache: {str(cache_info)}"
                            elif isinstance(item, str):
                                source_text = (
                                    f"**Source {i + 1} (Static Text):** `{item}`"
                                )
                            else:  # Should not happen with correct Agent typing
                                source_text = (
                                    f"**Source {i + 1} (Unknown):** `{str(item)}`"
                                )
                            st.markdown(source_text)
                            st.caption(cache_text)
                            if i < len(knowledge_list_agent) - 1:
                                st.markdown("---")

                # Add new knowledge (Session Only)
                with st.expander("➕ Add Knowledge (Session Only)", expanded=False):
                    new_knowledge_text = st.text_area(
                        "Enter URL or paste raw text:",
                        key="new_knowledge_text_url_input",
                        height=100,
                    )
                    new_knowledge_file = st.file_uploader(
                        "Upload knowledge file (.txt, .md)",  # Simplified types for now
                        type=["txt", "md"],
                        key="new_knowledge_file_input",
                    )
                    if st.button(
                        "Add to Session Knowledge",
                        key="add_session_knowledge_button_main",
                    ):
                        session_knowledge_list: List[SessionKnowledgeItem] = (
                            st.session_state.session_added_knowledge
                        )  # type: ignore
                        added_something = False
                        if new_knowledge_text:
                            session_knowledge_list.append(
                                {
                                    "type": "text_or_url",
                                    "name": "Text/URL Snippet",
                                    "content": new_knowledge_text,
                                    "data_bytes": None,
                                    "mime_type": "text/plain",
                                }
                            )
                            st.success(f"Added text/URL snippet to session knowledge.")
                            st.session_state.new_knowledge_text_url_input = (
                                ""  # Clear input
                            )
                            added_something = True
                        if new_knowledge_file is not None:
                            file_bytes = new_knowledge_file.getvalue()
                            try:
                                # For txt/md, decode to string
                                file_content_str = file_bytes.decode("utf-8")
                                session_knowledge_list.append(
                                    {
                                        "type": "file",
                                        "name": new_knowledge_file.name,
                                        "content": file_content_str,  # Store decoded content
                                        "data_bytes": file_bytes,  # Store raw bytes too if needed later
                                        "mime_type": new_knowledge_file.type
                                        or "text/plain",
                                    }
                                )
                                st.success(
                                    f"Added file '{new_knowledge_file.name}' to session knowledge."
                                )
                                added_something = True
                            except UnicodeDecodeError:
                                st.error(
                                    f"Could not decode file '{new_knowledge_file.name}' as UTF-8 text. Please upload plain text files (.txt, .md)."
                                )
                            # No easy way to clear file_uploader state reliably without complex keying/callbacks,
                            # user will have to manually clear/change it or it will re-submit if not careful.

                        if added_something:
                            st.session_state.session_added_knowledge = (
                                session_knowledge_list
                            )
                            st.rerun()
                        elif not new_knowledge_text and new_knowledge_file is None:
                            st.warning(
                                "Please provide text/URL or upload a file to add knowledge."
                            )

                # Display Session-Added Knowledge
                session_knowledge_to_display: List[SessionKnowledgeItem] = (
                    st.session_state.session_added_knowledge
                )  # type: ignore
                if session_knowledge_to_display:
                    with st.expander("Session-Added Knowledge", expanded=True):
                        for i, item_dict in enumerate(session_knowledge_to_display):
                            item_name = cast(str, item_dict.get("name", "Unknown Item"))
                            item_type = cast(str, item_dict.get("type", "unknown"))
                            item_content_preview = cast(
                                str, item_dict.get("content", "")
                            )
                            if len(item_content_preview) > 70:
                                item_content_preview = item_content_preview[:70] + "..."
                            st.markdown(f"**{i + 1}. {item_name}** ({item_type})")
                            st.caption(f"`{item_content_preview}`")
                            if i < len(session_knowledge_to_display) - 1:
                                st.markdown("---")
                st.divider()

                # Developer Zone
                current_display_mode_sidebar = cast(
                    Literal["dev", "presentation"], st.session_state.display_mode
                )
                if current_display_mode_sidebar == "dev":
                    st.header("Developer Zone")
                    with st.expander("📈 Usage Statistics", expanded=False):
                        current_token_usage_dev: List[Tuple[int, int]] = (
                            st.session_state.token_usage
                        )  # type: ignore
                        if current_token_usage_dev:
                            total_prompt = sum(p for p, _ in current_token_usage_dev)
                            total_completion = sum(
                                c for _, c in current_token_usage_dev
                            )
                            st.metric("Total Prompt Tokens", total_prompt)
                            st.metric("Total Completion Tokens", total_completion)
                            st.metric(
                                "Total Tokens Used", total_prompt + total_completion
                            )
                        else:
                            st.info("No token usage data recorded yet.")

                    if hasattr(agent, "config"):
                        with st.expander("🔧 Agent Configuration", expanded=False):
                            try:
                                config_obj = agent.config
                                if hasattr(config_obj, "model_dump") and callable(
                                    getattr(config_obj, "model_dump")
                                ):
                                    st.json(
                                        json.dumps(
                                            getattr(config_obj, "model_dump")(),
                                            indent=2,
                                            default=str,
                                        )
                                    )
                                elif hasattr(config_obj, "__dict__"):
                                    st.json(
                                        json.dumps(
                                            config_obj.__dict__, default=str, indent=2
                                        )
                                    )
                                else:
                                    st.text(str(config_obj))
                            except Exception as e_conf:
                                st.error(f"Could not display agent config: {e_conf}")
                    st.divider()

                if st.button(
                    "🗑️ Clear Conversation",
                    use_container_width=True,
                    key="clear_conversation_button_main",
                ):
                    st.session_state.messages = []
                    st.session_state.token_usage = []
                    st.session_state.uploaded_file_for_next_message = None
                    st.session_state.session_added_knowledge = []  # Clear session knowledge too
                    st.rerun()

            # --- Main Application Layout ---
            st.title(app_title)
            if app_description:
                st.caption(app_description)

            # Add custom CSS for chat layout
            st.markdown(
                """
            <style>
            .chat-container {
                display: flex;
                flex-direction: column;
                height: 70vh;
                margin-bottom: 120px;
            }
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 10px;
                margin-bottom: 10px;
            }
            .input-area {
                position: fixed;
                bottom: 0;
                left: 25%; /* Adjust based on sidebar width */
                right: 0;
                background-color: white;
                padding: 10px 30px 20px 10px;
                border-top: 1px solid #ddd;
                z-index: 1000;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            # Create container structure
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

            # 1. Chat Message Display Area
            chat_message_container = st.container()
            with chat_message_container:
                current_messages_main: List[Dict[str, Any]] = st.session_state.messages  # type: ignore
                if not current_messages_main:
                    st.info("Conversation will appear here. Send a message to start!")

                for msg_idx, message_data in enumerate(current_messages_main):
                    role = str(message_data.get("role", "unknown"))
                    content = str(message_data.get("content", ""))
                    metadata: Dict[str, Any] = cast(
                        Dict[str, Any], message_data.get("metadata", {})
                    )

                    if role == "user":
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(content)
                            files_metadata: Union[List[Dict[str, Any]], None] = (
                                metadata.get("files")
                            )  # type: ignore
                            if isinstance(files_metadata, list):
                                with st.container():
                                    for file_idx, file_info in enumerate(
                                        files_metadata
                                    ):
                                        file_name = str(file_info.get("name", "file"))
                                        file_data_bytes = cast(
                                            bytes, file_info.get("data", b"")
                                        )
                                        mime_type = str(
                                            file_info.get(
                                                "mime_type",
                                                "application/octet-stream",
                                            )
                                        )
                                        if file_data_bytes:
                                            if mime_type.startswith("image/"):
                                                st.image(
                                                    file_data_bytes,
                                                    caption=f"Sent: {file_name}",
                                                    width=100,
                                                )  # Smaller preview
                                            else:
                                                st.download_button(
                                                    f"📎 {file_name}",
                                                    file_data_bytes,
                                                    file_name=file_name,
                                                    mime=mime_type,
                                                    key=f"user_file_dl_{msg_idx}_{file_idx}_{file_name}",
                                                )
                    elif role == "assistant":
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(content)
                            # Dev mode details
                            current_display_mode_chat = cast(
                                Literal["dev", "presentation"],
                                st.session_state.display_mode,
                            )
                            if current_display_mode_chat == "dev":
                                tool_calls_md: Union[List[Dict[str, Any]], None] = (
                                    metadata.get("tool_calls")
                                )  # type: ignore
                                parsed_data_md: Any = metadata.get("parsed")

                                if isinstance(tool_calls_md, list) and tool_calls_md:
                                    with st.expander(
                                        "🛠️ Tool Calls Executed", expanded=False
                                    ):
                                        for tc_idx, tc_data in enumerate(tool_calls_md):
                                            if (
                                                "tool_name" in tc_data
                                                and "args" in tc_data
                                            ):
                                                tool_name_disp = str(
                                                    tc_data.get(
                                                        "tool_name", "Unknown Tool"
                                                    )
                                                )
                                                args_disp = cast(
                                                    Dict[str, Any],
                                                    tc_data.get("args", {}),
                                                )
                                                id_disp = str(
                                                    tc_data.get("id", uuid.uuid4())
                                                )
                                                tc_obj_disp = ToolExecutionSuggestion(
                                                    tool_name=tool_name_disp,
                                                    args=args_disp,
                                                    id=id_disp,
                                                )
                                                st.markdown(
                                                    _format_tool_call_display(
                                                        tc_obj_disp
                                                    )
                                                )
                                                if "result" in tc_data:
                                                    st.markdown("**Result:**")
                                                    st.code(
                                                        str(
                                                            tc_data.get("result", "")
                                                            or ""
                                                        ),
                                                        language="json",
                                                        line_numbers=False,
                                                    )  # type: ignore
                                                if tc_idx < len(tool_calls_md) - 1:
                                                    st.divider()

                                if parsed_data_md is not None:
                                    with st.expander(
                                        "🧩 Parsed Output", expanded=False
                                    ):
                                        try:
                                            st.json(
                                                json.dumps(
                                                    parsed_data_md,
                                                    default=str,
                                                    indent=2,
                                                )
                                            )
                                        except:
                                            st.text(str(parsed_data_md))
            st.markdown("</div>", unsafe_allow_html=True)  # Close chat-messages div

            # 2. Input Controls Area (fixed at bottom)
            st.markdown('<div class="input-area">', unsafe_allow_html=True)
            input_controls_container = st.container()
            with input_controls_container:
                # Using columns for file uploader and its preview
                uploader_col, preview_col, input_col = st.columns([0.3, 0.2, 0.5])

                with uploader_col:
                    # Ensure unique key for file_uploader if it needs to reset
                    messages_for_key_len = 0
                    if isinstance(st.session_state.messages, list):
                        messages_for_key_len = len(st.session_state.messages)
                    uploader_key = f"main_file_uploader_{messages_for_key_len}"

                    uploaded_file_obj = st.file_uploader(
                        "Attach file",
                        type=None,
                        key=uploader_key,
                        accept_multiple_files=False,
                        help="The file will be attached to your next message.",
                        label_visibility="collapsed",
                    )
                    if uploaded_file_obj is not None:
                        # Store this file in session state to be picked up by chat_input
                        st.session_state.uploaded_file_for_next_message = {
                            "name": uploaded_file_obj.name,
                            "data": uploaded_file_obj.getvalue(),
                            "mime_type": uploaded_file_obj.type
                            or "application/octet-stream",
                        }
                        # Rerun to show preview, chat_input will handle clearing it after send
                        st.rerun()

                with preview_col:
                    staged_file_info: Union[Dict[str, Any], None] = (
                        st.session_state.uploaded_file_for_next_message
                    )  # type: ignore
                    if staged_file_info:
                        fname = str(staged_file_info.get("name", "file"))
                        fmime = str(
                            staged_file_info.get(
                                "mime_type", "application/octet-stream"
                            )
                        )
                        fdata = cast(bytes, staged_file_info.get("data", b""))

                        if fdata:
                            if fmime.startswith("image/"):
                                st.image(fdata, caption=f"{fname}", width=70)
                            else:
                                st.info(
                                    f"📎 {fname[:10]}..."
                                    if len(fname) > 10
                                    else f"📎 {fname}"
                                )
                            if st.button(
                                f"❌",
                                key=f"remove_staged_file_btn",
                                help="Clear attached file",
                            ):
                                st.session_state.uploaded_file_for_next_message = None
                                st.rerun()

                # Chat input in the last column
                with input_col:
                    # Chat input bar
                    user_prompt = st.chat_input(
                        "Type your message here...", key="main_chat_input"
                    )

            st.markdown("</div>", unsafe_allow_html=True)  # Close input-area div
            st.markdown("</div>", unsafe_allow_html=True)  # Close chat-container div

            # --- Handle New User Input Processing ---
            if user_prompt:
                new_user_message_metadata: Dict[str, Any] = {}

                # Attach file from the staging area if it exists
                staged_file_to_send: Union[Dict[str, Any], None] = (
                    st.session_state.uploaded_file_for_next_message
                )  # type: ignore
                if staged_file_to_send:
                    new_user_message_metadata["files"] = [
                        staged_file_to_send
                    ]  # Agent expects a list of files
                    st.session_state.uploaded_file_for_next_message = (
                        None  # Clear after attaching
                    )

                # Add user message to chat history
                current_messages_processing: List[Dict[str, Any]] = (
                    st.session_state.messages
                )  # type: ignore
                current_messages_processing.append(
                    {
                        "role": "user",
                        "content": user_prompt,
                        "metadata": new_user_message_metadata,
                    }
                )
                st.session_state.messages = current_messages_processing

                # Prepare instructions including session-added knowledge
                original_instructions = ""
                if isinstance(agent.instructions, str):
                    original_instructions = agent.instructions
                elif callable(agent.instructions):
                    original_instructions = agent.instructions()
                elif isinstance(agent.instructions, list):
                    original_instructions = "\n".join(agent.instructions)

                session_knowledge_prompt_parts: List[str] = []
                session_knowledge_items_proc: List[SessionKnowledgeItem] = (
                    st.session_state.session_added_knowledge
                )  # type: ignore
                if session_knowledge_items_proc:
                    session_knowledge_prompt_parts.append(
                        "\n\n--- SESSION-ADDED KNOWLEDGE START ---"
                    )
                    for item in session_knowledge_items_proc:
                        item_name = cast(str, item.get("name", "Item"))
                        item_content = cast(str, item.get("content", ""))
                        session_knowledge_prompt_parts.append(
                            f"Knowledge Item: {item_name}\nContent:\n{item_content}"
                        )
                    session_knowledge_prompt_parts.append(
                        "--- SESSION-ADDED KNOWLEDGE END ---"
                    )

                final_instructions_for_run = original_instructions + "\n".join(
                    session_knowledge_prompt_parts
                )

                # Agent Processing
                with st.spinner("🤖 Agent is thinking..."):
                    agent_input_parts: List[Union[TextPart, FilePart]] = [
                        TextPart(text=user_prompt)
                    ]  # type: ignore

                    files_to_process_agent = new_user_message_metadata.get("files")
                    if isinstance(files_to_process_agent, list):
                        for file_info_agent in files_to_process_agent:
                            file_data_bytes_agent = cast(
                                bytes, file_info_agent.get("data", b"")
                            )
                            file_mime_type_agent = str(
                                file_info_agent.get("mime_type")
                                or "application/octet-stream"
                            )
                            if file_data_bytes_agent:
                                try:
                                    agent_input_parts.append(
                                        FilePart(
                                            data=file_data_bytes_agent,
                                            mime_type=file_mime_type_agent,
                                        )
                                    )
                                except ValueError as ve_filepart:
                                    st.warning(
                                        f"Skipping file for agent (invalid MIME: {file_mime_type_agent}). Error: {ve_filepart}"
                                    )

                    final_agent_input: Union[UserMessage, str]
                    if len(agent_input_parts) > 1 or any(
                        isinstance(p, FilePart) for p in agent_input_parts
                    ):  # type: ignore
                        final_agent_input = UserMessage(parts=agent_input_parts)  # type: ignore
                    else:
                        final_agent_input = user_prompt

                    try:
                        # Cloning the agent with modified instructions is the cleaner approach for this specific run.
                        temp_agent_for_run = agent.clone(
                            new_instructions=final_instructions_for_run
                        )

                        with temp_agent_for_run.with_mcp_servers():
                            result = temp_agent_for_run.run(final_agent_input)

                        generation = result.generation
                        response_text = generation.text or "..."

                        response_metadata_agent: Dict[str, Any] = {}
                        if hasattr(generation, "tool_calls") and generation.tool_calls:
                            tool_calls_list_agent_resp: List[
                                ToolExecutionSuggestion
                            ] = list(generation.tool_calls)
                            response_metadata_agent["tool_calls"] = [
                                {
                                    "tool_name": tc.tool_name,
                                    "args": tc.args,
                                    "id": tc.id,
                                    "result": getattr(tc, "_result", None),
                                }
                                for tc in tool_calls_list_agent_resp
                            ]

                        parsed_result_data_agent_resp: Any = result.parsed
                        if parsed_result_data_agent_resp is not None:
                            try:
                                if hasattr(
                                    parsed_result_data_agent_resp, "model_dump"
                                ) and callable(
                                    getattr(parsed_result_data_agent_resp, "model_dump")
                                ):
                                    response_metadata_agent["parsed"] = getattr(
                                        parsed_result_data_agent_resp, "model_dump"
                                    )()
                                elif hasattr(parsed_result_data_agent_resp, "__dict__"):
                                    response_metadata_agent["parsed"] = (
                                        parsed_result_data_agent_resp.__dict__
                                    )
                                else:
                                    response_metadata_agent["parsed"] = (
                                        parsed_result_data_agent_resp
                                    )
                            except Exception:
                                response_metadata_agent["parsed"] = str(
                                    parsed_result_data_agent_resp
                                )

                        # Append assistant's response to session_state.messages
                        assistant_response_messages: List[Dict[str, Any]] = (
                            st.session_state.messages
                        )  # type: ignore
                        assistant_response_messages.append(
                            {
                                "role": "assistant",
                                "content": response_text,
                                "metadata": response_metadata_agent,
                            }
                        )
                        st.session_state.messages = assistant_response_messages

                        # Update token usage
                        token_usage_update_list: List[Tuple[int, int]] = (
                            st.session_state.token_usage
                        )  # type: ignore
                        if hasattr(generation, "usage") and generation.usage:
                            token_usage_update_list.append(
                                (
                                    generation.usage.prompt_tokens,
                                    generation.usage.completion_tokens,
                                )
                            )
                            st.session_state.token_usage = token_usage_update_list

                    except Exception as e_agent_run_main:
                        error_msg = f"Agent error: {str(e_agent_run_main)}"
                        st.error(error_msg)
                        # Append error message to chat
                        error_handling_messages: List[Dict[str, Any]] = (
                            st.session_state.messages
                        )  # type: ignore
                        error_handling_messages.append(
                            {
                                "role": "assistant",
                                "content": f"⚠️ Error: {error_msg}",
                                "metadata": {"error": True},
                            }
                        )
                        st.session_state.messages = error_handling_messages
                st.rerun()

        return _streamlit_app
