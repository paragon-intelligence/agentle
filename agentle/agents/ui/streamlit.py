import json
from collections.abc import Callable
from typing import Any, Literal, cast, List, Tuple, Dict, Union

from rsb.adapters.adapter import Adapter

from agentle.agents.agent import Agent
from agentle.agents.knowledge.static_knowledge import StaticKnowledge
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.user_message import UserMessage

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
            ```
        """
        agent = _f
        app_title = self.title or f"{agent.name} Agent"
        app_description = self.description or (
            agent.description
            if agent.description and agent.description != "An AI agent"
            else None
        )

        def _format_tool_call_display(tool_call: ToolExecutionSuggestion) -> str:  # type: ignore
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

            # Initialize session state variables
            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "display_mode" not in st.session_state:
                st.session_state.display_mode = self.initial_mode
            if "token_usage" not in st.session_state:
                st.session_state.token_usage = []
            if "uploaded_file_for_next_message" not in st.session_state:
                st.session_state.uploaded_file_for_next_message = None
            if "session_added_knowledge" not in st.session_state:
                st.session_state.session_added_knowledge = []

            # Add custom CSS for entire app - IMPROVED STYLING
            st.markdown(
                """
            <style>
            /* Global styling */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            /* Base styling fixes */
            .main {
                background-color: #ffffff;
                padding: 0 !important;
            }
            
            /* Fix container spacing */
            .main > div {
                padding-top: 0 !important;
            }
            
            .stApp > header {
                display: none;
            }
            
            .block-container {
                padding-top: 0.5rem !important;
                padding-bottom: 5rem !important;
                max-width: 1000px;
            }
            
            /* Sidebar styling */
            .css-1544g2n.e1fqkh3o4 {  /* Target sidebar */
                padding-top: 0.5rem;
            }
            
            /* Chat header styling */
            .chat-header {
                background-color: white;
                border-radius: 8px;
                padding: 0.75rem;
                margin-bottom: 0.5rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                border: 1px solid #e6e6e6;
            }
            
            .chat-header h2 {
                margin: 0;
                color: #333;
                font-weight: 600;
                font-size: 1.25rem;
            }
            
            .chat-header p {
                margin: 0.25rem 0 0 0;
                color: #666;
                font-size: 0.9rem;
            }
            
            
            
            .chat-messages {
                flex: 1;
                max-height: 65vh;
                overflow-y: auto;
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e6e6e6;
                padding: 0.75rem;
                margin-bottom: 0.5rem;
                display: flex;
                flex-direction: column;
            }
            
            /* Message styling */
            .message {
                display: flex;
                margin-bottom: 0.5rem;
                padding: 0.5rem;
                border-radius: 8px;
                max-width: 85%;
            }
            
            .user-message {
                background-color: #f0f2f5;
                align-self: flex-end;
                margin-left: auto;
            }
            
            .assistant-message {
                background-color: #e3f2fd;
                align-self: flex-start;
            }
            
            .message-content {
                padding: 0.25rem;
            }
            
            /* Welcome styling - MORE COMPACT */
            .welcome-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 200px;
                padding: 1rem;
                text-align: center;
                color: #666;
            }
            
            .welcome-icon {
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
                color: #4285f4;
            }
            
            .welcome-title {
                font-size: 1.25rem;
                font-weight: 600;
                margin-bottom: 0.25rem;
                color: #333;
            }
            
            .welcome-message {
                font-size: 1rem;
                max-width: 600px;
                margin-bottom: 0.5rem;
            }
            
            /* Input area styling - POSITIONED CLOSER TO MESSAGES */
            .input-area {
                position: relative;
                background-color: white;
                padding: 0.75rem;
                border-top: 1px solid #e6e6e6;
                z-index: 1000;
                display: flex;
                justify-content: center;
                margin-top: 1rem;
            }
            
            /* Input container layout */
            .input-container {
                display: flex;
                width: 100%;
                max-width: 1000px;
                align-items: center;
                gap: 0.5rem;
                position: relative;
            }
            
            /* File upload styling - FIXED POSITION */
            .file-upload-container {
                position: absolute;
                right: -40px;
                top: 0;
                bottom: 0;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .file-upload-button {
                display: flex;
                align-items: center;
                justify-content: center;
                background-color: #f0f2f5;
                color: #555;
                border: 1px solid #ddd;
                border-radius: 50%;
                width: 36px;
                height: 36px;
                font-size: 1.1rem;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            
            .file-upload-button:hover {
                background-color: #e4e6e9;
                border-color: #ccc;
            }
            
            /* File preview styling */
            .file-preview {
                display: flex;
                align-items: center;
                background-color: #f0f2f5;
                padding: 0.5rem;
                border-radius: 4px;
                margin-bottom: 0.5rem;
                border: 1px solid #ddd;
                max-width: 100%;
            }
            
            /* Message attachment styling */
            .message-attachment {
                margin-top: 0.5rem;
                border-radius: 4px;
                background-color: #f0f2f5;
                padding: 0.5rem;
                display: inline-block;
                border: 1px solid #ddd;
            }
            
            .message-attachment-file {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            /* Remove file button styling */
            button.remove-file {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 0.25rem 0.5rem;
                cursor: pointer;
                font-size: 0.8rem;
                transition: all 0.2s ease;
            }
            
            button.remove-file:hover {
                background-color: #d32f2f;
            }
            
            /* Ensure proper text contrast */
            p, h1, h2, h3, h4, h5, h6, span, div {
                color: #333 !important;
            }
            
            .stMarkdown a, code {
                color: #0066cc !important;
            }
            
            /* Fix sidebar expanders */
            .sidebar .streamlit-expanderHeader {
                color: #333 !important;
                background-color: #f0f2f5;
                border-radius: 4px;
                padding: 0.5rem;
            }
            
            /* Fix chat input styling */
            .stChatInput {
                padding-right: 0 !important;
                flex: 1;
            }
            
            .stChatInput > div {
                width: 100%;
            }
            
            /* Fix stChatMessage */
            .stChatMessage {
                background-color: transparent !important;
                padding: 0 !important;
                margin-bottom: 0 !important;
                border-radius: 0 !important;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            # --- Sidebar ---
            with st.sidebar:
                st.title("⚙️ Settings & Info")
                st.divider()

                st.selectbox(
                    "Display Mode",
                    ["presentation", "dev"],
                    key="display_mode",
                    help="Switch between presentation view and developer view with more details.",
                )
                st.divider()

                st.header("Agent Details")
                st.write(f"**Name:** {agent.name}")
                if agent.description and agent.description != "An AI agent":
                    st.caption(f"{agent.description}")
                st.write(f"**Model:** `{agent.model or 'Not specified'}`")

                if agent.has_tools():
                    with st.expander("🛠️ Available Tools", expanded=False):
                        tools_list: List[Any] = list(agent.tools)
                        if not tools_list:
                            st.caption("No tools configured.")
                        else:
                            for _, tool_item in enumerate(tools_list):
                                tool_name = getattr(tool_item, "name", str(tool_item))
                                st.code(tool_name, language=None)

                # Display Agent's Original Static Knowledge
                if agent.static_knowledge:
                    with st.expander("📚 Knowledge Base", expanded=False):
                        knowledge_list_agent: List[Union[StaticKnowledge, str]] = list(
                            agent.static_knowledge
                        )
                        if not knowledge_list_agent:
                            st.caption("No initial knowledge items.")
                        for i, item in enumerate(knowledge_list_agent):
                            source_text, cache_text = "", "Cache: N/A"
                            if isinstance(item, StaticKnowledge):
                                source_text = item.content
                                cache_info = item.cache
                                if cache_info == "infinite":
                                    cache_text = "Cache: Infinite"
                                elif isinstance(cache_info, int) and cache_info > 0:
                                    cache_text = f"Cache: {cache_info}s"
                                elif cache_info == 0 or cache_info is None:
                                    cache_text = "Cache: Disabled/Default"
                                else:
                                    cache_text = f"Cache: {str(cache_info)}"
                            elif isinstance(item, str):  # type: ignore
                                source_text = item

                            st.markdown(f"**Source {i + 1}**")
                            st.text(
                                source_text[:100]
                                + ("..." if len(source_text) > 100 else "")
                            )
                            st.caption(cache_text)
                            if i < len(knowledge_list_agent) - 1:
                                st.divider()

                # Add new knowledge (Session Only)
                with st.expander("➕ Add Knowledge", expanded=False):
                    new_knowledge_text = st.text_area(
                        "Enter URL or paste raw text:",
                        key="new_knowledge_text_url_input",
                        height=100,
                    )
                    new_knowledge_file = st.file_uploader(
                        "Upload knowledge file (.txt, .md)",
                        type=["txt", "md"],
                        key="new_knowledge_file_input",
                    )
                    if st.button(
                        "Add to Knowledge Base",
                        key="add_session_knowledge_button_main",
                    ):
                        session_knowledge_list: List[SessionKnowledgeItem] = (
                            st.session_state.session_added_knowledge
                        )
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
                            st.session_state.new_knowledge_text_url_input = ""
                            added_something = True
                        if new_knowledge_file is not None:
                            file_bytes = new_knowledge_file.getvalue()
                            try:
                                file_content_str = file_bytes.decode("utf-8")
                                session_knowledge_list.append(
                                    {
                                        "type": "file",
                                        "name": new_knowledge_file.name,
                                        "content": file_content_str,
                                        "data_bytes": file_bytes,
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
                )
                if session_knowledge_to_display:
                    with st.expander("📝 Session Knowledge", expanded=True):
                        for i, item_dict in enumerate(session_knowledge_to_display):
                            item_name = cast(str, item_dict.get("name", "Unknown Item"))
                            item_type = cast(str, item_dict.get("type", "unknown"))
                            item_content_preview = cast(
                                str, item_dict.get("content", "")
                            )
                            if len(item_content_preview) > 70:
                                item_content_preview = item_content_preview[:70] + "..."

                            st.markdown(f"**{item_name}** ({item_type})")
                            st.text(item_content_preview)
                            if i < len(session_knowledge_to_display) - 1:
                                st.divider()

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
                        )
                        if current_token_usage_dev:
                            total_prompt = sum(p for p, _ in current_token_usage_dev)
                            total_completion = sum(
                                c for _, c in current_token_usage_dev
                            )

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Prompt Tokens", total_prompt)
                            with col2:
                                st.metric("Completion Tokens", total_completion)
                            with col3:
                                st.metric(
                                    "Total Tokens", total_prompt + total_completion
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

                # Clear conversation button
                if st.button(
                    "🗑️ Clear Conversation",
                    use_container_width=True,
                    key="clear_conversation_button_main",
                ):
                    st.session_state.messages = []
                    st.session_state.token_usage = []
                    st.session_state.uploaded_file_for_next_message = None
                    st.session_state.session_added_knowledge = []
                    st.rerun()

            # --- Main Application Layout ---
            # App header
            st.markdown(
                f"""
                <div class="chat-header">
                    <h2>{app_title}</h2>
                    {f"<p>{app_description}</p>" if app_description else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Create container structure - IMPROVED STRUCTURE
            message_container = st.container()

            # Message display area
            with message_container:
                # Add a container for messages that doesn't use chat-messages div
                message_area = st.empty()

                # Render messages using HTML instead of st.chat_message
                current_messages_main: List[Dict[str, Any]] = st.session_state.messages

                if not current_messages_main:
                    # Welcome message
                    message_area.markdown(
                        f"""
                        <div class="welcome-container">
                            <div class="welcome-icon">🤖</div>
                            <div class="welcome-title">Welcome to {agent.name}</div>
                            <div class="welcome-message">
                                {agent.description if agent.description and agent.description != "An AI agent" else "I'm here to assist you. What can I help you with today?"}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    # Render all messages as HTML
                    messages_html = ""

                    for _, message_data in enumerate(current_messages_main):
                        role = str(message_data.get("role", "unknown"))
                        content = str(message_data.get("content", ""))
                        metadata: Dict[str, Any] = cast(
                            Dict[str, Any], message_data.get("metadata", {})
                        )

                        message_class = (
                            "user-message" if role == "user" else "assistant-message"
                        )

                        # Start message div
                        messages_html += f'<div class="message {message_class}">'
                        messages_html += '<div class="message-content">'

                        # Message content
                        messages_html += f"{content}"

                        # Handle file attachments for user messages
                        if role == "user":
                            files_metadata: Union[List[Dict[str, Any]], None] = (
                                metadata.get("files")
                            )
                            if isinstance(files_metadata, list) and files_metadata:
                                for file_info in files_metadata:
                                    file_name = str(file_info.get("name", "file"))
                                    file_icon = "📄"  # Default icon
                                    messages_html += f"""
                                    <div class="message-attachment">
                                        <div class="message-attachment-file">
                                            <span>{file_icon}</span>
                                            <span><b>{file_name}</b></span>
                                        </div>
                                    </div>
                                    """

                        # Tool calls for assistant messages in dev mode
                        if (
                            role == "assistant"
                            and st.session_state.display_mode == "dev"
                        ):
                            tool_calls_md: Union[List[Dict[str, Any]], None] = (
                                metadata.get("tool_calls")
                            )
                            if isinstance(tool_calls_md, list) and tool_calls_md:
                                messages_html += (
                                    '<div style="margin-top:0.5rem;font-size:0.85rem;">'
                                )
                                messages_html += "<details>"
                                messages_html += "<summary>🛠️ Tool Calls</summary>"

                                for tc_data in tool_calls_md:
                                    if "tool_name" in tc_data and "args" in tc_data:
                                        tool_name = str(
                                            tc_data.get("tool_name", "Unknown Tool")
                                        )
                                        args_json = json.dumps(
                                            tc_data.get("args", {}),
                                            indent=2,
                                            default=str,
                                        )

                                        messages_html += f'<div style="margin-top:0.5rem;"><b>Tool:</b> {tool_name}</div>'
                                        messages_html += f'<pre style="background:#f5f5f5;padding:0.5rem;border-radius:4px;font-size:0.8rem;overflow:auto;">{args_json}</pre>'

                                        if "result" in tc_data:
                                            result_str = str(tc_data.get("result", ""))
                                            messages_html += "<div><b>Result:</b></div>"
                                            messages_html += f'<pre style="background:#f5f5f5;padding:0.5rem;border-radius:4px;font-size:0.8rem;overflow:auto;">{result_str}</pre>'

                                messages_html += "</details>"
                                messages_html += "</div>"

                        # Close message div
                        messages_html += "</div></div>"

                    # Render all messages at once
                    message_area.markdown(messages_html, unsafe_allow_html=True)

            # Input area with improved file upload
            st.markdown('<div class="input-area">', unsafe_allow_html=True)
            st.markdown('<div class="input-container">', unsafe_allow_html=True)

            # File preview if a file is selected
            staged_file_info = st.session_state.uploaded_file_for_next_message

            if staged_file_info:
                fname = str(staged_file_info.get("name", "file"))
                file_size = len(staged_file_info.get("data", b"")) // 1024
                file_size_display = (
                    f"{file_size} KB"
                    if file_size < 1024
                    else f"{file_size // 1024:.1f} MB"
                )

                col1, col2 = st.columns([0.7, 0.3])
                with col1:
                    st.markdown(
                        f"""
                        <div class="file-preview">
                            <span style="margin-right:8px;">📎</span>
                            <span><b>{fname}</b> ({file_size_display})</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button("Remove file", key="remove_file_button"):
                        st.session_state.uploaded_file_for_next_message = None
                        st.rerun()

            # Create a proper layout for input and upload
            input_col, _ = st.columns([0.9, 0.1])

            # Text input
            with input_col:
                user_prompt = st.chat_input(
                    "Type your message here...", key="main_chat_input"
                )

            # Completely redesigned file upload approach
            # Instead of using a column, place the file upload outside the main input flow

            # Create a hidden uploader with a unique key
            uploader_key = f"file_uploader_{len(st.session_state.messages) if isinstance(st.session_state.messages, list) else 0}"  # type: ignore

            # Create the actual uploader with minimal visibility
            uploaded_file = st.file_uploader(
                "Upload file", key=uploader_key, label_visibility="collapsed"
            )

            if uploaded_file is not None:
                # Store file data properly
                st.session_state.uploaded_file_for_next_message = {
                    "name": uploaded_file.name,
                    "data": uploaded_file.getvalue(),
                    "mime_type": uploaded_file.type or "application/octet-stream",
                }
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)  # Close input-container
            st.markdown("</div>", unsafe_allow_html=True)  # Close input-area

            # --- Handle New User Input Processing ---
            if user_prompt:
                new_user_message_metadata: Dict[str, Any] = {}

                # Attach file if it exists
                staged_file_to_send: Union[Dict[str, Any], None] = (
                    st.session_state.uploaded_file_for_next_message
                )
                if staged_file_to_send:
                    new_user_message_metadata["files"] = [staged_file_to_send]
                    st.session_state.uploaded_file_for_next_message = None

                # Add user message to chat history
                current_messages_processing: List[Dict[str, Any]] = (
                    st.session_state.messages
                )
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
                )
                if session_knowledge_items_proc:
                    session_knowledge_prompt_parts.append(
                        "\n\n--- SESSION-ADDED KNOWLEDGE START ---"
                    )
                    for item_dict in session_knowledge_items_proc:
                        item_name = str(item_dict.get("name", "Item"))  # type: ignore
                        item_content = str(item_dict.get("content", ""))  # type: ignore
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
                    ]

                    files_to_process_agent = new_user_message_metadata.get("files")
                    if isinstance(files_to_process_agent, list):
                        for file_info_agent in files_to_process_agent:
                            file_data_bytes_agent = cast(
                                bytes, file_info_agent.get("data", b"")
                            )
                            file_mime_type_agent = str(
                                file_info_agent.get("mime_type")  # type: ignore
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
                    ):
                        final_agent_input = UserMessage(parts=agent_input_parts)
                    else:
                        final_agent_input = user_prompt

                    try:
                        # Clone agent with modified instructions
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

                        # Append assistant's response
                        assistant_response_messages: List[Dict[str, Any]] = (
                            st.session_state.messages
                        )
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
                        )
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
                        )
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
