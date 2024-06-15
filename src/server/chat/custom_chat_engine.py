from llama_index.core.chat_engine.types import BaseChatEngine
import asyncio
from threading import Thread
from typing import Any, List, Optional, Tuple
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
    ToolOutput,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)
from src.components.translation.base import BaseTranslation
from src.settings.settings import Settings as RepoSettings


DEFAULT_CONTEXT_TEMPLATE = (
    "Context information is below."
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)
import time


class CustomChatEngineWithTranslation(BaseChatEngine):
    """Context Chat Engine.

    Uses a retriever to retrieve a context, set the context in the system prompt,
    and then translation from Vietnamese to English if necessary,
    after that uses an LLM to generate a response for a fluid chat experience.
    Finally, translation back to original language (vietnameses)
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        translation: BaseTranslation = None,
        repo_settings: RepoSettings = None,
        tokenizer_src=None,
        tokenizer_tag=None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        context_template: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._translation = translation
        self._repo_settings = repo_settings
        self._tokenizer_src = tokenizer_src
        self._tokenizer_tag = tokenizer_tag
        self._memory = memory
        self._prefix_messages = prefix_messages
        self._node_postprocessors = node_postprocessors or []
        self._context_template = context_template or DEFAULT_CONTEXT_TEMPLATE
        self.callback_manager = callback_manager or CallbackManager([])
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = self.callback_manager

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        service_context: Optional[ServiceContext] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        context_template: Optional[str] = None,
        llm: Optional[LLM] = None,
        translation: BaseTranslation = None,
        repo_settings: RepoSettings = None,
        tokenizer_src=None,
        tokenizer_tag=None,
        **kwargs: Any,
    ) -> "CustomChatEngine":
        """Initialize a ContextChatEngine from default parameters."""
        llm = llm or llm_from_settings_or_context(Settings, service_context)
        chat_history = chat_history or []
        memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=chat_history, token_limit=llm.metadata.context_window - 256
        )
        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [
                ChatMessage(content=system_prompt, role=llm.metadata.system_role)
            ]

        prefix_messages = prefix_messages or []
        node_postprocessors = node_postprocessors or []

        return cls(
            retriever,
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            node_postprocessors=node_postprocessors,
            callback_manager=callback_manager_from_settings_or_context(
                Settings, service_context
            ),
            context_template=context_template,
            translation=translation,
            repo_settings=repo_settings,
            tokenizer_src=tokenizer_src,
            tokenizer_tag=tokenizer_tag,
        )

    def _translate(self, input_message: str, context_str: str, forward: bool):
        translated_input_message, translated_context_str = self._translation.translation_pipeline(
            input_message=input_message,
            context_str=context_str,
            forward=forward,
            tokenizer=self._tokenizer_src
        )

        if translated_input_message is not None:
            return translated_input_message, translated_context_str
        print('translated_context_str', translated_context_str)
        return translated_context_str

    def _generate_context(self, message: str) -> Tuple[str, List[NodeWithScore]]:
        """Generate context information from a message."""
        nodes = self._retriever.retrieve(message)
        print("len_node", len(nodes))
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )

        context_str = "\n\n".join(
            [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes]
        )

        print('context_str', context_str)
        translated_input_message, translated_context_str = self._translate(
            message, context_str, forward=True
        )

        print("translated_context_str:", translated_context_str)
        # print('translated_input_message:', translated_input_message)

        return (
            translated_input_message,
            self._context_template.format(context_str=translated_context_str),
            nodes,
        )

    async def _agenerate_context(self, message: str) -> Tuple[str, List[NodeWithScore]]:
        """Generate context information from a message."""
        nodes = await self._retriever.aretrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )

        print(
            "\n\n".join(
                [n.node.get_content(metadata_mode=MetadataMode.LLM) for n in nodes]
            )
        )
        context_str = "\n\n".join(
            [
                self._translate(
                    n.node.get_content(metadata_mode=MetadataMode.LLM).strip(),
                    forward=True,
                )
                for n in nodes
            ]
        )

        return self._context_template.format(context_str=context_str), nodes

    def _get_prefix_messages_with_context(self, context_str: str) -> List[ChatMessage]:
        """Get the prefix messages with context."""
        # ensure we grab the user-configured system prompt
        system_prompt = ""
        prefix_messages = self._prefix_messages
        if (
            len(self._prefix_messages) != 0
            and self._prefix_messages[0].role == MessageRole.SYSTEM
        ):
            system_prompt = str(self._prefix_messages[0].content)
            prefix_messages = self._prefix_messages[1:]

        context_str_w_sys_prompt = system_prompt.strip() + "\n" + context_str
        return [
            ChatMessage(
                content=context_str_w_sys_prompt, role=self._llm.metadata.system_role
            ),
            *prefix_messages,
        ]

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        import time

        st_time = time.time()
        translated_input_message, context_str_template, nodes = self._generate_context(
            message
        )
        print("time translate", time.time() - st_time)

        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=translated_input_message, role="user"))

        prefix_messages = self._get_prefix_messages_with_context(context_str_template)
        prefix_messages_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in prefix_messages])
            )
        )
        all_messages = prefix_messages + self._memory.get(
            initial_token_count=prefix_messages_token_count
        )

        # print("ALLLLLLL:   ", all_messages)
        st_time = time.time()
        chat_response = self._llm.chat(all_messages)
        print("time_llm", time.time() - st_time)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        print("respone", (chat_response.message.content))
        return AgentChatResponse(
            response=self._translate(
                input_message=None,
                context_str=chat_response.message.content,
                forward=False,
            ),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        translated_input_message, context_str_template, nodes = self._generate_context(
            message
        )

        self._memory.put(ChatMessage(content=translated_input_message, role="user"))

        prefix_messages = self._get_prefix_messages_with_context(context_str_template)
        initial_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in prefix_messages])
            )
        )
        all_messages = prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = StreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(all_messages),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )
        thread = Thread(
            target=chat_response.write_response_to_history, args=(self._memory,)
        )
        thread.start()

        return chat_response

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(
            ChatMessage(
                content=self._translate(message=message, forward=True)[0],
                role="user",
            )
        )

        context_str_template, nodes = await self._agenerate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)
        initial_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in prefix_messages])
            )
        )
        all_messages = prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = await self._llm.achat(all_messages)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        return AgentChatResponse(
            response=str(chat_response.message.content),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(
            ChatMessage(
                content=self._translate(messages=message, forward=True)[0],
                role="user",
            )
        )

        context_str_template, nodes = await self._agenerate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)
        initial_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in prefix_messages])
            )
        )
        all_messages = prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = StreamingAgentChatResponse(
            achat_stream=await self._llm.astream_chat(all_messages),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )
        thread = Thread(
            target=lambda x: asyncio.run(chat_response.awrite_response_to_history(x)),
            args=(self._memory,),
        )
        thread.start()

        return chat_response

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        return self._memory.get_all()
