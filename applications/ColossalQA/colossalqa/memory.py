"""
Implement a memory class for storing conversation history
Support long term and short term memory
"""

from typing import Any, Dict, List

from colossalqa.chain.memory.summary import ConversationSummaryMemory
from colossalqa.chain.retrieval_qa.load_chain import load_qa_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage
from langchain.schema.retriever import BaseRetriever
from pydantic import Field


class ConversationBufferWithSummary(ConversationSummaryMemory):
    """Memory class for storing information about entities."""

    # Define dictionary to store information about entities.
    # Store the most recent conversation history
    buffered_history: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    # Temp buffer for summarized history
    summarized_history_temp: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    human_prefix: str = "Human"
    ai_prefix: str = "Assistant"
    buffer: str = ""  # Formatted conversation in string format
    existing_summary: str = ""  # Summarization of stale conversation in string format
    # Define key to pass information about entities into prompt.
    memory_key: str = "chat_history"
    input_key: str = "question"
    retriever: BaseRetriever = None
    max_tokens: int = 2000
    chain: BaseCombineDocumentsChain = None
    input_chain_type_kwargs: List = {}

    @property
    def buffer(self) -> Any:
        """String buffer of memory."""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    @property
    def buffer_as_str(self) -> str:
        """Exposes the buffer as a string in case return_messages is True."""
        self.buffer = self.format_dialogue()
        return self.buffer

    @property
    def buffer_as_messages(self) -> List[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        return self.buffered_history.messages

    def clear(self):
        """Clear all the memory"""
        self.buffered_history.clear()
        self.summarized_history_temp.clear()

    def initiate_document_retrieval_chain(
        self, llm: Any, prompt_template: Any, retriever: Any, chain_type_kwargs: Dict[str, Any] = {}
    ) -> None:
        """
        Since we need to calculate the length of the prompt, we need to initiate a retrieval chain
        to calculate the length of the prompt.
        Args:
            llm: the language model for the retrieval chain (we won't actually return the output)
            prompt_template: the prompt template for constructing the retrieval chain
            retriever: the retriever for the retrieval chain
            chain_type_kwargs: the kwargs for the retrieval chain
        """
        self.retriever = retriever
        input_chain_type_kwargs = {k: v for k, v in chain_type_kwargs.items() if k not in [self.memory_key]}
        self.input_chain_type_kwargs = input_chain_type_kwargs
        self.chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template, **self.input_chain_type_kwargs)

    @property
    def memory_variables(self) -> List[str]:
        """Define the variables we are providing to the prompt."""
        return [self.memory_key]

    def format_dialogue(self, lang: str = "en") -> str:
        """Format memory into two parts--- summarization of historical conversation and most recent conversation"""
        if len(self.summarized_history_temp.messages) != 0:
            for i in range(int(len(self.summarized_history_temp.messages) / 2)):
                self.existing_summary = (
                    self.predict_new_summary(
                        self.summarized_history_temp.messages[i * 2 : i * 2 + 2], self.existing_summary, stop=["\n\n"]
                    )
                    .strip()
                    .split("\n")[0]
                    .strip()
                )
            for i in range(int(len(self.summarized_history_temp.messages) / 2)):
                self.summarized_history_temp.messages.pop(0)
                self.summarized_history_temp.messages.pop(0)
        conversation_buffer = []
        for t