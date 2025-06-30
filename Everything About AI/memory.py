# Import necessary libraries
from langchain.memory import ConversationBufferMemory

# Class for memory to store old chats
class FixedMemory(ConversationBufferMemory):
    def save_context(self, inputs, outputs):
        # Only pass 'answer' to memory
        filtered_outputs = {"answer": outputs["answer"]}
        super().save_context(inputs, filtered_outputs)

# FUnction to get the conversation memory
def get_conversation_memory():
    return FixedMemory(
        memory_key="chat_history",
        return_messages=True
    )