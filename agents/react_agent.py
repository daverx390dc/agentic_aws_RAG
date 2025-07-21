"""
ReAct Agent using LangChain's ReAct agent type and tool-calling.
"""
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.llms.bedrock import BedrockLLM
from config.settings import settings

class ReActAgent:
    """Agent that uses ReAct and tool-calling for advanced reasoning and action."""
    def __init__(self):
        self.embeddings = BedrockEmbeddings()
        self.vectorstore = OpenSearchVectorSearch(
            opensearch_url=settings.opensearch_url,
            index_name=settings.opensearch_index,
            embedding_function=self.embeddings,
            http_auth=(settings.opensearch_user, settings.opensearch_password),
            use_ssl=settings.opensearch_use_ssl,
            verify_certs=settings.opensearch_verify_certs
        )
        self.llm = BedrockLLM()
        self.tools = [self.search_tool, self.summarize_tool]
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.REACT_DESCRIPTION,
            verbose=True,
        )

    @tool
    def search_tool(self, query: str) -> str:
        """Search the vectorstore for relevant documents."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant documents found."
        return "\n---\n".join([doc.page_content[:300] for doc in docs])

    @tool
    def summarize_tool(self, text: str) -> str:
        """Summarize the given text using the LLM."""
        prompt = f"Summarize the following text in 3 sentences:\n\n{text}"
        return self.llm(prompt)

    def run(self, query: str) -> str:
        """Run the ReAct agent on a user query."""
        return self.agent.run(query) 