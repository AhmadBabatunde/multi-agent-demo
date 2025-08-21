import os
import uuid
from datetime import datetime
import shutil
from typing import TypedDict, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, Send
from langchain_core.messages import BaseMessage
# --- FastAPI & Pydantic Imports ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
import requests

# --- Langmem and Langgraph Memory Imports ---
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_store

from dotenv import load_dotenv
load_dotenv()

# This should be configured if you are using the genai client directly, but langchain handles it.
# client = genai.Client() 

app = FastAPI(
    title="Multi-Agent RAG System API",
    description="An API for interacting with a multi-agent system built with LangGraph.",
    version="1.0.0"
)

# Configure CORS to allow the frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a temporary directory for file uploads
TEMP_UPLOAD_DIR = "./temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# --- Langmem Persistent Memory Setup ---
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small"
    }
)

namespace = ("agent_memories", "{user_id}", "{thread_id}")

memory_tools = [
    create_manage_memory_tool(namespace),
    create_search_memory_tool(namespace)
]

checkpointer = InMemorySaver()

# --- MODIFIED AND FIXED PROMPT FUNCTION ---
def create_eager_prompt_function(base_prompt_text: str, store_instance):
    """Creates a function that injects memories into an agent's prompt."""
    def eager_prompt_with_base(state: MessagesState):
        # The store is now accessed directly from the parent function's scope.
        # We no longer need get_store().

        # Search for memories using the last message as a query.
        items = store_instance.search(namespace, query=state["messages"][-1].content)
        memories = "\n\n".join(str(item) for item in items)

        # Create the prompt messages.
        base_prompt_msg = SystemMessage(content=base_prompt_text)

        if not memories:
            return [base_prompt_msg] + state["messages"]

        memory_msg = SystemMessage(content=f"## Relevant Memories:\n\n{memories}")
        return [memory_msg, base_prompt_msg] + state["messages"]

    return eager_prompt_with_base

# --- LLM and Embeddings Initialization ---
openai_api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
llm = ChatOpenAI(model="gpt-4", temperature=0.0, api_key=openai_api_key)

# --- RAG Agent Setup ---
def load_file(file_path: str):
    """Load and extract text from a file."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path)
    else:
        loader = UnstructuredFileLoader(file_path)
    return loader.load()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Initialize vector store - MODIFIED: Now we'll create it once and reuse it
import chromadb

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from fastapi import Depends

# Connect to the running server
#client = chromadb.HttpClient(host="localhost", port=7000)
client = chromadb.CloudClient(
  api_key='ck-AWp8K3jRGnUnmviTfLBoU8UVufrxYyn6PVKPdT4cJxZe',
  tenant='2b67c962-4b87-4a3c-a60c-77e11df14e50',
  database='multi-agent-demo'
)
vector_store = Chroma(
    collection_name="multiAgent_collection",
    embedding_function=embeddings,
    client=client
)

# MODIFIED: Remove the initial document loading to allow for multiple uploads
# Instead, we'll check if the vector store is empty and handle it in the upload endpoint
print(f"Initial number of vectors in store: {vector_store._collection.count()}")

@tool
def retrieve_docs(question: str) -> str:
    """Retrieve relevant documents as plain text for the agent to read."""
    docs = vector_store.similarity_search(question, k=3)
    if not docs:
        return "No relevant documents found."
    
    output = "\n\n".join([f"Doc {i+1}: {doc.page_content[:500]}" for i, doc in enumerate(docs)])
    print(f"🔎 Retrieved {len(docs)} docs for: {question}")
    return output

# --- MODIFIED: Agent Setups with Memory ---
rag_prompt = """
You are a Retrieval-Augmented Generation (RAG) agent.
When a user asks a question, ALWAYS:
1. Call the `retrieve_docs` tool with the full question.
2. Read the retrieved docs.
3. Summarize or answer using only the retrieved content.

Do not say you don't know if docs are returned. If docs are empty, then respond with "No relevant documents found."
"""

rag_agent = create_react_agent(
    model=llm,
    tools=[retrieve_docs] + memory_tools,
    prompt=create_eager_prompt_function(rag_prompt, store), # <-- FIX: Pass store directly
    name="rag_agent",
)

@tool
def execute_api_call(api_endpoint: str, params: dict):
    """A tool for executing API calls."""
    print(f"Executing API call to {api_endpoint} with params {params}")
    return "API call result: Success"

api_prompt = """
You are an API execution agent.
Your role is to handle tasks where the user requests data from an external API.

Instructions:
1. ALWAYS call the `execute_api_call` tool to perform the request.
   - Pass the correct `api_endpoint` and `params`.
   - Do not attempt to fabricate or guess the API response.
2. If the request requires information from a document, FIRST consult the `rag_agent` via the supervisor before making the API call.
3. Return the API response in a clean, structured format (JSON or plain text summary).
4. Never provide the final answer yourself without calling the tool.
5. If the API call fails, return the error clearly with suggestions for correction.
"""

api_agent = create_react_agent(
    model=llm,
    tools=[execute_api_call] + memory_tools,
    prompt=create_eager_prompt_function(api_prompt, store), # <-- FIX: Pass store directly
    name="api_agent",
)

@tool
def generate_form_tool(form_purpose: str, required_fields: list):
    """
    Generates a form based on a given purpose and a list of required fields.
    `required_fields` can be either a list of strings or a list of dicts with {"name": ..., "type": ...}.
    """
    form_html = f"<h2>{form_purpose}</h2>\n<form>\n"
    for field in required_fields:
        if isinstance(field, dict):
            field_name = field.get("name", "unknown_field")
            field_type = field.get("type", "text")
        else:
            field_name = str(field)
            field_type = "text"
        form_html += f"  <label for='{field_name}'>{field_name.replace('_', ' ').title()}:</label><br>\n"
        form_html += f"  <input type='{field_type}' id='{field_name}' name='{field_name}'><br><br>\n"
    form_html += "</form>"
    return form_html

form_prompt = """
You are a form generator agent.
When the user requests a form:
1. ALWAYS call the `generate_form_tool` with the form purpose and fields.
2. Do not attempt to answer without calling the tool.
3. Return the generated HTML form.
"""

form_agent = create_react_agent(
    model=llm,
    tools=[generate_form_tool] + memory_tools,
    prompt=create_eager_prompt_function(form_prompt, store), # <-- FIX: Pass store directly
    name="form_agent",
)

@tool
def superset_analytics_tool(query: str):
    """
    Runs an analytics query against Apache Superset to retrieve data.
    """
    try:
        superset_url = "http://your-superset-instance/api/v1/query"
        payload = {"query": query}
        response = requests.post(superset_url, json=payload)
        response.raise_for_status()
        data = response.json()
        return f"Successfully retrieved data for '{query}'. Here are the results:\n{data}"
    except Exception as e:
        return f"Error connecting to Apache Superset: {e}"

analytics_prompt = """
You are an analytics agent specialized in querying Apache Superset.

Instructions:
1. ALWAYS call the `superset_analytics_tool` to execute analytics queries.
   - If the query depends on document information, FIRST consult the `rag_agent` via the supervisor.
2. Return the results in a user-friendly format (summarize the data or provide JSON tables).
3. Do not fabricate results if the query fails — instead, return the error.
4. Never provide answers without calling the tool."
"""

analytics_agent = create_react_agent(
    model=llm,
    tools=[superset_analytics_tool] + memory_tools,
    prompt=create_eager_prompt_function(analytics_prompt, store), # <-- FIX: Pass store directly
    name="analytics_agent",
)

# --- Supervisor and Handoff Tools ---
def create_task_description_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."
    
    @tool(name, description=description)
    def handoff_tool(
        task_description: Annotated[str, "Description of what the next agent should do, including all of the relevant context."],
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = HumanMessage(content=task_description)
        return Command(
            goto=[Send(agent_name, {"messages": [task_description_message]})],
            graph=Command.PARENT,
        )
    return handoff_tool

assign_to_rag_agent = create_task_description_handoff_tool(
    agent_name="rag_agent", description="Assign task to the RAG agent for document Q&A.")
assign_to_api_agent = create_task_description_handoff_tool(
    agent_name="api_agent", description="Assign task to the API agent for external API calls.")
assign_to_form_agent = create_task_description_handoff_tool(
    agent_name="form_agent", description="Assign task to the form agent for generating forms.")
assign_to_analytics_agent = create_task_description_handoff_tool(
    agent_name="analytics_agent", description="Assign task to the analytics agent for data analysis.")

supervisor_prompt = """
You are a supervisor managing a team of specialized agents.

Team:
- **rag_agent**: Retrieves and summarizes information from documents using `retrieve_docs` also does the Q&A.
- **api_agent**: Executes external API calls with `execute_api_call`.
- **form_agent**: Generates forms using `generate_form_tool`.
- **analytics_agent**: Runs Superset analytics queries using `superset_analytics_tool`.

Rules:
1. Anwser with a friendly warm, welcoming tone. it is general question eg "Hi, how are you?"
2. If the request involves **documents**, you MUST FIRST send the request to `rag_agent` to extract relevant information.
3. If the request is purely API-related, route to `api_agent`.
4. If it is purely form generation, route to `form_agent`.
5. If it is purely analytics, route to `analytics_agent`.
6. When in doubt, always consult `rag_agent` first.
7. Your primary role: **route every request to the correct agent(s).**
8. **IMPORTANT**: Before routing, review the "Relevant Memories" section (if present) to see if the user's query is a follow-up or related to a past interaction. Use this context to make a more informed routing decision.
"""
supervisor_agent = create_react_agent(
    model=llm,
    tools=[
        assign_to_rag_agent,
        assign_to_api_agent,
        assign_to_form_agent,
        assign_to_analytics_agent,
    ] + memory_tools,
    prompt=create_eager_prompt_function(supervisor_prompt, store), # <-- FIX: Pass store directly
    name="supervisor",
)

# --- Graph Setup ---
graph_builder = StateGraph(MessagesState)

graph_builder.add_node("supervisor", supervisor_agent)
graph_builder.add_node("rag_agent", rag_agent)
graph_builder.add_node("api_agent", api_agent)
graph_builder.add_node("form_agent", form_agent)
graph_builder.add_node("analytics_agent", analytics_agent)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_edge("rag_agent", "supervisor")
graph_builder.add_edge("api_agent", "supervisor")
graph_builder.add_edge("form_agent", "supervisor")
graph_builder.add_edge("analytics_agent", "supervisor")

# This is our main LangGraph application
langgraph_app = graph_builder.compile(checkpointer=checkpointer)

# ===============================================================
#  3. API DATA MODELS (PYDANTIC)
# ===============================================================

class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

class HealthStatus(BaseModel):
    status: str
    vector_store_count: int
    timestamp: str

# ===============================================================
#  4. API ENDPOINTS
# ===============================================================

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serves the frontend HTML file."""
    return FileResponse("index.html")

@app.get("/health", response_model=HealthStatus)
async def get_health_status():
    """Provides the health and status of the backend system."""
    try:
        vector_count = vector_store._collection.count()
    except Exception:
        vector_count = -1
        
    return HealthStatus(
        status="online",
        vector_store_count=vector_count,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a document, processes it, and adds it to the vector store."""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
    
    try:
        # Save the uploaded file temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Load, split, and add the document to the vector store
        docs = load_file(file_path)
        all_splits = text_splitter.split_documents(docs)
        
        # MODIFIED: Check if documents already exist to avoid duplicates
        existing_ids = set()
        if hasattr(vector_store, '_collection'):
            existing_ids = set(vector_store._collection.get()['ids'])
        
        # Add only new documents that don't already exist in the vector store
        new_docs = []
        for doc in all_splits:
            # Create a unique ID based on content to check for duplicates
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content))
            if doc_id not in existing_ids:
                doc.metadata['doc_id'] = doc_id
                new_docs.append(doc)
        
        if new_docs:
            vector_store.add_documents(documents=new_docs)
            new_count = vector_store._collection.count()
            return {
                "message": f"Successfully uploaded and processed {file.filename}. Added {len(new_docs)} new document chunks.",
                "vector_count": new_count
            }
        else:
            return {
                "message": f"Document {file.filename} already exists in the vector store. No new chunks added.",
                "vector_count": vector_store._collection.count()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """Main endpoint to chat with the multi-agent system."""
    if not request.message or not request.user_id:
        raise HTTPException(status_code=400, detail="Message and User ID cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())
    
    config = {
        "configurable": {
            "thread_id": session_id,
            "user_id": request.user_id,
            "store": store
        }
    }
    
    try:
        # Invoke the LangGraph app
        final_state = langgraph_app.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )
        
        def extract_final_ai_response(messages: list) -> str:
            """Extracts only the most recent AI response(s) from the latest turn."""
            ai_responses = []
            
            # Find the last human message to identify the current turn
            last_human_index = -1
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    last_human_index = i
                    break
            
            # If no human message found, look at all messages
            if last_human_index == -1:
                last_human_index = 0
            
            # Collect AI responses only after the last human message
            for i in range(last_human_index + 1, len(messages)):
                msg = messages[i]
                if (hasattr(msg, 'content') and msg.content and 
                    not isinstance(msg, (HumanMessage, ToolMessage))):
                    ai_responses.append(msg.content.strip())
                elif isinstance(msg, AIMessage) and msg.content:
                    ai_responses.append(msg.content.strip())
            
            # Join AI responses from the current turn
            if ai_responses:
                return " ".join(ai_responses)
            else:
                return "I couldn't generate a proper response. Please try again."

        response_content = extract_final_ai_response(final_state["messages"])
        return ChatResponse(response=response_content, session_id=session_id)
        
    except Exception as e:
        print(f"Error during agent invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Note: The following session management endpoints are for demonstration with InMemorySaver.
# In a production environment, you would interact with a persistent checkpointer (e.g., Redis, Postgres).

@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Retrieves the message history for a given session."""
    # This is a conceptual implementation for InMemorySaver.
    # A real implementation would query the persistent store.
    try:
        # Getting the state doesn't give a nicely formatted history.
        # This is a limitation of this demo setup. We return a placeholder.
        # To get real history, you'd need to track messages separately or use a persistent checkpointer.
        return {"session_id": session_id, "messages": [{"role": "system", "content": "History retrieval is not fully supported with InMemorySaver. Start a new chat to see messages."}]}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found or error retrieving history: {e}")


# ===============================================================
#  5. SCRIPT EXECUTION
# ===============================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting Multi-Agent RAG System Backend...")
    # The frontend expects the API at port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)