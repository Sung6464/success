from typing import List, Dict, Optional
from config import Config
import logging
import sys
from utils.vector_store import VectorStore
from utils.rag_pipeline import SuccessStoryChatBot
from utils.embedding_generator import EmbeddingGenerator
# from utils.sharepoint_manager import SharePointManager
# from utils.chatbot import SuccessStoryChatBot
from server import mcp
from langchain_core.messages import SystemMessage, HumanMessage
from main import session_manager

@mcp.tool()
def start_conversation():
    
    session_id = session_manager.create_session()
    logging.info(f"Session Created with id: {session_id}")
    return {"response":  session_id}

@mcp.tool()
def get_conversation_history(workspace_id: str, user_id: str, limit: Optional[int] = None) -> Dict:
    """
    Get recent chat sessions for a user.
    
    Args:
        workspace_id: Workspace identifier
        user_id: User identifier
        limit: Maximum number of sessions to return
    
    Returns:
        List of session objects with metadata
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    try:
        if limit == 5:
            sessions = session_manager.get_recent_sessions(workspace_id, user_id, limit)
            return {"response": sessions}
        else:
            sessions = session_manager.get_recent_sessions(workspace_id, user_id, 0)
            return {"response": sessions}
        
    except Exception as e:
        logging.error(f"Error fetching sessions: {e}")
        return {"error":f"Error occured while fetching conversation's {e}"}
    
@mcp.tool()   
def load_conversation(workspace_id: str, user_id: str, session_id: str) -> Dict:
    """Load the full conversation for a given user and session."""
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    try:
        response = session_manager.load_history(workspace_id, user_id, session_id)
        return {"response": response}
    except Exception as e:
        return {"error": f"Error occurred while loading conversation: {e}"}
    
@mcp.tool()
def rename_chat_session(workspace_id: str, user_id: str, session_id: str, new_title: str) -> Dict:
    """
    Rename a chat session.
    
    Args:
        workspace_id: Workspace identifier
        user_id: User identifier
        session_id: Session identifier
        new_title: New session title
    
    Returns:
        Dict with status and message
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    try:
        result = session_manager.rename_session(workspace_id, user_id, session_id, new_title)
        return result
    except Exception as e:
        logging.error(f"Error renaming session: {e}")
        return {"response": "error", "message": str(e)}
    
@mcp.tool()  
def delete_chat_session(workspace_id: str, user_id: str, session_id: str):
    """
    Delete a chat session (soft delete).
    
    Args:
        workspace_id: Workspace identifier
        user_id: User identifier
        session_id: Session identifier
    
    Returns:
        Dict with status and modified_count
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    try:
        result = session_manager.delete_session(workspace_id, user_id, session_id)
        return result
    except Exception as e:
        logging.error(f"Error deleting session: {e}")
        return {"response": "error", "message": str(e)}


def initialize_components():
    
    embedding_generator = EmbeddingGenerator(
            api_key=Config.AZURE_OPENAI_EMBEDDING_API_KEY,
            model=Config.AZURE_OPENAI_EMBEDDING_MODEL,
            use_azure=True,
            azure_endpoint=Config.AZURE_OPENAI_EMBEDDING_API_BASE,
            azure_deployment=Config.AZURE_OPENAI_EMBEDDING_MODEL,
            api_version=Config.AZURE_OPENAI_EMBEDDING_API_VERSION
        )

    postgres_conn = Config.POSTGRES_CONNECTION_STRING
    vector_store = VectorStore(postgres_conn, min_conn=2, max_conn=10)

    rag_pipeline_inference = SuccessStoryChatBot(embedding_generator, vector_store)
    
    return rag_pipeline_inference

@mcp.tool()
def message_gpt(message: str, workspace_id: str, user_id: str, session_id: str, history: Optional[List[Dict[str, str]]] = None):
    # Initialize components
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    try:
        print(" Initializing components...\n")
        rag_pipeline = initialize_components()
        print("\n Components initialized successfully. Processing message...\n")
    except Exception as e:
        print(f"\n Initialization failed: {e}\n")
        return {"response": f"Error: {e}", "sources": []}

    try:    
        user_message_id = session_manager.append_message(
            workspace_id=workspace_id,
            user_id=user_id,
            session_id=session_id,
            role="user",
            content=message,
            sources=[]
        )
        logging.info(f" User message saved: {user_message_id}")
        conversation_history = session_manager.load_history(workspace_id, user_id, session_id)
        
        # Format history for LLM context (last 5 messages for context window)
        history_context = ""
        formatted_history = []
        if len(conversation_history) > 1:  # More than just the current message
            recent_messages = conversation_history[-6:-1]  # Last 5 messages before current
            for msg in recent_messages:
                formatted_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        logging.info(" Searching for relevant success stories...")
        
        previous_sources = session_manager.get_last_assistant_sources(workspace_id, user_id, session_id)
        is_followup = rag_pipeline._is_followup_query(message, formatted_history)
        
        if is_followup and previous_sources:
            print(f" Follow-up detected! Including {len(previous_sources)} previous sources")


        search_results = rag_pipeline.search(
            query=message,
            top_k=5,
            similarity_threshold=0.4,
            previous_sources=previous_sources,
            is_followup=is_followup
        )

        response_dict = rag_pipeline.generate_response_structured(
            query=message,
            search_results=search_results,
            conversation_history=formatted_history  # Pass history here
        )
        
        sources_to_save = []
        for source in response_dict.get('sources', []):
            # Find matching search result to get chunk_text
            matching_result = next(
                (r for r in search_results if r.get('file_name') == source.get('file_name', '').replace('[1] ', '').replace('[2] ', '').replace('[3] ', '').replace('[4] ', '').replace('[5] ', '')),
                None
            )
            
            source_data = {
                'file_name': source.get('file_name'),
                'download_url': source.get('download_url'),
                'similarity': matching_result.get('similarity', 0.85) if matching_result else 0.85,
                'category': matching_result.get('category', 'N/A') if matching_result else 'N/A',
                'chunk_text': matching_result.get('chunk_text', matching_result.get('content', '')) if matching_result else '',
                'source': source.get('download_url')  # For compatibility
            }
            sources_to_save.append(source_data)
               
        assistant_message_id = session_manager.append_message(
            workspace_id=workspace_id,
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=response_dict.get('response'),
            sources=sources_to_save
        )
        logging.info(f" Assistant response saved: {assistant_message_id}")
        
        response = {
                'response': response_dict.get('response'),
                'sources': response_dict.get('sources')
            }
        return response
        
    except Exception as e:
        print(f"\nError: {e}\n")
        return {"Error": f"Error: {e}", "sources": []}
    
    finally:
        # Cleanup
        rag_pipeline.vector_store.close_all_connections()


