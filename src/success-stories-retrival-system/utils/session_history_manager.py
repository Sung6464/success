import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
import certifi

class SessionHistoryManager:
    """
    Azure Cosmos DB optimized session history manager.
    
    Data Model Strategy:
    - Uses hierarchical partition key: [workspace_id, user_id, session_id]
    - Embeds messages within session document (up to 2MB limit)
    - Minimizes cross-partition queries
    """
    
    def __init__(self, mongo_client):
        try:
            self.client = mongo_client.client
            self.db = mongo_client.chatbot_db
            self.sessions_collection = self.db["ss_chat_history"]
            
            # Create indexes for efficient queries
            self._create_indexes()
            
            logging.info("✅ Connected to Azure Cosmos DB for MongoDB")
            
        except Exception as e:
            logging.error(f"❌ Error connecting to Azure Cosmos DB: {e}")
            raise

    def _create_indexes(self):
        """Create indexes optimized for Cosmos DB queries."""
        try:
            # Compound index for workspace + user queries (most common pattern)
            self.sessions_collection.create_index([
                ("workspace_id", 1),
                ("user_id", 1),
                ("updated_at", -1)
            ])
            
            # Index for session lookup
            self.sessions_collection.create_index([
                ("workspace_id", 1),
                ("user_id", 1),
                ("session_id", 1)
            ], unique=True)
            
            logging.info("✅ Indexes created successfully")
        except Exception as e:
            logging.warning(f"⚠️ Index creation warning: {e}")

    @staticmethod
    def create_session() -> str:
        """Generate a new session ID with timestamp."""
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        return session_id

    def create_new_session(self, workspace_id: str, user_id: str, session_id: str, title: str = "New Chat") -> Dict:
        """
        Create a new chat session document.
        
        This follows Azure Cosmos DB best practice of embedding related data
        within a single document to minimize cross-partition queries.
        """
        try:
            # session_id = self.create_session()
            
            # Embed session metadata and messages in single document
            session_doc = {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "session_id": session_id,
                "title": title,
                "messages": [],  # Embedded messages array
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "message_count": 0,
                "is_active": True
            }
            
            result = self.sessions_collection.insert_one(session_doc)
            
            logging.info(f"✅ Created new session: {session_id}")
            
            return {
                "session_id": session_id,
                "title": title,
                "created_at": session_doc["created_at"],
                "status": "success"
            }
            
        except Exception as e:
            logging.error(f"❌ Error creating session: {e}")
            return {"status": "error", "message": str(e)}

    def append_message(self, workspace_id: str, user_id: str, session_id: str, 
                      role: str, content: str, sources: List[Dict] = None) -> Optional[str]:
        """
        Append message to existing session document.
        
        Uses $push operator for atomic updates - more efficient than
        retrieving, modifying, and re-saving the entire document.
        """
        print("Inside append message:  ")
        try:
            
            # Check whether session is availbale or not
            sess_cnt = self.session_exists(workspace_id, user_id, session_id)
            if sess_cnt > 0:
                session_id = session_id
            else:
                logging.info(f" Session not found, creating new session: {session_id}")
                session_res = self.create_new_session(workspace_id, user_id, session_id, content[0:20])
                session_id = session_res.get('session_id')
                print(f"Created session with session id:", session_id)

            message = {
                "message_id": str(uuid.uuid4()),
                "role": role,
                "content": content,
                "sources": sources or [],
                "timestamp": datetime.utcnow()
            }
            
            # Atomic update using $push and $set
            result = self.sessions_collection.update_one(
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "session_id": session_id
                },
                {
                    "$push": {"messages": message},
                    "$set": {"updated_at": datetime.utcnow()},
                    "$inc": {"message_count": 1}
                }
            )

            print(f"Result : ", result)

            # if result.modified_count == 0:
            #     logging.warning(f"⚠️ Session not found, creating new one: {session_id}")
            #     # Auto-create session if it doesn't exist
            #     self.create_new_session(workspace_id, user_id, "Untitled Chat")
            #     # Retry append
            #     self.sessions_collection.update_one(
            #         {
            #             "workspace_id": workspace_id,
            #             "user_id": user_id,
            #             "session_id": session_id
            #         },
            #         {
            #             "$push": {"messages": message},
            #             "$set": {"updated_at": datetime.utcnow()},
            #             "$inc": {"message_count": 1}
            #         }
            #     )
            
            return message["message_id"]
            
        except Exception as e:
            logging.error(f"❌ Error appending message: {e}")
            return None

    def get_last_assistant_sources(self, workspace_id: str, user_id: str, session_id: str) -> List[Dict]:
        """
        Get sources from the last assistant message.
        Critical for chat continuation - allows reusing previous documents.
        
        Returns:
            List of source dicts with file_name, download_url, similarity, chunk_text, etc.
        """
        try:
            session = self.sessions_collection.find_one(
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "session_id": session_id
                },
                {"messages": 1}
            )
            
            if not session or "messages" not in session:
                logging.info("📭 No previous messages found")
                return []
            
            # Find last assistant message with sources
            for message in reversed(session["messages"]):
                if message.get("role") == "assistant" and message.get("sources"):
                    sources = message.get("sources", [])
                    logging.info(f"📚 Retrieved {len(sources)} sources from last assistant message")
                    return sources
            
            logging.info("📭 No assistant messages with sources found")
            return []
            
        except Exception as e:
            logging.error(f"❌ Error getting last assistant sources: {e}")
            return []
    
    def get_recent_sessions(self, workspace_id: str, user_id: str, limit: int = 10) -> List[Dict]:
        """
        Get recent sessions for a user.
        
        This query is optimized for Cosmos DB as it uses the partition key
        (workspace_id + user_id) and avoids cross-partition queries.
        """
        try:
            query = {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "is_active": True
            }
            
            # Project only necessary fields to reduce RU consumption
            projection = {
                "session_id": 1,
                "title": 1,
                "created_at": 1,
                "updated_at": 1,
                "message_count": 1,
                "_id": 0
            }
            
            sessions = list(
                self.sessions_collection
                .find(query, projection)
                .sort("updated_at", -1)
                .limit(limit)
            )
            
            return sessions
            
        except Exception as e:
            logging.error(f"❌ Error fetching recent sessions: {e}")
            return []

    def load_history(self, workspace_id: str, user_id: str, session_id: str) -> List[Dict]:
        """
        Load complete chat history for a session.
        
        Single document read - very efficient in Cosmos DB.
        """
        try:
            query = {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "session_id": session_id
            }
            
            session = self.sessions_collection.find_one(query)
            
            if not session:
                logging.warning(f"⚠️ Session not found: {session_id}")
                return []
            
            return session.get("messages", [])
            
        except Exception as e:
            logging.error(f"❌ Error loading history: {e}")
            return []

    def rename_session(self, workspace_id: str, user_id: str, session_id: str, new_title: str) -> Dict:
        """Rename a chat session."""
        try:
            result = self.sessions_collection.update_one(
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "session_id": session_id
                },
                {
                    "$set": {
                        "title": new_title,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            if result.modified_count > 0:
                return {"response": "success", "message": f"Session renamed to '{new_title}'"}
            else:
                return {"response": "error", "message": "Session not found"}
                
        except Exception as e:
            logging.error(f"❌ Error renaming session: {e}")
            return {"response": "error", "message": str(e)}

    def delete_session(self, workspace_id: str, user_id: str, session_id: str) -> Dict:
        """
        Soft delete a session by marking as inactive.
        
        This is better than hard delete for audit trails and recovery.
        """
        try:
            result = self.sessions_collection.update_one(
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "session_id": session_id
                },
                {
                    "$set": {
                        "is_active": False,
                        "deleted_at": datetime.utcnow()
                    }
                }
            )
            
            return {
                "response": "success" if result.modified_count > 0 else "not_found",
                "modified_count": result.modified_count
            }
            
        except Exception as e:
            logging.error(f"❌ Error deleting session: {e}")
            return {"response": "error", "message": str(e)}

    def get_session_info(self, workspace_id: str, user_id: str, session_id: str) -> Optional[Dict]:
        """Get session metadata without loading all messages."""
        try:
            query = {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "session_id": session_id
            }
            
            projection = {
                "session_id": 1,
                "title": 1,
                "created_at": 1,
                "updated_at": 1,
                "message_count": 1,
                "is_active": 1,
                "_id": 0
            }
            
            return self.sessions_collection.find_one(query, projection)
            
        except Exception as e:
            logging.error(f"❌ Error getting session info: {e}")
            return None
        
    def session_exists(self, workspace_id: str, user_id: str, session_id: str) -> bool:
        """
        Check if a session exists.
        
        Optimized for Azure Cosmos DB - uses partition key and minimal projection.
        Returns True if session exists and is active, False otherwise.
        """
        try:
            query = {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "session_id": session_id,
                "is_active": True
            }
            
            # Use count_documents with limit 1 for efficiency
            # This is more efficient than find_one when you only need existence check
            count = self.sessions_collection.count_documents(query, limit=1)
            
            return count
        
        except Exception as e:
            logging.error(f"❌ Error checking session existence: {e}")
            return False

    def close(self):
        """Close MongoDB connection."""
        if hasattr(self, 'client'):
            self.client.close()
            logging.info("✅ MongoDB connection closed")