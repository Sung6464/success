import logging
import time
import json
from typing import List, Dict, Optional
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_batch
from pgvector.psycopg2 import register_vector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Production-ready PostgreSQL database manager with pgvector for embeddings.
    
    Features:
    - Connection pooling for better resource management
    - Automatic reconnection on connection failures
    - Batch insert operations for better performance
    - Transaction management with rollback on errors
    - Comprehensive error handling and logging
    """
    
    def __init__(self, connection_string: str, min_conn: int = 1, max_conn: int = 10):
        """
        Initialize PostgreSQL connection pool.
        
        Args:
            connection_string: PostgreSQL connection string
                Format: "postgresql://user:password@host:port/database"
            min_conn: Minimum number of connections in pool (default: 1)
            max_conn: Maximum number of connections in pool (default: 10)
        """
        self.connection_string = connection_string
        self.min_conn = min_conn
        self.max_conn = max_conn
        self.pool = None
        self.logger = logging.getLogger(__name__)
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool."""
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                self.min_conn,
                self.max_conn,
                self.connection_string
            )
            self.logger.info(f"  Connection pool initialized ({self.min_conn}-{self.max_conn} connections)")
        except Exception as e:
            self.logger.error(f"  Connection pool initialization failed: {e}")
            raise
    
    def _get_connection(self):
        """
        Get a connection from the pool with retry logic.
        
        Returns:
            Database connection
        """
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                conn = self.pool.getconn()
                if conn:
                    # Register pgvector for this connection
                    register_vector(conn)
                    return conn
            except Exception as e:
                self.logger.warning(f"  Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise
        
        raise Exception("Failed to get database connection after retries")
    
    def _return_connection(self, conn, close: bool = False):
        """
        Return connection to pool.
        
        Args:
            conn: Database connection
            close: If True, close connection instead of returning to pool
        """
        try:
            if close:
                self.pool.putconn(conn, close=True)
            else:
                self.pool.putconn(conn)
        except Exception as e:
            self.logger.warning(f"  Error returning connection to pool: {e}")
    
    def create_tables(self, embedding_dimension: int = 1536) -> None:
        """Create necessary tables with proper vector dimensions - FIXED SCHEMA"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Drop existing tables to start fresh (IMPORTANT: This will delete all data!)
            cursor.execute("DROP TABLE IF EXISTS story_embeddings CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS stories CASCADE;")
            
            # Create stories table (main story metadata)
            # cursor.execute("""
            #     CREATE TABLE stories (
            #         story_id VARCHAR(255) PRIMARY KEY,
            #         full_content TEXT,
            #         category VARCHAR(100),
            #         source_file TEXT,
            #         blob_url TEXT,
            #         file_type VARCHAR(50),
            #         chunk_count INTEGER,
            #         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            #         updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            #     );
            # """)

            cursor.execute("""
                CREATE TABLE stories (
                    story_id VARCHAR(255) PRIMARY KEY,
                    file_name TEXT,
                    file_type VARCHAR(50),
                    category VARCHAR(100),
                    source TEXT,
                    url TEXT,
                    graph_url TEXT,
                    size BIGINT,
                    created_at TIMESTAMP,
                    modified_at TIMESTAMP,
                    sharepoint_id TEXT,
                    mime_type TEXT,
                    chunk_count INTEGER,
                    metadata JSONB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create story_embeddings table with dynamic dimensions
            # cursor.execute(f"""
            #     CREATE TABLE story_embeddings (
            #         id SERIAL PRIMARY KEY,
            #         story_id VARCHAR(255) NOT NULL,
            #         chunk_index INTEGER NOT NULL,
            #         chunk_text TEXT NOT NULL,
            #         embedding VECTOR({embedding_dimension}) NOT NULL,
            #         category VARCHAR(100),
            #         source_file TEXT,
            #         blob_url TEXT,
            #         file_type VARCHAR(50),
            #         total_chunks INTEGER,
            #         metadata JSONB,
            #         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            #         FOREIGN KEY (story_id) REFERENCES stories(story_id) ON DELETE CASCADE
            #     );
            # """)

            cursor.execute(f"""
                CREATE TABLE story_embeddings (
                    id SERIAL PRIMARY KEY,
                    story_id VARCHAR(255) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding VECTOR({embedding_dimension}) NOT NULL,
                    total_chunks INTEGER,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (story_id) REFERENCES stories(story_id) ON DELETE CASCADE
                );
            """)
            
            # Create indexes for faster queries (AFTER tables are created)
            cursor.execute("CREATE INDEX idx_story_embeddings_story_id ON story_embeddings(story_id);")
            # cursor.execute("CREATE INDEX idx_story_embeddings_category ON story_embeddings(category);")
            cursor.execute("CREATE INDEX idx_stories_story_id ON stories(story_id);")
            cursor.execute("CREATE INDEX idx_stories_category ON stories(category);")
            
            # Create vector index for faster similarity search
            # cursor.execute(f"""
            #     CREATE INDEX idx_story_embeddings_embedding ON story_embeddings 
            #     USING ivfflat (embedding vector_cosine_ops)
            #     WITH (lists = 100);
            # """)
            
            conn.commit()
            self.logger.info(f"  Tables created successfully with {embedding_dimension}-dimensional vectors")
            self.logger.info("  Indexes created for optimized queries")
            
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"  Error creating tables: {e}")
            raise
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)
    
    def insert_story(self, story_id: str, category: str, source_file: str,
                    graph_url: str, file_type: str, chunks: List[str],
                    embeddings: List[List[float]], metadata_list: List[Dict]) -> bool:
        """
        Insert a story and its embeddings using batch operations
        Now stores only relevant metadata in stories table.
        """
        conn = None
        max_retries = 1

        # Extract file-level metadata from the first chunk's metadata (assuming all chunks share the same file metadata)
        file_metadata = metadata_list[0] if metadata_list else {}

        for attempt in range(max_retries):
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Insert or update story metadata (no full_content)
                cursor.execute("""
                    INSERT INTO stories (
                        story_id, file_name, file_type, category, source, url, graph_url, size,
                        created_at, modified_at, sharepoint_id, mime_type, chunk_count, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (story_id)
                    DO UPDATE SET
                        file_name = EXCLUDED.file_name,
                        file_type = EXCLUDED.file_type,
                        category = EXCLUDED.category,
                        source = EXCLUDED.source,
                        url = EXCLUDED.url,
                        graph_url = EXCLUDED.graph_url,
                        size = EXCLUDED.size,
                        created_at = EXCLUDED.created_at,
                        modified_at = EXCLUDED.modified_at,
                        sharepoint_id = EXCLUDED.sharepoint_id,
                        mime_type = EXCLUDED.mime_type,
                        chunk_count = EXCLUDED.chunk_count,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    story_id,
                    file_metadata.get('name', source_file),
                    file_metadata.get('file_type', file_type),
                    category,
                    file_metadata.get('source', ''),
                    file_metadata.get('url', ''),
                    file_metadata.get('graph_url', graph_url),
                    file_metadata.get('size', 0),
                    file_metadata.get('created_at', None),
                    file_metadata.get('modified_at', None),
                    file_metadata.get('sharepoint_id', ''),
                    file_metadata.get('mime_type', ''),
                    len(chunks),
                    json.dumps(file_metadata)
                ))

                # Delete existing embeddings for this story
                cursor.execute("DELETE FROM story_embeddings WHERE story_id = %s", (story_id,))

                # Prepare batch data for embeddings
                embedding_data = [
                    (story_id, i, chunk, embedding, len(chunks), json.dumps(metadata))
                    for i, (chunk, embedding, metadata) in enumerate(zip(chunks, embeddings, metadata_list))
                ]

                # Batch insert embeddings
                execute_batch(cursor, """
                    INSERT INTO story_embeddings
                    (story_id, chunk_index, chunk_text, embedding, total_chunks, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, embedding_data, page_size=100)

                conn.commit()
                self.logger.info(f"  Successfully inserted story '{story_id}' with {len(chunks)} chunks")
                return True

            except Exception as e:
                if conn:
                    conn.rollback()

                if attempt < max_retries - 1:
                    self.logger.warning(f"   Retry {attempt + 1}/{max_retries} for story '{story_id}': {e}")
                    time.sleep(2)
                    continue
                else:
                    self.logger.error(f"  Error inserting story '{story_id}': {e}")
                    return False
            finally:
                if conn:
                    self._return_connection(conn)

        return False
    
    # def insert_story(self, story_id: str, category: str, source_file: str, 
    #                 blob_url: str, file_type: str, chunks: List[str], 
    #                 embeddings: List[List[float]], metadata_list: List[Dict]) -> bool:
    #     """
    #     Insert a story and its embeddings using batch operations
        
    #     Args:
    #         story_id: Unique identifier for the story
    #         category: Story category
    #         source_file: Source file name
    #         blob_url: Azure Blob Storage URL
    #         file_type: File type (pdf, docx, pptx, etc.)
    #         chunks: List of text chunks
    #         embeddings: List of embedding vectors
    #         metadata_list: List of metadata dicts for each chunk
            
    #     Returns:
    #         True if successful, False otherwise
    #     """
    #     conn = None
    #     max_retries = 1
        
    #     for attempt in range(max_retries):
    #         try:
    #             conn = self._get_connection()
    #             cursor = conn.cursor()
                
    #             # Combine all chunks into full content
    #             full_content = "\n\n".join(chunks)
                
    #             # Insert or update story metadata
    #             cursor.execute("""
    #                 INSERT INTO stories (story_id, full_content, category, source_file, blob_url, file_type, chunk_count)
    #                 VALUES (%s, %s, %s, %s, %s, %s, %s)
    #                 ON CONFLICT (story_id) 
    #                 DO UPDATE SET 
    #                     full_content = EXCLUDED.full_content,
    #                     category = EXCLUDED.category,
    #                     source_file = EXCLUDED.source_file,
    #                     blob_url = EXCLUDED.blob_url,
    #                     file_type = EXCLUDED.file_type,
    #                     chunk_count = EXCLUDED.chunk_count,
    #                     updated_at = CURRENT_TIMESTAMP
    #             """, (story_id, full_content, category, source_file, blob_url, file_type, len(chunks)))
                
    #             # Delete existing embeddings for this story
    #             cursor.execute("DELETE FROM story_embeddings WHERE story_id = %s", (story_id,))
                
    #             # Prepare batch data for embeddings
    #             embedding_data = [
    #                 (story_id, i, chunk, embedding, category, source_file, blob_url, 
    #                  file_type, len(chunks), json.dumps(metadata))
    #                 for i, (chunk, embedding, metadata) in enumerate(zip(chunks, embeddings, metadata_list))
    #             ]
                
    #             # Batch insert embeddings (much faster than individual inserts)
    #             execute_batch(cursor, """
    #                 INSERT INTO story_embeddings 
    #                 (story_id, chunk_index, chunk_text, embedding, category, source_file, 
    #                  blob_url, file_type, total_chunks, metadata)
    #                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    #             """, embedding_data, page_size=100)
                
    #             conn.commit()
    #             self.logger.info(f"  Successfully inserted story '{story_id}' with {len(chunks)} chunks")
    #             return True
                
    #         except Exception as e:
    #             if conn:
    #                 conn.rollback()
                
    #             if attempt < max_retries - 1:
    #                 self.logger.warning(f"   Retry {attempt + 1}/{max_retries} for story '{story_id}': {e}")
    #                 time.sleep(2)
    #                 continue
    #             else:
    #                 self.logger.error(f"  Error inserting story '{story_id}': {e}")
    #                 return False
    #         finally:
    #             if conn:
    #                 self._return_connection(conn)
        
    #     return False
    
    def search(self, query_embedding: List[float], top_k: int = 5, 
          category_filter: Optional[str] = None,
          similarity_threshold: float = 0.5) -> List[Dict]:
        """
        Search for similar stories using vector similarity (optimized schema)
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            category_filter: Optional category to filter by
            similarity_threshold: Minimum similarity score (default: 0.5)
            
        Returns:
            List of matching stories with metadata
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build query with optional category filter
            category_condition = ""
            params = [query_embedding]
            
            if category_filter:
                category_condition = "WHERE s.category = %s"
                params.append(category_filter)
            
            params.append(top_k)
            
            # Search using cosine similarity with optimized schema
            query = f"""
                SELECT 
                    e.story_id,
                    e.chunk_text,
                    e.chunk_index,
                    e.total_chunks,
                    1 - (e.embedding <=> %s::vector) as similarity,
                    s.file_name,
                    s.file_type,
                    s.category,
                    s.source,
                    s.url,
                    s.graph_url,
                    s.size,
                    s.created_at,
                    s.modified_at,
                    s.sharepoint_id,
                    s.mime_type,
                    e.metadata as chunk_metadata,
                    s.metadata as story_metadata
                FROM story_embeddings e
                INNER JOIN stories s ON e.story_id = s.story_id
                {category_condition}
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
            """
            
            params_with_embedding = [query_embedding] + params
            cursor.execute(query, params_with_embedding)
            
            results = []
            for row in cursor.fetchall():
                similarity = float(row[4])
                if similarity >= similarity_threshold:
                    def ensure_dict(val):
                        if isinstance(val, dict):
                            return val
                        if isinstance(val, str):
                            return json.loads(val)
                        return {}
                    results.append({
                        'story_id': row[0],
                        'chunk_text': row[1],
                        'chunk_index': row[2],
                        'total_chunks': row[3],
                        'similarity': similarity,
                        'file_name': row[5],
                        'file_type': row[6],
                        'category': row[7],
                        'source': row[8],
                        'url': row[9],
                        'graph_url': row[10],
                        'size': row[11],
                        'created_at': row[12],
                        'modified_at': row[13],
                        'sharepoint_id': row[14],
                        'mime_type': row[15],
                        'chunk_metadata': ensure_dict(row[16]),
                        'story_metadata': ensure_dict(row[17])
                    })
            
            self.logger.info(f"  Found {len(results)} matching chunks (similarity >= {similarity_threshold})")
            return results
            
        except Exception as e:
            self.logger.error(f"  Error searching stories: {e}")
            return []
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)

    # def search(self, query_embedding: List[float], top_k: int = 5, 
    #           category_filter: Optional[str] = None,
    #           similarity_threshold: float = 0.5) -> List[Dict]:
    #     """
    #     Search for similar stories using vector similarity
        
    #     Args:
    #         query_embedding: Query embedding vector
    #         top_k: Number of results to return
    #         category_filter: Optional category to filter by
            
    #     Returns:
    #         List of matching stories with metadata
    #     """
    #     conn = None
    #     try:
    #         conn = self._get_connection()
    #         cursor = conn.cursor()
            
    #         # Build query with optional category filter
    #         category_condition = ""
    #         params = [query_embedding]
            
    #         if category_filter:
    #             category_condition = "WHERE e.category = %s"
    #             params.append(category_filter)
            
    #         params.append(top_k)
            
    #         # Search using cosine similarity
    #         query = f"""
    #             SELECT 
    #                 e.story_id,
    #                 e.chunk_text,
    #                 e.chunk_index,
    #                 1 - (e.embedding <=> %s::vector) as similarity,
    #                 e.category,
    #                 e.source_file,
    #                 e.blob_url,
    #                 e.file_type,
    #                 e.total_chunks,
    #                 s.full_content
    #             FROM story_embeddings e
    #             LEFT JOIN stories s ON e.story_id = s.story_id
    #             {category_condition}
    #             ORDER BY e.embedding <=> %s::vector
    #             LIMIT %s
    #         """
            
    #         params_with_embedding = [query_embedding] + params
    #         cursor.execute(query, params_with_embedding)
            
    #         results = []
    #         for row in cursor.fetchall():
    #             similarity = float(row[3])
                
    #             # Filter by similarity threshold
    #             if similarity >= similarity_threshold:
    #                 results.append({
    #                     'story_id': row[0],
    #                     'chunk_text': row[1],
    #                     'chunk_index': row[2],
    #                     'similarity': similarity,
    #                     'category': row[4],
    #                     'source_file': row[5],
    #                     'blob_url': row[6],
    #                     'file_type': row[7],
    #                     'total_chunks': row[8],
    #                     'full_content': row[9]
    #                 })

    #         # results = []
    #         # for row in cursor.fetchall():
    #         #     results.append({
    #         #         'story_id': row[0],
    #         #         'chunk_text': row[1],
    #         #         'chunk_index': row[2],
    #         #         'similarity': float(row[3]),
    #         #         'category': row[4],
    #         #         'source_file': row[5],
    #         #         'blob_url': row[6],
    #         #         'file_type': row[7],
    #         #         'total_chunks': row[8],
    #         #         'full_content': row[9]
    #         #     })
            
    #         self.logger.info(f"  Found {len(results)} matching stories")
    #         return results
            
    #     except Exception as e:
    #         self.logger.error(f"  Error searching stories: {e}")
    #         return []
    #     finally:
    #         if conn:
    #             self._return_connection(conn)
    
    def get_story_count(self) -> int:
        """Get total number of stories in database"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM stories")
            return cursor.fetchone()[0]
        except Exception as e:
            self.logger.error(f"  Error getting story count: {e}")
            return 0
        finally:
            if conn:
                self._return_connection(conn)
    
    def get_embedding_count(self) -> int:
        """Get total number of embeddings in database"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM story_embeddings")
            return cursor.fetchone()[0]
        except Exception as e:
            self.logger.error(f"  Error getting embedding count: {e}")
            return 0
        finally:
            if conn:
                self._return_connection(conn)
    
    def close_all_connections(self):
        """Close all connections in the pool"""
        if self.pool:
            self.pool.closeall()
            self.logger.info("  All connections in pool closed")