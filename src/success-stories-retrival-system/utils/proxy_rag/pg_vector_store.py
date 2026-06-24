"""PostgreSQL vector store manager using pgvector."""
from __future__ import annotations

import json
import logging
import time
from typing import List, Dict, Optional
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_batch
from pgvector.psycopg2 import register_vector

from . import config

logger = logging.getLogger(__name__)


class PgVectorStore:
    """Manages connection pool and queries to PostgreSQL database with pgvector."""

    def __init__(self):
        self.connection_string = config.Config.POSTGRES_CONNECTION_STRING
        self.pool = None
        self._initialize_pool()

    def _initialize_pool(self):
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.connection_string
            )
            logger.info("Database connection pool initialized for PgVectorStore")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise e

    def _get_connection(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = self.pool.getconn()
                if conn:
                    register_vector(conn)
                    return conn
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise e
        raise Exception("Failed to acquire connection from database pool")

    def _return_connection(self, conn, close=False):
        try:
            self.pool.putconn(conn, close=close)
        except Exception as e:
            logger.warning(f"Error returning connection to pool: {e}")

    def create_tables(self, embedding_dimension: int = 3072) -> None:
        """Create proxy_story_embeddings table if not exists."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS proxy_story_embeddings (
                    id SERIAL PRIMARY KEY,
                    doc_id VARCHAR(255) NOT NULL,
                    node_id VARCHAR(50) NOT NULL,
                    breadcrumb TEXT NOT NULL,
                    title VARCHAR(255),
                    snippet TEXT,
                    text TEXT,
                    images JSONB,
                    url TEXT,
                    download_url TEXT,
                    category VARCHAR(100),
                    embedding VECTOR({embedding_dimension}) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_proxy_story_embeddings_doc_id ON proxy_story_embeddings(doc_id);")
            conn.commit()
            
            # Try to build vector index
            cursor = conn.cursor()
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_proxy_story_embeddings_embedding ON proxy_story_embeddings USING hnsw (embedding vector_cosine_ops);")
                conn.commit()
            except Exception:
                conn.rollback()
                cursor = conn.cursor()
                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_proxy_story_embeddings_embedding ON proxy_story_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
                    conn.commit()
                except Exception as e:
                    logger.warning(f"Could not create vector index (will perform exact searches): {e}")
                    conn.rollback()

            logger.info(f"PgVectorStore tables and indexes validated (dimension={embedding_dimension})")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error creating PgVectorStore tables: {e}")
            raise e
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)

    def delete_document(self, doc_id: str) -> None:
        """Remove all nodes and embeddings for doc_id."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM proxy_story_embeddings WHERE doc_id = %s", (doc_id,))
            conn.commit()
            logger.info(f"Deleted old proxy nodes for document: {doc_id}")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error deleting proxy document {doc_id}: {e}")
            raise e
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)

    def insert_nodes(self, nodes: List[Dict], embeddings: List[List[float]]) -> None:
        """Batch insert node definitions and their vector embeddings."""
        if not nodes:
            return
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Prepare data
            batch_data = []
            for n, emb in zip(nodes, embeddings):
                images_json = json.dumps(n.get("images", []))
                batch_data.append((
                    n["doc_id"],
                    n["node_id"],
                    n["breadcrumb"],
                    n.get("title"),
                    n.get("snippet"),
                    n.get("text"),
                    images_json,
                    n.get("url"),
                    n.get("download_url"),
                    n.get("category", "general"),
                    emb
                ))

            # Batch insert
            execute_batch(cursor, """
                INSERT INTO proxy_story_embeddings
                (doc_id, node_id, breadcrumb, title, snippet, text, images, url, download_url, category, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, batch_data, page_size=100)

            conn.commit()
            logger.info(f"Inserted {len(nodes)} proxy nodes into PostgreSQL")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error inserting proxy nodes: {e}")
            raise e
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Perform cosine similarity vector search using pgvector."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    doc_id,
                    node_id,
                    breadcrumb,
                    title,
                    snippet,
                    text,
                    images,
                    url,
                    download_url,
                    category,
                    1 - (embedding <=> %s::vector) as similarity
                FROM proxy_story_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))

            results = []
            for row in cursor.fetchall():
                # Parse images JSONB/TEXT
                try:
                    images_val = row[6]
                    if isinstance(images_val, str):
                        imgs = json.loads(images_val)
                    elif isinstance(images_val, list):
                        imgs = images_val
                    else:
                        imgs = []
                except Exception:
                    imgs = []

                results.append({
                    "doc_id": row[0],
                    "node_id": row[1],
                    "breadcrumb": row[2],
                    "title": row[3],
                    "snippet": row[4],
                    "text": row[5],
                    "images": imgs,
                    "url": row[7],
                    "download_url": row[8],
                    "category": row[9],
                    "score": float(row[10])
                })
            return results
        except Exception as e:
            logger.error(f"Error running vector database search: {e}")
            return []
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)

    def close_all_connections(self):
        if self.pool:
            self.pool.closeall()
            logger.info("PgVectorStore pool connections closed")
