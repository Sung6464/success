"""Adapted indexing pipeline using Proxy RAG."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

from utils.proxy_rag import index_document, build_tree, extract_file

logger = logging.getLogger(__name__)


class Index_Success_Stories:
    def __init__(self, sharepoint_manager, embedding_generator=None, vector_store=None):
        """Initialize adapted indexer class."""
        self.sharepoint_manager = sharepoint_manager
        logger.info("✅ Index_Success_Stories initialized with Proxy RAG indexer")

    def index_all_stories(self, folder_path: Optional[str] = None) -> Dict:
        """Alias for index_all_stories_new, indexing via Proxy RAG."""
        return self.index_all_stories_new(folder_path)

    def index_all_stories_new(self, folder_path: Optional[str] = None) -> Dict:
        """
        Index all stories from SharePoint.
        Downloads each document, extracts sections/images, builds section tree,
        and indexes it into the local FAISS index.
        """
        logger.info("Starting SharePoint story indexing pipeline via Proxy RAG...")
        try:
            files = self.sharepoint_manager._get_all_files_recursive(folder_path)
            logger.info(f"📁 Found {len(files)} SharePoint files to process")

            stats = {
                'total_files': len(files),
                'processed_files': 0,
                'failed_files': 0,
                'total_chunks': 0,  # total nodes in tree
                'total_stories': 0,
                'failed_files_list': []
            }

            for doc in files:
                file_name = doc.get('name', '')
                file_id = doc.get('id', '')

                # Download content
                try:
                    content = self.sharepoint_manager.download_file(file_id)
                    if not content:
                        logger.warning(f"⚠️ Could not download {file_name}")
                        stats['failed_files'] += 1
                        stats['failed_files_list'].append(file_name)
                        continue
                except Exception as e:
                    logger.error(f"  Download failed for {file_name}: {e}")
                    stats['failed_files'] += 1
                    stats['failed_files_list'].append(file_name)
                    continue

                # Save bytes to a temp file to run extraction
                temp_path = None
                try:
                    suffix = Path(file_name).suffix.lower()
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                        temp_file.write(content)
                        temp_path = Path(temp_file.name)

                    # Step 1: Extract PDF/DOCX/PPTX to markdown and images
                    doc_id, out_dir = extract_file(temp_path, original_name=file_name)

                    # Step 2: Build structure tree
                    md_path = out_dir / f"{doc_id}.md"
                    build_tree(doc_id, md_path)

                    # Step 3: Embed tree nodes and index in FAISS
                    # Prepare doc-level metadata to store in FAISS metadata
                    doc_metadata = {
                        'url': doc.get('webUrl', ''),
                        'graph_url': doc.get('@microsoft.graph.downloadUrl', ''),
                        'category': 'general',  # Default category
                    }
                    num_nodes = index_document(doc_id, doc_metadata=doc_metadata)

                    stats['processed_files'] += 1
                    stats['total_chunks'] += num_nodes
                    stats['total_stories'] += 1
                    logger.info(f"✅ Indexed '{file_name}' with {num_nodes} nodes/chunks")

                except Exception as e:
                    logger.error(f"  Failed to process {file_name}: {e}")
                    stats['failed_files'] += 1
                    stats['failed_files_list'].append(file_name)
                finally:
                    # Clean up temp file
                    if temp_path and temp_path.exists():
                        try:
                            temp_path.unlink()
                        except Exception:
                            pass

            logger.info(f"\n🎉 Indexing complete!")
            logger.info(f"   Processed: {stats['processed_files']}/{stats['total_files']} files")
            logger.info(f"   Total stories: {stats['total_stories']}")
            logger.info(f"   Total chunks/nodes: {stats['total_chunks']}")
            logger.info(f"   Failed: {stats['failed_files']}")

            return stats

        except Exception as e:
            logger.error(f"Failed to process files from SharePoint: {e}")
            return {
                'total_files': 0,
                'processed_files': 0,
                'failed_files': 0,
                'total_chunks': 0,
                'total_stories': 0,
                'failed_files_list': []
            }
