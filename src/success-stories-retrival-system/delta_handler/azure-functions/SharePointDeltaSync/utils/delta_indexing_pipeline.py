"""
Delta Indexing Pipeline
Integrates SharePoint Delta Handler with Embedding Generation and Vector Storage

This pipeline:
1. Uses delta handler to detect new/updated files
2. Downloads files from SharePoint
3. Extracts text content
4. Generates embeddings using OpenAI
5. Stores embeddings in PostgreSQL vector database
"""

import logging
import hashlib
import io
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import tempfile
import os

# Azure Document Intelligence for text extraction
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

# Import our existing components
from .sharepoint_delta_handler import SharePointDeltaHandler
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeltaIndexingPipeline:
    """
    Complete pipeline for detecting, downloading, and indexing SharePoint files
    """
    
    def __init__(
        self,
        # SharePoint config
        sharepoint_client_id: str,
        sharepoint_client_secret: str,
        sharepoint_tenant_id: str,
        sharepoint_site_hostname: str,
        sharepoint_site_path: str,
        
        # OpenAI config
        openai_api_key: str,
        
        # PostgreSQL config
        postgres_connection_string: str,

        openai_model: str = "text-embedding-3-small",
        use_azure_openai: bool = False,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        
        # Azure Document Intelligence config
        doc_intelligence_endpoint: Optional[str] = None,
        doc_intelligence_key: Optional[str] = None,
        
        # Processing config
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        category: str = "success_stories"
    ):
        """
        Initialize the complete pipeline
        
        Args:
            sharepoint_*: SharePoint authentication and site details
            openai_*: OpenAI/Azure OpenAI configuration
            doc_intelligence_endpoint: Azure Document Intelligence endpoint (optional)
            doc_intelligence_key: Azure Document Intelligence API key (optional)
            postgres_connection_string: PostgreSQL connection string
            chunk_size: Text chunk size for embeddings
            chunk_overlap: Overlap between chunks
            category: Category for categorizing documents
        """
        # Initialize SharePoint Delta Handler
        self.delta_handler = SharePointDeltaHandler(
            client_id=sharepoint_client_id,
            client_secret=sharepoint_client_secret,
            tenant_id=sharepoint_tenant_id,
            site_hostname=sharepoint_site_hostname,
            site_path=sharepoint_site_path,
            postgres_connection_string=postgres_connection_string
        )
        
        # Initialize Embedding Generator
        self.embedding_generator = EmbeddingGenerator(
            api_key=openai_api_key,
            model=openai_model,
            use_azure=use_azure_openai,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=azure_api_version,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize Vector Store
        self.vector_store = VectorStore(
            connection_string=postgres_connection_string
        )
        
        # Initialize Azure Document Intelligence (optional)
        self.doc_intelligence_client = None
        if doc_intelligence_endpoint and doc_intelligence_key:
            try:
                self.doc_intelligence_client = DocumentIntelligenceClient(
                    doc_intelligence_endpoint,
                    AzureKeyCredential(doc_intelligence_key)
                )
                logger.info("✅ Azure Document Intelligence initialized")
            except Exception as e:
                logger.warning(f"⚠️  Could not initialize Document Intelligence: {e}")
        else:
            logger.info("ℹ️  Azure Document Intelligence not configured (using fallback extraction)")
        
        self.category = category
        logger.info("✅ Delta Indexing Pipeline initialized")    
        
        def extract_text_from_file(self, file_content: bytes, file_name: str) -> Optional[str]:
            """
            Extract text from various file types using Azure Document Intelligence
            Falls back to basic extraction if Document Intelligence is not available
            
            Args:
                file_content: Raw file bytes
                file_name: Name of file (used to determine type)
            
            Returns:
                Extracted text or None if extraction fails
            """
            file_ext = Path(file_name).suffix.lower()
            
            try:
                # Use Azure Document Intelligence if available
                if self.doc_intelligence_client:
                    return self._extract_with_document_intelligence(file_content, file_name)
                
                # Fallback to basic extraction
                if file_ext == '.txt':
                    return file_content.decode('utf-8', errors='ignore')
                else:
                    logger.warning(f"Azure Document Intelligence not configured. Limited support for {file_ext}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error extracting text from {file_name}: {e}")
                return None
    
    def _extract_with_document_intelligence(self, file_content: bytes, file_name: str) -> Optional[str]:
        """
        Extract text using Azure Document Intelligence (prebuilt-read model)
        Supports PDF, DOCX, PPTX, images, and more
        
        Args:
            file_content: Raw file bytes
            file_name: File name for logging
        
        Returns:
            Extracted text or None if extraction fails
        """
        try:
            logger.debug(f"Using Azure Document Intelligence for: {file_name}")
            
            # Analyze document with prebuilt-read model
            poller = self.doc_intelligence_client.begin_analyze_document(
                "prebuilt-read",
                body=AnalyzeDocumentRequest(bytes_source=file_content),
                locale="en-US"
            )
            result = poller.result()
            
            # Extract content
            text_content = result.content
            
            if text_content and len(text_content.strip()) > 0:
                logger.debug(f"✓ Extracted {len(text_content)} characters via Document Intelligence")
                return text_content
            else:
                logger.warning(f"No text content extracted from {file_name}")
                return None
                
        except Exception as e:
            logger.error(f"Document Intelligence extraction failed for {file_name}: {e}")
            return None
        
    def generate_story_id(self, file_id: str, file_name: str) -> str:
        """
        Generate unique story ID from file ID and name
        
        Args:
            file_id: SharePoint file ID
            file_name: File name
        
        Returns:
            Unique story ID
        """
        # Use file_id as primary identifier
        return f"{file_id}"
    
    
    def process_file(self, file_info: Dict) -> bool:
        """
        Process a single file: download, extract, embed, store
        
        Args:
            file_info: File information from delta handler
        
        Returns:
            True if successful, False otherwise
        """
        file_id = file_info['id']
        file_name = file_info['name']
        
        logger.info(f"📄 Processing file: {file_name}")
        
        try:
            # Step 1: Download file
            logger.info(f"   ⬇️  Downloading...")
            file_content = self.delta_handler.download_file(file_id)
            
            if not file_content:
                logger.error(f"   ❌ Failed to download file: {file_name}")
                return False
            
            # Step 2: Extract text
            logger.info(f"   📝 Extracting text...")
            text_content = self.extract_text_from_file(file_content, file_name)
            
            if not text_content or len(text_content.strip()) < 50:
                logger.warning(f"   ⚠️  Insufficient text content extracted: {file_name}")
                return False
            
            logger.info(f"   ✓ Extracted {len(text_content)} characters")
            
            # Step 3: Generate story ID
            # story_id = self.generate_story_id(file_id, file_name)
            
            # Step 4: Prepare document for embedding
            document = {
                'content': text_content,
                'name': file_name,
                'metadata': {
                    'name': file_name,
                    'file_type': Path(file_name).suffix.lower(),
                    'source': 'SharePoint',
                    'sharepoint_id': file_id,
                    'url': file_info.get('web_url', ''),
                    'graph_url': file_info.get('web_url', ''),
                    'size': file_info.get('size', 0),
                    'created_at': file_info.get('createdDateTime'),
                    'modified_at': file_info.get('last_modified'),
                    'mime_type': self._get_mime_type(file_name),
                    'is_new': file_info.get('is_new', False),
                    'indexed_at': datetime.utcnow().isoformat()
                }
            }
            
            # Step 5: Chunk and embed
            logger.info(f"   🔢 Generating embeddings...")
            embedded_doc = self.embedding_generator.chunk_and_embed_document(document)
            
            if not embedded_doc['chunk_texts']:
                logger.warning(f"   ⚠️  No chunks generated for: {file_name}")
                return False
            
            logger.info(f"   ✓ Generated {len(embedded_doc['chunk_texts'])} chunks")
            
            # Step 6: Store in vector database
            logger.info(f"   💾 Storing in database...")
            success = self.vector_store.insert_story(
                story_id=file_id,
                category=self.category,
                source_file=file_name,
                graph_url=file_info.get('web_url', ''),
                file_type=Path(file_name).suffix.lower(),
                chunks=embedded_doc['chunk_texts'],
                embeddings=embedded_doc['embeddings'],
                metadata_list=embedded_doc['metadata_list']
            )
            
            if success:
                logger.info(f"   ✅ Successfully indexed: {file_name}")
                return True
            else:
                logger.error(f"   ❌ Failed to store in database: {file_name}")
                return False
                
        except Exception as e:
            logger.error(f"   ❌ Error processing {file_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_mime_type(self, file_name: str) -> str:
        """Get MIME type from file extension"""
        ext = Path(file_name).suffix.lower()
        mime_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.txt': 'text/plain'
        }
        return mime_types.get(ext, 'application/octet-stream')
    
    def run_full_pipeline(self, folder_path: str = "", include_subfolders: bool = True) -> Dict:
        """
        Run the complete pipeline: detect changes, download, embed, store
        
        Args:
            folder_path: SharePoint folder path to scan
            include_subfolders: Include subfolders recursively
        
        Returns:
            Summary dictionary with statistics
        """
        logger.info("=" * 80)
        logger.info("DELTA INDEXING PIPELINE - FULL RUN")
        logger.info("=" * 80)
        logger.info(f"Folder: {folder_path or '(root)'}")
        logger.info(f"Include subfolders: {include_subfolders}")
        logger.info("=" * 80)
        
        # Step 1: Detect changes using delta handler
        logger.info("\n📊 STEP 1: Detecting Changes")
        logger.info("-" * 80)
        
        delta_summary = self.delta_handler.process_changes(folder_path, include_subfolders)
        
        logger.info(f"✓ Changes detected:")
        logger.info(f"   • New files:     {delta_summary['new_files']}")
        logger.info(f"   • Updated files: {delta_summary['updated_files']}")
        logger.info(f"   • Total changes: {delta_summary['total_changes']}")
        
        if delta_summary['total_changes'] == 0:
            logger.info("\n✅ No changes detected - pipeline complete")
            return {
                'changes_detected': 0,
                'files_processed': 0,
                'files_successful': 0,
                'files_failed': 0,
                'details': []
            }
        
        # Step 2: Process each file
        logger.info(f"\n🔄 STEP 2: Processing {len(delta_summary['files_to_process'])} Files")
        logger.info("-" * 80)
        
        results = {
            'changes_detected': delta_summary['total_changes'],
            'files_processed': 0,
            'files_successful': 0,
            'files_failed': 0,
            'details': []
        }
        
        for i, file_info in enumerate(delta_summary['files_to_process'], 1):
            logger.info(f"\n[{i}/{len(delta_summary['files_to_process'])}] {file_info['name']}")
            
            results['files_processed'] += 1
            success = self.process_file(file_info)
            
            if success:
                results['files_successful'] += 1
                results['details'].append({
                    'file_name': file_info['name'],
                    'status': 'success',
                    'is_new': file_info['is_new']
                })
            else:
                results['files_failed'] += 1
                results['details'].append({
                    'file_name': file_info['name'],
                    'status': 'failed',
                    'is_new': file_info['is_new']
                })
        
        # Step 3: Summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 Summary:")
        logger.info(f"   • Changes detected:  {results['changes_detected']}")
        logger.info(f"   • Files processed:   {results['files_processed']}")
        logger.info(f"   • Successfully indexed: {results['files_successful']}")
        logger.info(f"   • Failed:            {results['files_failed']}")
        logger.info("=" * 80)
        
        return results
    
    def process_specific_files(self, file_ids: List[str]) -> Dict:
        """
        Process specific files by their IDs (useful for reprocessing)
        
        Args:
            file_ids: List of SharePoint file IDs
        
        Returns:
            Summary dictionary
        """
        logger.info(f"Processing {len(file_ids)} specific files...")
        
        results = {
            'files_processed': 0,
            'files_successful': 0,
            'files_failed': 0,
            'details': []
        }
        
        for file_id in file_ids:
            # Create minimal file_info for processing
            file_info = {'id': file_id, 'name': f"File_{file_id}"}
            
            results['files_processed'] += 1
            success = self.process_file(file_info)
            
            if success:
                results['files_successful'] += 1
            else:
                results['files_failed'] += 1
        
        return results
