"""
SharePoint Delta Handler
Tracks changes in SharePoint and processes new/updated files
Uses PostgreSQL for tracking instead of Cosmos DB
"""
import os
import json
import logging
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
import msal
import requests
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Import database models
from .db_models import DatabaseManager, SharePointDeltaLink, SharePointFileTracking

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SharePointDeltaHandler:
    """
    Handles delta changes in SharePoint by tracking file modifications
    Uses PostgreSQL to store state of processed files
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        site_hostname: str,
        site_path: str,
        postgres_connection_string: str,
        sharepoint_manager,
        embedding_generator,
        vector_store,
        drive_id: Optional[str] = None,
        default_folder_path: str = ""
    ):
        """
        Initialize Delta Handler
        
        Args:
            client_id: Azure AD app client ID
            client_secret: Azure AD app client secret
            tenant_id: Azure AD tenant ID
            site_hostname: SharePoint site hostname (e.g., 'yourtenant.sharepoint.com')
            site_path: Site path (e.g., '/sites/YourSite')
            postgres_connection_string: PostgreSQL connection string
            drive_id: Optional drive ID to track specific document library
            default_folder_path: Optional default folder path to track (e.g., 'Shared Documents/Projects')
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.site_hostname = site_hostname
        self.site_path = site_path
        self.drive_id = drive_id
        self.default_folder_path = default_folder_path
        self.access_token = None
        self.site_id = None
        
        # Initialize MSAL
        self.authority = f"https://login.microsoftonline.com/{tenant_id}"
        self.scope = ["https://graph.microsoft.com/.default"]
        self.app = msal.ConfidentialClientApplication(
            client_id,
            authority=self.authority,
            client_credential=client_secret
        )
        
        # Initialize PostgreSQL Database Manager
        self.db_manager = DatabaseManager(postgres_connection_string)
        self.sharepoint_manager = sharepoint_manager
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.db_manager.initialize()
        self.db_manager.create_tables()  # Ensure tables exist
        
    def authenticate(self) -> bool:
        """Authenticate with Microsoft Graph API"""
        try:
            result = self.app.acquire_token_for_client(scopes=self.scope)
            if "access_token" in result:
                self.access_token = result["access_token"]
                logger.info("✓ Authenticated with Microsoft Graph API")
                return True
            else:
                logger.error(f"✗ Authentication failed: {result.get('error_description')}")
                return False
        except Exception as e:
            logger.error(f"✗ Authentication error: {str(e)}")
            return False
    
    def _ensure_authenticated(self):
        """Ensure valid access token"""
        if not self.access_token:
            self.authenticate()
    
    def get_site_id(self) -> Optional[str]:
        """Get SharePoint site ID"""
        self._ensure_authenticated()
        
        try:
            site_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_hostname}:{self.site_path}"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = requests.get(site_url, headers=headers)
            response.raise_for_status()
            
            site_data = response.json()
            self.site_id = site_data.get('id')
            logger.info(f"✓ Retrieved site ID: {self.site_id}")
            return self.site_id
        except Exception as e:
            logger.error(f"✗ Error getting site ID: {str(e)}")
            return None
    
    def _get_files_in_folder(self, folder_path: str) -> List[Dict]:
        """
        Get all files in a specific folder (non-recursive)
        Uses the same API pattern as sharepoint_manager
        """
        self._ensure_authenticated()
        
        if not self.site_id:
            self.get_site_id()
        
        try:
            # Build URL exactly like sharepoint_manager does
            if folder_path:
                folder_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/root:/{folder_path}:/children"
            else:
                folder_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/root/children"
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(folder_url, headers=headers)
            response.raise_for_status()
            
            items = response.json().get('value', [])
            
            # Filter to only files
            files = [item for item in items if 'file' in item]
            
            logger.info(f"✓ Found {len(files)} files in folder: {folder_path or 'root'}")
            return files
            
        except Exception as e:
            logger.error(f"✗ Error getting files from folder: {str(e)}")
            return []
    
    def _get_all_files_recursive(self, folder_path: str = "") -> List[Dict]:
        """
        Recursively get all files from folder and subfolders
        Uses the same logic as sharepoint_manager for consistency
        """
        all_files = []
        
        if not self.site_id:
            self.get_site_id()

        try:
            # Build URL
            if folder_path:
                folder_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/root:/{folder_path}:/children"
            else:
                folder_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/root/children"
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(folder_url, headers=headers)
            response.raise_for_status()
            
            items = response.json().get('value', [])
            
            for item in items:
                # If it's a folder, recurse
                if 'folder' in item:
                    subfolder_path = f"{folder_path}/{item['name']}" if folder_path else item['name']
                    logger.debug(f"📁 Recursing into: {subfolder_path}")
                    all_files.extend(self._get_all_files_recursive(subfolder_path))
                
                # If it's a file, add it
                elif 'file' in item:
                    all_files.append(item)
                    
            logger.debug(f"✓ Found {len(all_files)} total files in: {folder_path or 'root'}")
        except Exception as e:
            logger.error(f"✗ Error listing folder contents: {str(e)}")
        
        return all_files
    
    def get_delta_changes(self, folder_path: str = "", include_subfolders: bool = True) -> Dict:
        """
        Get delta changes using Microsoft Graph Delta query
        
        On first run (no stored delta link):
        - Uses _get_all_files_recursive() to scan folder structure directly
        - This ensures we get all files in the folder and subfolders
        
        On subsequent runs (with stored delta link):
        - Uses the stored delta link to get only changes since last run
        - This is efficient and catches all modifications
        
        Args:
            folder_path: Optional folder path to track (e.g., "General/Publish Ready Success Stories/IBU/ANZ")
            include_subfolders: If True (default), includes all subfolders recursively
        
        Returns:
            Dictionary with 'new', 'updated', and 'deleted' file lists
        """
        self._ensure_authenticated()
        
        if not self.site_id:
            self.get_site_id()
        
        try:
            # Check if we have a stored delta link
            delta_key = folder_path if folder_path else ""
            delta_link = self._get_stored_delta_link(delta_key)
            
            if not delta_link:
                # First run: Use direct folder scanning (same approach as sharepoint_manager)
                logger.info(f"First run - scanning folder structure directly: {folder_path or 'root'}")
                
                if include_subfolders:
                    all_items = self._get_all_files_recursive(folder_path)
                else:
                    all_items = self._get_files_in_folder(folder_path)
                
                # All items are "new" on first run
                changes = {
                    'new': all_items,
                    'updated': [],
                    'deleted': []
                }
                
                # Now get a delta link for future runs
                if folder_path:
                    # For specific folder, we still use root delta since folder delta is not recursive
                    url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/root/delta"
                else:
                    url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/root/delta"
                
                headers = {"Authorization": f"Bearer {self.access_token}"}
                
                # Consume the delta to get the delta link (don't use the results)
                while url:
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    
                    url = data.get('@odata.nextLink') or data.get('@odata.deltaLink')
                    
                    if '@odata.deltaLink' in data:
                        self._store_delta_link(data['@odata.deltaLink'], delta_key)
                        logger.info(f"✓ Stored delta link for future runs")
                        break
                
                logger.info(f"✓ Scan complete: {len(all_items)} files found")
                
            else:
                # Subsequent runs: Use delta link for incremental changes
                logger.info(f"Using stored delta link for incremental changes")
                
                headers = {"Authorization": f"Bearer {self.access_token}"}
                all_changes = []
                url = delta_link
                
                # Handle pagination
                while url:
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    
                    all_changes.extend(data.get('value', []))
                    
                    url = data.get('@odata.nextLink') or data.get('@odata.deltaLink')
                    
                    if '@odata.deltaLink' in data:
                        self._store_delta_link(data['@odata.deltaLink'], delta_key)
                        break
                
                # Categorize the changes
                changes = self._categorize_changes(all_changes)
                
                logger.info(f"✓ Delta changes: {len(changes['new'])} new, "
                           f"{len(changes['updated'])} updated, "
                           f"{len(changes['deleted'])} deleted")
            
            return changes
            
        except Exception as e:
            logger.error(f"✗ Error getting delta changes: {str(e)}")
            return {'new': [], 'updated': [], 'deleted': []}
    

    
    def _categorize_changes(self, items: List[Dict]) -> Dict:
        """Categorize items into new, updated, or deleted"""
        new_files = []
        updated_files = []
        deleted_files = []
        
        for item in items:
            # Skip folders
            if 'folder' in item:
                continue
            
            file_id = item.get('id')
            
            # Check if deleted
            if item.get('deleted'):
                deleted_files.append(item)
                self._mark_file_deleted(file_id)
                continue
            
            # Check if file (not folder)
            if 'file' not in item:
                continue
            
            # Check if we've seen this file before
            stored_state = self._get_file_state(file_id)
            
            if not stored_state:
                # New file
                new_files.append(item)
                self._store_file_state(item)
            else:
                # Check if modified
                last_modified = item.get('lastModifiedDateTime')
                stored_modified = stored_state.get('last_modified')
                
                if last_modified != stored_modified:
                    updated_files.append(item)
                    self._store_file_state(item)
        
        return {
            'new': new_files,
            'updated': updated_files,
            'deleted': deleted_files
        }
    
    def _get_file_state(self, file_id: str) -> Optional[SharePointFileTracking]:
        """Get stored file state from PostgreSQL"""
        session = self.db_manager.get_session()
        try:
            file_record = session.query(SharePointFileTracking).filter(
                SharePointFileTracking.file_id == file_id
            ).first()
            return file_record
        except SQLAlchemyError as e:
            logger.error(f"✗ Database error getting file state: {str(e)}")
            return None
        finally:
            session.close()
    
    def _store_file_state(self, file_item: Dict):
        """Store file state in PostgreSQL"""
        session = self.db_manager.get_session()
        try:
            file_id = file_item['id']
            
            # Check if record exists
            existing = session.query(SharePointFileTracking).filter(
                SharePointFileTracking.file_id == file_id
            ).first()
            
            # Parse SharePoint datetime
            sp_modified = file_item.get('lastModifiedDateTime')
            if sp_modified:
                sp_modified = datetime.fromisoformat(sp_modified.replace('Z', '+00:00'))
            
            sp_created = file_item.get('createdDateTime')
            if sp_created:
                sp_created = datetime.fromisoformat(sp_created.replace('Z', '+00:00'))
            
            if existing:
                # Update existing record
                existing.file_name = file_item['name']
                existing.file_path = file_item.get('parentReference', {}).get('path', '')
                existing.file_size = file_item.get('size')
                existing.web_url = file_item.get('webUrl')
                existing.download_url = file_item.get('@microsoft.graph.downloadUrl')
                existing.sharepoint_modified_time = sp_modified
                existing.processing_status = 'pending'
                existing.updated_at = datetime.utcnow()
            else:
                # Create new record
                new_file = SharePointFileTracking(
                    file_id=file_id,
                    site_id=self.site_id or '',
                    drive_id=self.drive_id or '',
                    file_name=file_item['name'],
                    file_path=file_item.get('parentReference', {}).get('path', ''),
                    file_extension=Path(file_item['name']).suffix,
                    file_size=file_item.get('size'),
                    web_url=file_item.get('webUrl'),
                    download_url=file_item.get('@microsoft.graph.downloadUrl'),
                    sharepoint_created_time=sp_created,
                    sharepoint_modified_time=sp_modified,
                    processing_status='pending',
                    is_deleted=False
                )
                session.add(new_file)
            
            session.commit()
            logger.debug(f"✓ Stored file state: {file_item['name']}")
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"✗ Database error storing file state: {str(e)}")
        finally:
            session.close()
    
    def _mark_file_deleted(self, file_id: str):
        """Mark file as deleted in PostgreSQL"""
        session = self.db_manager.get_session()
        try:
            file_record = session.query(SharePointFileTracking).filter(
                SharePointFileTracking.file_id == file_id
            ).first()
            
            if file_record:
                file_record.is_deleted = True
                file_record.processing_status = 'deleted'
                file_record.updated_at = datetime.utcnow()
                session.commit()
                logger.debug(f"✓ Marked file as deleted: {file_id}")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"✗ Database error marking file deleted: {str(e)}")
        finally:
            session.close()
    
    def _get_stored_delta_link(self, folder_path: str = "") -> Optional[str]:
        """Get stored delta link from PostgreSQL"""
        session = self.db_manager.get_session()
        try:
            # Use site_id, drive_id, and folder_path as the identifier for delta link
            folder_key = folder_path.replace('/', '_').replace(' ', '_') if folder_path else 'root'
            delta_id = f"{self.site_id}_{self.drive_id or 'default'}_{folder_key}"
            
            delta_record = session.query(SharePointDeltaLink).filter(
                SharePointDeltaLink.id == delta_id
            ).first()
            
            if delta_record:
                return delta_record.delta_link
            return None
        except SQLAlchemyError as e:
            logger.error(f"✗ Database error getting delta link: {str(e)}")
            return None
        finally:
            session.close()
    
    def _store_delta_link(self, link: str, folder_path: str = ""):
        """Store delta link in PostgreSQL"""
        session = self.db_manager.get_session()
        try:
            folder_key = folder_path.replace('/', '_').replace(' ', '_') if folder_path else 'root'
            delta_id = f"{self.site_id}_{self.drive_id or 'default'}_{folder_key}"
            
            # Check if record exists
            existing = session.query(SharePointDeltaLink).filter(
                SharePointDeltaLink.id == delta_id
            ).first()
            
            if existing:
                existing.delta_link = link
                existing.last_sync_time = datetime.utcnow()
            else:
                new_delta = SharePointDeltaLink(
                    id=delta_id,
                    site_id=self.site_id or '',
                    drive_id=self.drive_id,
                    library_name=folder_path or 'root',
                    delta_link=link,
                    last_sync_time=datetime.utcnow()
                )
                session.add(new_delta)
            
            session.commit()
            logger.debug("✓ Stored delta link")
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"✗ Database error storing delta link: {str(e)}")
        finally:
            session.close()
    
    def download_file(self, file_id: str) -> Optional[bytes]:
        """Download file content"""
        self._ensure_authenticated()
        
        try:
            file_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/items/{file_id}"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = requests.get(file_url, headers=headers)
            response.raise_for_status()
            
            file_data = response.json()
            download_url = file_data.get('@microsoft.graph.downloadUrl')
            
            if not download_url:
                return None
            
            response = requests.get(download_url)
            response.raise_for_status()
            
            return response.content
        except Exception as e:
            logger.error(f"✗ Error downloading file: {str(e)}")
            return None
    
    def process_changes(self, folder_path: str = "", include_subfolders: bool = True) -> Dict:
        """
        Main method to detect and process changes
        
        Args:
            folder_path: Optional folder path to track (e.g., 'Documents/Projects')
                        If empty, tracks entire site root
            include_subfolders: If True (default), includes all files in subfolders recursively
                               If False, only tracks files directly in the specified folder
        
        Returns:
            Dictionary with summary of processed files
        """
        if folder_path:
            recursive_note = "(including subfolders)" if include_subfolders else "(non-recursive)"
            logger.info(f"🔍 Starting delta change detection for folder: {folder_path} {recursive_note}")
        else:
            logger.info("🔍 Starting delta change detection for entire site...")
        
        changes = self.get_delta_changes(folder_path, include_subfolders)
        
        summary = {
            'new_files': len(changes['new']),
            'updated_files': len(changes['updated']),
            'deleted_files': len(changes['deleted']),
            'total_changes': len(changes['new']) + len(changes['updated']) + len(changes['deleted']),
            'timestamp': datetime.utcnow().isoformat(),
            'files_to_process': []
        }
        
        # Collect files that need processing (new + updated)
        files_to_process = changes['new'] + changes['updated']
        # print(f"Files to process --",files_to_process)
        
        for file_item in files_to_process:
            file_info = {
                'id': file_item['id'],
                'name': file_item['name'],
                'last_modified': file_item.get('lastModifiedDateTime'),
                'created_at': file_item.get('createdDateTime'),
                'size': file_item.get('size'),
                'web_url': file_item.get('webUrl'),
                'is_new': file_item in changes['new']
            }
            summary['files_to_process'].append(file_info)
        
        logger.info(f"✓ Delta scan complete: {summary['total_changes']} total changes")
        
        return summary

    
    def index_stories(self, new_summary: Dict) -> Dict:
        """
        Index all stories from SharePoint.
        This is the main function to process and store all documents.
        Args:
            folder_path: Optional SharePoint folder path to filter files
        Returns:
            Summary statistics
        """
        logger.info("Starting SharePoint story indexing pipeline...")

        try:
            # Use SharePointManager to extract all documents
            # documents = self.sharepoint_manager.extract_data(folder_path or "")
            # files = self.sharepoint_manager._get_all_files_recursive(folder_path)
            # logger.info(f"📁 Found {len(files)} SharePoint files to process")

            # print(f"Summary {type(new_summary)}: ", new_summary)

            stats = {
                'total_files': new_summary.get('new_files'),
                'processed_files': 0,
                'failed_files': 0,
                'total_chunks': 0,
                'total_stories': 0,
                'failed_files_list': []
            }

            for doc in new_summary.get('files_to_process'):
                file_name = doc.get('name', '')
                file_id = doc.get('id', '')

                # Download content
                try:
                    content = self.download_file(file_id)
                    if not content:
                        logger.warning(f"⚠️ Could not download {file_name}")
                        stats['failed_files'] += 1
                        stats['failed_files_list'].append(file_name)
                        continue
                except Exception as e:
                    logger.error(f"❌ Download failed for {file_name}: {e}")
                    stats['failed_files'] += 1
                    stats['failed_files_list'].append(file_name)
                    continue
                
                # Extract text
                file_content = None
                try:
                    file_content = self.sharepoint_manager.get_data_from_file(content)
                    if not file_content:
                        logger.warning(f"⚠️ No content extracted from {file_name}")
                        stats['failed_files'] += 1
                        stats['failed_files_list'].append(file_name)
                        continue
                except Exception as e:
                    logger.error(f"❌ Text extraction failed for {file_name}: {e}")
                    stats['failed_files'] += 1
                    stats['failed_files_list'].append(file_name)
                    continue
                
                extension = Path(file_name).suffix.lower()
                text = None
                
                # Build document in exact schema format
                doc_metadata = {
                    'id': file_id,
                    'name': file_name,
                    'content': file_content,
                    'text': text,
                    'metadata': {
                        'source': doc.get('web_url', ''),
                        'file_type': extension,
                        'size': doc.get('size', 0),
                        'created_at': doc.get('created_at', ''),
                        'modified_at': doc.get('last_modified', ''),
                        'url': doc.get('web_url', ''),
                        'sharepoint_id': file_id,
                        'mime_type': doc.get('file', {}).get('mimeType', ''),
                        'graph_url': doc.get('@microsoft.graph.downloadUrl', '')
                    }
                }

                # print(f"Documnet Details : {doc}")

                chunked = self.embedding_generator.chunk_and_embed_document(doc_metadata)
                chunk_texts = chunked['chunk_texts']
                embeddings = chunked['embeddings']
                metadata_list = chunked['metadata_list']

                if not chunk_texts or not embeddings:
                    logger.warning(f"⚠️ Skipping {file_name} - no chunks or embeddings generated")
                    stats['failed_files'] += 1
                    stats['failed_files_list'].append(file_name)
                    continue

                # Step 2: Prepare file-level metadata
                file_metadata = doc_metadata.get('metadata', {})
                category = file_metadata.get('category', 'general')
                source_file = file_name
                graph_url = file_metadata.get('graph_url', '')  # SharePoint download link
                file_type = file_metadata.get('file_type', '') 

                # Step 3: Store in PostgreSQL
                self.vector_store.insert_story(
                    story_id=file_id,
                    category=category,
                    source_file=source_file,
                    graph_url=graph_url,
                    file_type=file_type,
                    chunks=chunk_texts,
                    embeddings=embeddings,
                    metadata_list=metadata_list
                )

                stats['processed_files'] += 1
                stats['total_chunks'] += len(chunk_texts)
                stats['total_stories'] += 1
            
            
                logger.info(f"📄 Processing: {file_name}")
                logger.info(f"✅ Indexed '{file_name}' with {len(chunk_texts)} chunks")

        except Exception as e:
            logger.error(f"❌ Failed to process: {e}")
            # stats['failed_files'] += 1
            # stats['failed_files_list'].append(file_name)

            # logger.info(f"\n🎉 Indexing complete!")
            # logger.info(f"   Processed: {stats['processed_files']}/{stats['total_files']} files")
            # logger.info(f"   Total stories: {stats['total_stories']}")
            # logger.info(f"   Total chunks: {stats['total_chunks']}")
            # logger.info(f"   Failed: {stats['failed_files']}")

        return stats

    