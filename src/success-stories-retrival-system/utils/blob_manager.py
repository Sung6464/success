"""
Azure Blob Storage Manager for Success Stories
"""
from typing import List, Optional, Dict, BinaryIO
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, BlobSasPermissions, generate_blob_sas
from azure.core.exceptions import ResourceNotFoundError
import io
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlobStorageManager:
    """Manages Azure Blob Storage operations for success stories."""
    
    def __init__(self, connection_string: Optional[str] = None, container_name: Optional[str] = None):
        """
        Initialize the Blob Storage Manager.
        
        Args:
            connection_string: Azure Storage connection string (uses Config if not provided)
            container_name: Container name for success stories (uses Config if not provided)
        """
        self.connection_string = connection_string or Config.AZURE_STORAGE_CONNECTION_STRING
        self.container_name = container_name or Config.AZURE_CONTAINER_NAME
        
        if not self.connection_string:
            raise ValueError("Azure Storage connection string is required. Set AZURE_STORAGE_CONNECTION_STRING in .env")
        
        # Initialize blob service client
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)
        
        # Ensure container exists
        self._ensure_container_exists()
    
    def _ensure_container_exists(self):
        """Create container if it doesn't exist."""
        try:
            self.container_client.get_container_properties()
            logger.info(f"Container '{self.container_name}' exists")
        except ResourceNotFoundError:
            logger.info(f"Creating container '{self.container_name}'")
            self.container_client.create_container()
    
    def list_all_blobs(self, prefix: Optional[str] = None) -> List[Dict[str, any]]:
        """
        List all blobs in the container.
        
        Args:
            prefix: Optional prefix to filter blobs (e.g., 'pdf/' for only PDFs in pdf folder)
            
        Returns:
            List of blob metadata dictionaries
        """
        blobs = []
        blob_list = self.container_client.list_blobs(name_starts_with=prefix)
        
        for blob in blob_list:
            blob_info = {
                'name': blob.name,
                'size': blob.size,
                'content_type': blob.content_settings.content_type if blob.content_settings else None,
                'last_modified': blob.last_modified,
                'metadata': blob.metadata,
                'url': f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{self.container_name}/{blob.name}"
            }
            blobs.append(blob_info)
        
        logger.info(f"Found {len(blobs)} blobs in container '{self.container_name}'")
        return blobs
    
    def download_blob_to_memory(self, blob_name: str) -> bytes:
        """
        Download a blob to memory.
        
        Args:
            blob_name: Name of the blob to download
            
        Returns:
            Blob content as bytes
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            logger.info(f"Downloading blob '{blob_name}' to memory")
            blob_data = blob_client.download_blob().readall()
            return blob_data
        
        except ResourceNotFoundError:
            logger.error(f"Blob '{blob_name}' not found")
            raise
        except Exception as e:
            logger.error(f"Error downloading blob '{blob_name}': {str(e)}")
            raise
    
    def download_blob_to_file(self, blob_name: str, local_path: str) -> str:
        """
        Download a blob to a local file.
        
        Args:
            blob_name: Name of the blob to download
            local_path: Local file path to save the blob
            
        Returns:
            Path to the downloaded file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            logger.info(f"Downloading blob '{blob_name}' to '{local_path}'")
            
            with open(local_path, "wb") as file:
                blob_data = blob_client.download_blob()
                file.write(blob_data.readall())
            
            return local_path
        
        except Exception as e:
            logger.error(f"Error downloading blob '{blob_name}': {str(e)}")
            raise
    
    def download_all_blobs_to_folder(self, local_folder: str, prefix: Optional[str] = None) -> List[str]:
        """
        Download all blobs to a local folder.
        
        Args:
            local_folder: Local folder to download blobs to
            prefix: Optional prefix to filter blobs
            
        Returns:
            List of downloaded file paths
        """
        blobs = self.list_all_blobs(prefix=prefix)
        downloaded_files = []
        
        for blob in blobs:
            blob_name = blob['name']
            # Preserve folder structure
            local_path = os.path.join(local_folder, blob_name)
            
            try:
                self.download_blob_to_file(blob_name, local_path)
                downloaded_files.append(local_path)
            except Exception as e:
                logger.error(f"Failed to download '{blob_name}': {str(e)}")
        
        logger.info(f"Downloaded {len(downloaded_files)} files to '{local_folder}'")
        return downloaded_files
    
    def get_blob_url(self, blob_name: str) -> str:
        """
        Get the URL of a blob.
        
        Args:
            blob_name: Name of the blob
            
        Returns:
            URL of the blob
        """
        return f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"
    
    def upload_file(self, file_path: str, blob_name: Optional[str] = None, overwrite: bool = True) -> str:
        """
        Upload a file to blob storage.
        
        Args:
            file_path: Path to the local file
            blob_name: Name for the blob (uses filename if not provided)
            overwrite: Whether to overwrite existing blob
            
        Returns:
            Name of the uploaded blob
        """
        if not blob_name:
            blob_name = os.path.basename(file_path)
        
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        logger.info(f"Uploading '{file_path}' as '{blob_name}'")
        
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite)
        
        return blob_name
    
    def delete_blob(self, blob_name: str) -> bool:
        """
        Delete a blob from storage.
        
        Args:
            blob_name: Name of the blob to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            blob_client.delete_blob()
            logger.info(f"Deleted blob '{blob_name}'")
            return True
        except ResourceNotFoundError:
            logger.warning(f"Blob '{blob_name}' not found")
            return False
        except Exception as e:
            logger.error(f"Error deleting blob '{blob_name}': {str(e)}")
            raise

    def download_blob(self, blob_url: str) -> Optional[bytes]:
        """
        Download blob content from Azure Blob Storage.
        
        Args:
            blob_url: Full blob URL (e.g., https://account.blob.core.windows.net/container/path/file.pdf)
            
        Returns:
            Blob content as bytes, or None if download fails
        """
        try:
            # Parse the blob URL to extract container and blob name
            # URL format: https://{account}.blob.core.windows.net/{container}/{blob_path}
            from urllib.parse import urlparse
            
            parsed_url = urlparse(blob_url)
            path_parts = parsed_url.path.lstrip('/').split('/', 1)
            
            if len(path_parts) < 2:
                logger.error(f"❌ Invalid blob URL format: {blob_url}")
                return None
            
            container_name = path_parts[0]
            blob_name = path_parts[1]
            
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Download blob content
            logger.info(f"⬇️  Downloading blob: {blob_name}")
            blob_data = blob_client.download_blob().readall()
            
            logger.info(f"✅ Downloaded {len(blob_data)} bytes from {blob_name}")
            return blob_data
            
        except Exception as e:
            logger.error(f"❌ Error downloading blob from {blob_url}: {e}")
            return None
    
    def generate_download_url(self, blob_name: str, expiry_hours: int = 24) -> str:
        """
        Generate a download URL with SAS token for a blob.
        
        Args:
            blob_name: Name of the blob
            expiry_hours: Hours until the SAS token expires (default: 24)
            
        Returns:
            URL with SAS token for downloading
        """
        try:
            # Get account name and key from connection string
            conn_parts = dict(item.split('=', 1) for item in self.connection_string.split(';') if '=' in item)
            account_name = conn_parts.get('AccountName')
            account_key = conn_parts.get('AccountKey')
            
            if not account_name or not account_key:
                logger.error("Could not extract account name or key from connection string")
                return self.get_blob_url(blob_name)  # Return URL without SAS
            
            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
            )
            
            # Construct full URL with SAS token
            blob_url = f"https://{account_name}.blob.core.windows.net/{self.container_name}/{blob_name}?{sas_token}"
            
            logger.info(f"✅ Generated SAS URL for {blob_name} (expires in {expiry_hours}h)")
            return blob_url
            
        except Exception as e:
            logger.error(f"❌ Error generating SAS URL for {blob_name}: {e}")
            return self.get_blob_url(blob_name)  # Fallback to regular URL   


# Example usage and testing
if __name__ == "__main__":
    # Test the blob manager
    try:
        manager = BlobStorageManager()
        
        # List all blobs
        print("\n=== Listing all blobs ===")
        blobs = manager.list_all_blobs()
        for blob in blobs:
            print(f"- {blob['name']} ({blob['size']} bytes)")
        
        # Example: Download first blob
        if blobs:
            first_blob = blobs[0]['name']
            print(f"\n=== Downloading '{first_blob}' ===")
            content = manager.download_blob_to_memory(first_blob)
            print(f"Downloaded {len(content)} bytes")
        
    except Exception as e:
        print(f"Error: {e}")