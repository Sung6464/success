import os
import azure.functions as func
from datetime import datetime, timezone
from typing import Dict

# # Import your delta handler
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# from utils.sharepoint_delta_handler import SharePointDeltaHandler
# from config import Config

# # Configure logging
# logger = logging.getLogger(__name__)

import sys
import logging
from pathlib import Path
from .utils.sharepoint_delta_handler import SharePointDeltaHandler
from .utils.db_models import DatabaseManager, SharePointDeltaLink
from .utils.embedding_generator import EmbeddingGenerator
from .utils.sharepoint_manager import SharePointManager
from .utils.vector_store import VectorStore
from .config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(mytimer: func.TimerRequest) -> None:
    """
    Main function triggered by timer
    Default schedule: Every hour (0 0 * * * *)
    
    This function:
    1. Connects to SharePoint using Microsoft Graph API
    2. Detects changes using delta queries
    3. Tracks files in PostgreSQL database
    4. Returns summary of detected changes
    """
    utc_timestamp = datetime.utcnow().replace(
        tzinfo=timezone.utc).isoformat()
    
    logger.info(f'🚀 SharePoint Delta Sync started at {utc_timestamp}')
    
    # Check if running on schedule or manually triggered
    if mytimer.past_due:
        logger.warning('⚠️ Timer is past due!')
    
    try:

        sharepoint_manager = SharePointManager(
        client_id= Config.SHAREPOINT_CLIENT_ID,
        client_secret=Config.SHAREPOINT_CLIENT_SECRET,
        tenant_id=Config.SHAREPOINT_TENANT_ID,
        site_hostname=Config.SHAREPOINT_SITE_HOSTNAME,
        site_path=Config.SHAREPOINT_SITE_PATH
        )

        embedding_generator = EmbeddingGenerator(
                api_key=Config.AZURE_OPENAI_EMBEDDING_API_KEY,
                model=Config.AZURE_OPENAI_EMBEDDING_MODEL,
                use_azure=True,
                azure_endpoint=Config.AZURE_OPENAI_EMBEDDING_API_BASE,
                azure_deployment=Config.AZURE_OPENAI_EMBEDDING_MODEL,
                api_version=Config.AZURE_OPENAI_EMBEDDING_API_VERSION
        )

        postgres_conn = Config.POSTGRES_CONNECTION_STRING
        # Recreate vector store with connection pooling (PRODUCTION-READY)
        vector_store = VectorStore(postgres_conn, min_conn=2, max_conn=10)

        delta_handler = SharePointDeltaHandler(
            client_id=Config.SHAREPOINT_CLIENT_ID,
            client_secret=Config.SHAREPOINT_CLIENT_SECRET,
            tenant_id=Config.SHAREPOINT_TENANT_ID,
            site_hostname=Config.SHAREPOINT_SITE_HOSTNAME,
            site_path=Config.SHAREPOINT_SITE_PATH,
            postgres_connection_string=Config.POSTGRES_CONNECTION_STRING,
            sharepoint_manager = sharepoint_manager,
            embedding_generator = embedding_generator,
            vector_store = vector_store
        )

        
        logger.info("✓ Delta handler initialized")

        folder_path = "General/Publish Ready Success Stories" 
        CLEAR_STATE = True

        # Detect changes in specified folder (or entire site if folder_path is empty)
        summary = delta_handler.process_changes(folder_path=folder_path)
        data = delta_handler.index_stories(summary)
        
        
        logger.info(f"Delta Summary: {summary['new_files']} new, "
                    f"{summary['updated_files']} updated, "
                    f"{summary['deleted_files']} deleted")
        
        # Log files to process
        if summary['files_to_process']:
            logger.info(f"Files tracked in database: {len(summary['files_to_process'])}")
            for file_info in summary['files_to_process'][:5]:  # Log first 5
                logger.info(f"  - {file_info['name']} ({'new' if file_info['is_new'] else 'updated'})")
            
            if len(summary['files_to_process']) > 5:
                logger.info(f"  ... and {len(summary['files_to_process']) - 5} more")
        else:
            logger.info("✓ No changes detected")
        
        # Log summary
        result = {
            'status': 'success',
            'timestamp': utc_timestamp,
            'summary': {
                'new_files': summary['new_files'],
                'updated_files': summary['updated_files'],
                'deleted_files': summary['deleted_files'],
                'total_changes': summary['total_changes']
            }
        }
        
        logger.info(f"✅ Delta sync completed successfully")
        
    except ValueError as ve:
        # Configuration errors
        logger.error(f"❌ Configuration error: {str(ve)}")
        raise
        
    except Exception as e:
        # Unexpected errors
        logger.error(f"❌ Error in delta sync: {str(e)}", exc_info=True)
        raise