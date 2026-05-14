"""
Database models for SharePoint Delta Tracking
PostgreSQL tables for tracking SharePoint files and delta links
"""
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class SharePointDeltaLink(Base):
    """
    Stores delta links for SharePoint sites/drives
    Used to track the last sync state for each SharePoint location
    """
    __tablename__ = 'sharepoint_delta_links'
    
    id = Column(String(255), primary_key=True)  # Unique identifier (e.g., site_id or drive_id)
    site_id = Column(String(255), nullable=False, index=True)
    drive_id = Column(String(255), nullable=True, index=True)
    library_name = Column(String(500), nullable=True)
    delta_link = Column(Text, nullable=False)  # The delta token/link from Microsoft Graph
    last_sync_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<SharePointDeltaLink(id='{self.id}', site_id='{self.site_id}', last_sync='{self.last_sync_time}')>"


class SharePointFileTracking(Base):
    """
    Tracks individual files from SharePoint
    Stores file metadata and processing status
    """
    __tablename__ = 'sharepoint_file_tracking'
    
    file_id = Column(String(255), primary_key=True)  # SharePoint file unique ID
    site_id = Column(String(255), nullable=False, index=True)
    drive_id = Column(String(255), nullable=True, index=True)
    file_name = Column(String(1000), nullable=False)
    file_path = Column(Text, nullable=False)
    file_extension = Column(String(50), nullable=True)
    file_size = Column(Integer, nullable=True)  # Size in bytes
    
    # SharePoint metadata
    web_url = Column(Text, nullable=True)
    download_url = Column(Text, nullable=True)
    
    # Timestamps
    sharepoint_created_time = Column(DateTime, nullable=True)
    sharepoint_modified_time = Column(DateTime, nullable=True)
    last_processed_time = Column(DateTime, nullable=True)
    
    # Processing status
    processing_status = Column(String(50), default='pending', nullable=False, index=True)
    # Status values: 'pending', 'processing', 'completed', 'failed', 'deleted'
    
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    
    # Tracking
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<SharePointFileTracking(file_id='{self.file_id}', name='{self.file_name}', status='{self.processing_status}')>"


class DatabaseManager:
    """
    Manages database connections and operations
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize database manager
        
        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string
        self.engine = None
        self.SessionLocal = None
        
    def initialize(self):
        """Initialize database engine and session maker"""
        try:
            self.engine = create_engine(
                self.connection_string,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,  # Recycle connections after 1 hour
                echo=False  # Set to True for SQL debugging
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            logger.info("Database engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def create_tables(self):
        """Create all tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created/verified successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise
    
    def get_session(self):
        """Get a new database session"""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.SessionLocal()
    
    def close(self):
        """Close database engine"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine disposed")
