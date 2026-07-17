import uuid
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

import os

DB_PATH = os.getenv("DATABASE_URL", "sqlite:///./ct200.db")
engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def new_id():
    return str(uuid.uuid4())


def utcnow():
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"
    id         = Column(String, primary_key=True, default=new_id)
    name       = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    versions   = relationship("Version", back_populates="document", order_by="Version.version_num")


class Version(Base):
    __tablename__ = "versions"
    id          = Column(String, primary_key=True, default=new_id)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    version_num = Column(Integer, nullable=False)
    ingested_at = Column(DateTime, default=utcnow)
    filename    = Column(String)
    document    = relationship("Document", back_populates="versions")
    nodes       = relationship("Node", back_populates="version", cascade="all, delete-orphan")


class Node(Base):
    __tablename__ = "nodes"
    id             = Column(String, primary_key=True, default=new_id)
    version_id     = Column(String, ForeignKey("versions.id"), nullable=False)
    logical_id     = Column(String, nullable=False)   # stays same across versions (e.g. "sec_3.2")
    section_number = Column(String)                     # "3.2", "2.1.1.1", None for title
    heading        = Column(String, nullable=False)
    level          = Column(Integer, nullable=False)    # 0=doc title, 1="1.", 2="1.1", etc
    body           = Column(Text, default="")
    content_hash   = Column(String, nullable=False)     # sha256 of normalized body
    parent_id      = Column(String, ForeignKey("nodes.id"), nullable=True)
    position       = Column(Integer, default=0)         # ordering among siblings
    has_table      = Column(Boolean, default=False)
    version        = relationship("Version", back_populates="nodes")
    parent         = relationship("Node", remote_side=[id], backref="children")


class Selection(Base):
    __tablename__ = "selections"
    id          = Column(String, primary_key=True, default=new_id)
    name        = Column(String, nullable=False)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    created_at  = Column(DateTime, default=utcnow)
    items       = relationship("SelectionItem", back_populates="selection", cascade="all, delete-orphan")
    document    = relationship("Document")


class SelectionItem(Base):
    __tablename__ = "selection_items"
    id              = Column(String, primary_key=True, default=new_id)
    selection_id    = Column(String, ForeignKey("selections.id"), nullable=False)
    node_id         = Column(String, ForeignKey("nodes.id"), nullable=False)
    version_id      = Column(String, ForeignKey("versions.id"), nullable=False)
    pinned_hash     = Column(String, nullable=False)   # content_hash at time of selection
    selection       = relationship("Selection", back_populates="items")
    node            = relationship("Node")
    version         = relationship("Version")


def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
