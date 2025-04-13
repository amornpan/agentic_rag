#!/usr/bin/env python3
"""Script to reimport 1-tool_use.pdf file into Qdrant."""

import os
import sys
import asyncio
import logging

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragdocs.processors import PDFProcessor

async def reimport_file():
    """Reimport 1-tool_use.pdf file."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("reimport")
    
    # Get environment variables
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_collection = os.environ.get("QDRANT_COLLECTION", "ragdocs")
    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "ollama")
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    
    logger.info(f"Setting up services with: QDRANT_URL={qdrant_url}, EMBEDDING_PROVIDER={embedding_provider}")
    
    # Initialize services
    try:
        # Create embedding service
        if embedding_provider.lower() == "ollama":
            from ragdocs.embeddings import OllamaEmbedding
            embedding_service = OllamaEmbedding(
                base_url=ollama_url,
                model=embedding_model,
                logger=logger
            )
            logger.info(f"Initialized Ollama embedding service with model: {embedding_model}")
        
        elif embedding_provider.lower() == "openai":
            from ragdocs.embeddings import OpenAIEmbedding
            embedding_service = OpenAIEmbedding(
                api_key=openai_api_key,
                model=embedding_model,
                logger=logger
            )
            logger.info(f"Initialized OpenAI embedding service with model: {embedding_model}")
        
        else:
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
        
        # Create storage service
        from ragdocs.storage import QdrantStorage
        storage_service = QdrantStorage(
            url=qdrant_url,
            collection_name=qdrant_collection,
            embedding_dimension=embedding_service.dimension,
            logger=logger
        )
        logger.info(f"Initialized Qdrant storage service at: {qdrant_url} with collection: {qdrant_collection}")
    except ImportError as e:
        logger.error(f"Error importing modules: {e}")
        raise
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise
    
    # Initialize PDF processor
    pdf_processor = PDFProcessor(logger=logger)
    
    # Process the PDF file
    # แก้ file_path ให้ตรงกับตำแหน่งไฟล์ในเครื่องของคุณ
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "1-tool_use.pdf")
    
    try:
        logger.info(f"Processing PDF file: {file_path}")
        chunks = await pdf_processor.process(file_path)
        
        if not chunks:
            logger.info(f"No content extracted from: {file_path}")
            return
        
        logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
        
        # Generate embeddings and store chunks
        embeddings = []
        for chunk in chunks:
            # Generate embedding
            embedding = await embedding_service.embed(chunk.text)
            embeddings.append(embedding)
        
        # Store chunks
        await storage_service.add(embeddings, chunks)
        
        logger.info(f"Successfully added {len(chunks)} chunks to Qdrant")
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
        
    # List sources to verify
    try:
        sources = await storage_service.list_sources()
        logger.info(f"Sources in database: {sources}")
    except Exception as e:
        logger.error(f"Error listing sources: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(reimport_file())
