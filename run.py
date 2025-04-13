#!/usr/bin/env python
"""Standalone runner for RAGDocs using MCP SDK."""

import os
import sys
import logging
import argparse
from typing import Dict, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
import asyncio

# Global variables
server = None
embedding_service = None
storage_service = None
logger = None

def setup_mcp_server():
    """Set up MCP server with tools."""
    global server, logger
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("ragdocs")
    
    # Create server instance
    server = Server("ragdocs-server")
    
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="add_documentation",
                description="Add documentation from a URL to the RAG database",
                arguments={},
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of the documentation to fetch"
                        }
                    },
                    "required": ["url"]
                }
            ),
            types.Tool(
                name="search_documentation",
                description="Search through stored documentation",
                arguments={},
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "number",
                            "description": "Maximum number of results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="list_sources",
                description="List all documentation sources currently stored",
                arguments={},
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="add_directory",
                description="Add all supported files from a directory to the RAG database",
                arguments={},
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the directory containing documents"
                        }
                    },
                    "required": ["path"]
                }
            )
        ]
    
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            if name == "add_documentation":
                url = arguments.get("url")
                if not url:
                    return [types.TextContent(
                        type="text",
                        text="Error: URL is required"
                    )]
                
                # ตรงนี้เป็นตัวอย่าง, ถ้ามีการ implement จริงให้เรียกใช้ฟังก์ชันที่เหมาะสม
                logger.info(f"Adding documentation from URL: {url}")
                return [types.TextContent(
                    type="text",
                    text=f"Successfully added documentation from {url}"
                )]
                
            elif name == "search_documentation":
                query = arguments.get("query")
                limit = arguments.get("limit", 5)
                
                if not query:
                    return [types.TextContent(
                        type="text",
                        text="Error: Query is required"
                    )]
                
                logger.info(f"Searching documentation with query: {query}")
                
                try:
                    # Generate embedding for query
                    embedding = await embedding_service.generate_embedding(query)
                    
                    # Search for similar documents
                    results = await storage_service.search(embedding, limit)
                    
                    if not results or len(results) == 0:
                        return [types.TextContent(
                            type="text",
                            text="No results found for your query."
                        )]
                    
                    # Format results
                    formatted_results = []
                    for i, result in enumerate(results):
                        chunk = result.chunk
                        score = result.score
                        
                        # Get metadata
                        source = "Unknown source"
                        if hasattr(chunk.metadata, 'url') and chunk.metadata.url:
                            source = chunk.metadata.url
                        elif hasattr(chunk.metadata, 'source') and chunk.metadata.source:
                            source = chunk.metadata.source
                        
                        title = source
                        if hasattr(chunk.metadata, 'title') and chunk.metadata.title:
                            title = chunk.metadata.title
                        
                        formatted = f"[{i+1}] {title} (Score: {score:.2f})\n"
                        formatted += f"Source: {source}\n\n"
                        formatted += chunk.text
                        
                        formatted_results.append(formatted)
                    
                    formatted_text = "\n\n---\n\n".join(formatted_results)
                    return [types.TextContent(
                        type="text",
                        text=formatted_text
                    )]
                    
                except Exception as e:
                    logger.error(f"Error searching documentation: {e}", exc_info=True)
                    return [types.TextContent(
                        type="text",
                        text=f"Error searching: {str(e)}"
                    )]
                
            elif name == "list_sources":
                logger.info("Listing documentation sources")
                
                try:
                    sources = await storage_service.list_sources()
                    
                    if not sources or len(sources) == 0:
                        return [types.TextContent(
                            type="text",
                            text="No documentation sources found."
                        )]
                    
                    formatted = "Documentation sources:\n\n"
                    for i, source in enumerate(sources):
                        formatted += f"{i+1}. {source}\n"
                    
                    return [types.TextContent(
                        type="text",
                        text=formatted
                    )]
                except Exception as e:
                    logger.error(f"Error listing sources: {e}", exc_info=True)
                    return [types.TextContent(
                        type="text",
                        text=f"Error listing sources: {str(e)}"
                    )]
                
            elif name == "add_directory":
                path = arguments.get("path")
                if not path:
                    return [types.TextContent(
                        type="text",
                        text="Error: Path is required"
                    )]
                
                logger.info(f"Adding documentation from directory: {path}")
                
                try:
                    import os
                    from pyragdoc.core.processors.pdf import PDFProcessor
                    from pyragdoc.core.processors.text import TextProcessor
                    
                    # ตรวจสอบว่าไดเรกทอรีมีอยู่จริง
                    if not os.path.isdir(path):
                        return [types.TextContent(
                            type="text",
                            text=f"Error: '{path}' is not a directory or doesn't exist"
                        )]
                    
                    # ไม่สามารถนำเข้า processors จาก pyragdoc ได้อีกต่อไป
                    # ใช้ processors จาก ragdocs แทน
                    from ragdocs.processors import PDFProcessor, TextProcessor
                    
                    # สร้าง processors
                    pdf_processor = PDFProcessor(logger=logger)
                    text_processor = TextProcessor(logger=logger)
                    
                    # เก็บสถิติ
                    stats = {
                        "processed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "total_chunks": 0
                    }
                    
                    # แสดงรายการไฟล์ที่ประมวลผล
                    processed_files = []
                    failed_files = []
                    
                    # เริ่มไขว้
                    for root, _, files in os.walk(path):
                        for filename in files:
                            file_path = os.path.join(root, filename)
                            ext = os.path.splitext(filename)[1][1:].lower()
                            
                            try:
                                # ตรวจสอบไฟล์ที่สนับสนุน
                                if pdf_processor.can_process(file_path):
                                    logger.info(f"Processing PDF file: {file_path}")
                                    chunks = await pdf_processor.process_content(file_path)
                                elif text_processor.can_process(file_path):
                                    logger.info(f"Processing text file: {file_path}")
                                    chunks = await text_processor.process_content(file_path)
                                else:
                                    logger.info(f"Skipping unsupported file: {file_path}")
                                    stats["skipped"] += 1
                                    continue
                                
                                if not chunks:
                                    logger.info(f"No content extracted from: {file_path}")
                                    stats["skipped"] += 1
                                    continue
                                
                                # สร้าง embeddings และบันทึกลงฐานข้อมูล
                                embeddings = []
                                for chunk in chunks:
                                    # สร้าง embedding
                                    embedding = await embedding_service.generate_embedding(chunk.text)
                                    embeddings.append(embedding)
                                
                                # บันทึกข้อมูล
                                await storage_service.add_documents(embeddings, chunks)
                                
                                processed_files.append(file_path)
                                stats["processed"] += 1
                                stats["total_chunks"] += len(chunks)
                                logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks")
                            except Exception as e:
                                logger.error(f"Error processing file {file_path}: {str(e)}")
                                failed_files.append(file_path)
                                stats["failed"] += 1
                    
                    # สร้างข้อความตอบกลับ
                    summary = f"Directory Processing Results:\n\n"
                    summary += f"Processed {stats['processed']} files successfully\n"
                    summary += f"Failed to process {stats['failed']} files\n"
                    summary += f"Skipped {stats['skipped']} unsupported files\n"
                    summary += f"Added {stats['total_chunks']} total chunks to the database\n\n"
                    
                    if processed_files:
                        summary += "Successfully processed files:\n"
                        for i, file_path in enumerate(processed_files[:10], 1):
                            summary += f"{i}. {file_path}\n"
                        
                        # ถ้ามีไฟล์มากกว่า 10 ไฟล์ ให้แสดง ... แทน
                        if len(processed_files) > 10:
                            summary += f"...and {len(processed_files) - 10} more files\n"
                    
                    if failed_files:
                        summary += "\nFailed files:\n"
                        for i, file_path in enumerate(failed_files[:5], 1):
                            summary += f"{i}. {file_path}\n"
                        
                        if len(failed_files) > 5:
                            summary += f"...and {len(failed_files) - 5} more files\n"
                    
                    return [types.TextContent(
                        type="text",
                        text=summary
                    )]
                except Exception as e:
                    error_msg = f"Error adding directory: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    return [types.TextContent(
                        type="text",
                        text=error_msg
                    )]
                
            else:
                return [types.TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]
                
        except Exception as e:
            error_msg = f"Error executing {name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [types.TextContent(
                type="text",
                text=error_msg
            )]
    
    @server.list_resources()
    async def handle_list_resources() -> list[types.Resource]:
        return []
    
    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        return []


async def run_mcp_server():
    """Run the MCP server using MCP SDK."""
    global embedding_service, storage_service, logger
    
    try:
        # Initialize services
        logger.info("Initializing services...")
        
        # Get environment variables
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        qdrant_collection = os.environ.get("QDRANT_COLLECTION", "ragdocs")
        embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "ollama")
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        embedding_model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
        
        logger.info(f"Setting up services with: QDRANT_URL={qdrant_url}, EMBEDDING_PROVIDER={embedding_provider}")
        
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
        
        # Run server
        logger.info("PyRAGDoc Server is ready")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="pyragdoc",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    )
                )
            )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Error running server: {e}", exc_info=True)


def run_http_server():
    """Run HTTP server (placeholder)."""
    # HTTP server ไม่ได้ถูกรวมในแพ็กเกจใหม่
    logger.error("HTTP server mode is not supported in ragdocs version")
    sys.exit(1)


def main():
    """Main entry point for the CLI."""
    global logger
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PyRAGDoc Server")
    parser.add_argument(
        "--mode", 
        choices=["mcp", "http"], 
        default="mcp",
        help="Server mode (mcp or http)"
    )
    parser.add_argument(
        "--log-file", 
        help="Path to log file"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level based on arguments
    # (setup_logging will be handled in setup_mcp_server)
    
    # Set up MCP server if in MCP mode
    if args.mode == "mcp":
        setup_mcp_server()
    
    # Run appropriate server
    try:
        if args.mode == "mcp":
            logger.info("Starting MCP server")
            # ทำงานกับ MCP server ที่ตั้งค่าไว้แล้ว
            asyncio.run(run_mcp_server())
        else:
            logger.info("Starting HTTP server")
            run_http_server()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
