import os
import sqlite3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import traceback
from dotenv import load_dotenv
from models import QueryRequest
from logging_config import logger
from db.db_manager import get_db_connection
from services.answer_service import generate_answer, process_multimodal_query, parse_llm_response
from services.search_service import find_similar_content, enrich_with_adjacent_chunks


load_dotenv()
API_KEY = os.getenv("API_KEY") 
DB_PATH = os.getenv("DB_PATH")

app = FastAPI(title="TDS helper API", description="API for Tools for Data science")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not API_KEY:
    logger.error("API_KEY environment variable is not set. The application will not function correctly.")

if not os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Create discourse_chunks table
    c.execute('''
    CREATE TABLE IF NOT EXISTS discourse_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER,
        topic_id INTEGER,
        topic_title TEXT,
        post_number INTEGER,
        author TEXT,
        created_at TEXT,
        likes INTEGER,
        chunk_index INTEGER,
        content TEXT,
        url TEXT,
        embedding BLOB
    )
    ''')
    
    # Create markdown_chunks table
    c.execute('''
    CREATE TABLE IF NOT EXISTS markdown_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_title TEXT,
        original_url TEXT,
        downloaded_at TEXT,
        chunk_index INTEGER,
        content TEXT,
        embedding BLOB
    )
    ''')
    conn.commit()
    conn.close()

@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    try:
        # Log the incoming request
        logger.info(f"Received query request: question='{request.question[:50]}...', image_provided={request.image is not None}")
        
        if not API_KEY:
            error_msg = "API_KEY environment variable not set"
            logger.error(error_msg)
            return JSONResponse(
                status_code=500,
                content={"error": error_msg}
            )
            
        conn = get_db_connection()
        
        try:
            # Process the query (handle text and optional image)
            logger.info("Processing query and generating embedding")
            query_embedding = await process_multimodal_query(
                request.question,
                request.image
            )
            
            # Find similar content
            logger.info("Finding similar content")
            relevant_results = await find_similar_content(query_embedding, conn)
            
            if not relevant_results:
                logger.info("No relevant results found")
                return {
                    "answer": "I couldn't find any relevant information in my knowledge base.",
                    "links": []
                }
            
            # Enrich results with adjacent chunks for better context
            logger.info("Enriching results with adjacent chunks")
            enriched_results = await enrich_with_adjacent_chunks(conn, relevant_results)
            
            # Generate answer
            logger.info("Generating answer")
            llm_response = await generate_answer(request.question, enriched_results)
            
            # Parse the response
            logger.info("Parsing LLM response")
            result = parse_llm_response(llm_response)
            
            # If links extraction failed, create them from the relevant results
            if not result["links"]:
                logger.info("No links extracted, creating from relevant results")
                # Create a dict to deduplicate links from the same source
                links = []
                unique_urls = set()
                
                for res in relevant_results[:5]:  # Use top 5 results
                    url = res["url"]
                    if url not in unique_urls:
                        unique_urls.add(url)
                        snippet = res["content"][:100] + "..." if len(res["content"]) > 100 else res["content"]
                        links.append({"url": url, "text": snippet})
                
                result["links"] = links
            
            # Log the final result structure (without full content for brevity)
            logger.info(f"Returning result: answer_length={len(result['answer'])}, num_links={len(result['links'])}")
            
            # Return the response in the exact format required
            return result
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": error_msg}
            )
        finally:
            conn.close()
    except Exception as e:
        # Catch any exceptions at the top level
        error_msg = f"Unhandled exception in query_knowledge_base: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )

@app.get("/health")
async def health_check():
    try:
        # Try to connect to the database as part of health check
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if tables exist and have data
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        
        # Check if any embeddings exist
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        discourse_embeddings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        markdown_embeddings = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "healthy", 
            "database": "connected", 
            "api_key_set": bool(API_KEY),
            "discourse_chunks": discourse_count,
            "markdown_chunks": markdown_count,
            "discourse_embeddings": discourse_embeddings,
            "markdown_embeddings": markdown_embeddings
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e), "api_key_set": bool(API_KEY)}
        )

# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 