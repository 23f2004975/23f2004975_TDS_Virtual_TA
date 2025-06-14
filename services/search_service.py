from dotenv import load_dotenv
from logging_config import logger
from fastapi import HTTPException
import os
import aiohttp
import asyncio
import traceback
import json
import numpy as np


MAX_RESULTS = 10 
MAX_CONTEXT_CHUNKS = 4 

API_KEY = os.getenv("API_KEY") 
load_dotenv()

SIMILARITY_THRESHOLD = 0.50

def cosine_similarity(vec1, vec2):
    try:
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Handle zero vectors
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
            
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
        logger.error(f"Error in cosine_similarity: {e}")
        logger.error(traceback.format_exc())
        return 0.0  # Return 0 similarity on error rather than crashing

async def find_similar_content(query_embedding, conn):
    try:
        logger.info("Finding similar content in database")
        cursor = conn.cursor()
        results = []
        
        # Search discourse chunks
        logger.info("Querying discourse chunks")
        cursor.execute("""
        SELECT id, post_id, topic_id, topic_title, post_number, author, created_at, 
               likes, chunk_index, content, url, embedding 
        FROM discourse_chunks 
        WHERE embedding IS NOT NULL
        """)
        
        discourse_chunks = cursor.fetchall()
        logger.info(f"Processing {len(discourse_chunks)} discourse chunks")
        processed_count = 0
        
        for chunk in discourse_chunks:
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    # Ensure URL is properly formatted
                    url = chunk["url"]
                    if not url.startswith("http"):
                        # Fix missing protocol
                        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{url}"
                    
                    results.append({
                        "source": "discourse",
                        "id": chunk["id"],
                        "post_id": chunk["post_id"],
                        "topic_id": chunk["topic_id"],
                        "title": chunk["topic_title"],
                        "url": url,
                        "content": chunk["content"],
                        "author": chunk["author"],
                        "created_at": chunk["created_at"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count}/{len(discourse_chunks)} discourse chunks")
                    
            except Exception as e:
                logger.error(f"Error processing discourse chunk {chunk['id']}: {e}")
        
        # Search markdown chunks
        logger.info("Querying markdown chunks")
        cursor.execute("""
        SELECT id, doc_title, original_url, downloaded_at, chunk_index, content, embedding 
        FROM markdown_chunks 
        WHERE embedding IS NOT NULL
        """)
        
        markdown_chunks = cursor.fetchall()
        logger.info(f"Processing {len(markdown_chunks)} markdown chunks")
        processed_count = 0
        
        for chunk in markdown_chunks:
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    # Ensure URL is properly formatted
                    url = chunk["original_url"]
                    if not url or not url.startswith("http"):
                        # Use a default URL if missing
                        url = f"https://docs.onlinedegree.iitm.ac.in/{chunk['doc_title']}"
                    
                    results.append({
                        "source": "markdown",
                        "id": chunk["id"],
                        "title": chunk["doc_title"],
                        "url": url,
                        "content": chunk["content"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count}/{len(markdown_chunks)} markdown chunks")
                    
            except Exception as e:
                logger.error(f"Error processing markdown chunk {chunk['id']}: {e}")
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Found {len(results)} relevant results above threshold")
        
        # Group by source document and keep most relevant chunks
        grouped_results = {}
        
        for result in results:
            # Create a unique key for the document/post
            if result["source"] == "discourse":
                key = f"discourse_{result['post_id']}"
            else:
                key = f"markdown_{result['title']}"
            
            if key not in grouped_results:
                grouped_results[key] = []
            
            grouped_results[key].append(result)
        
        # For each source, keep only the most relevant chunks
        final_results = []
        for key, chunks in grouped_results.items():
            # Sort chunks by similarity
            chunks.sort(key=lambda x: x["similarity"], reverse=True)
            # Keep top chunks
            final_results.extend(chunks[:MAX_CONTEXT_CHUNKS])
        
        # Sort again by similarity
        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top results, limited by MAX_RESULTS
        logger.info(f"Returning {len(final_results[:MAX_RESULTS])} final results after grouping")
        return final_results[:MAX_RESULTS]
    except Exception as e:
        error_msg = f"Error in find_similar_content: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

async def enrich_with_adjacent_chunks(conn, results):
    try:
        logger.info(f"Enriching {len(results)} results with adjacent chunks")
        cursor = conn.cursor()
        enriched_results = []
        
        for result in results:
            enriched_result = result.copy()
            additional_content = ""
            
            # Try to get adjacent chunks for context
            if result["source"] == "discourse":
                post_id = result["post_id"]
                current_chunk_index = result["chunk_index"]
                
                # Try to get previous chunk
                if current_chunk_index > 0:
                    cursor.execute("""
                    SELECT content FROM discourse_chunks 
                    WHERE post_id = ? AND chunk_index = ?
                    """, (post_id, current_chunk_index - 1))
                    prev_chunk = cursor.fetchone()
                    if prev_chunk:
                        additional_content = prev_chunk["content"] + " "
                
                # Try to get next chunk
                cursor.execute("""
                SELECT content FROM discourse_chunks 
                WHERE post_id = ? AND chunk_index = ?
                """, (post_id, current_chunk_index + 1))
                next_chunk = cursor.fetchone()
                if next_chunk:
                    additional_content += " " + next_chunk["content"]
                
            elif result["source"] == "markdown":
                title = result["title"]
                current_chunk_index = result["chunk_index"]
                
                # Try to get previous chunk
                if current_chunk_index > 0:
                    cursor.execute("""
                    SELECT content FROM markdown_chunks 
                    WHERE doc_title = ? AND chunk_index = ?
                    """, (title, current_chunk_index - 1))
                    prev_chunk = cursor.fetchone()
                    if prev_chunk:
                        additional_content = prev_chunk["content"] + " "
                
                # Try to get next chunk
                cursor.execute("""
                SELECT content FROM markdown_chunks 
                WHERE doc_title = ? AND chunk_index = ?
                """, (title, current_chunk_index + 1))
                next_chunk = cursor.fetchone()
                if next_chunk:
                    additional_content += " " + next_chunk["content"]
            
            # Add the enriched content
            if additional_content:
                enriched_result["content"] = f"{result['content']} {additional_content}"
            
            enriched_results.append(enriched_result)
        
        logger.info(f"Successfully enriched {len(enriched_results)} results")
        return enriched_results
    except Exception as e:
        error_msg = f"Error in enrich_with_adjacent_chunks: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

