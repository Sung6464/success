from typing import List, Dict, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import logging
from config import Config
import re

logger = logging.getLogger(__name__)


class SourceReference(BaseModel):
    """Schema for individual source reference"""
    file_name: str = Field(description="Citation number and filename, e.g., '[1] filename.pdf'")
    downloadable_link: str = Field(description="SharePoint download URL")


class ChatbotResponse(BaseModel):
    """Schema for structured chatbot response"""
    response: str = Field(description="The main text response without References section")
    sources: List[SourceReference] = Field(default=[], description="List of cited sources with download links")


class Index_Success_Stories:
    
    def __init__(self, sharepoint_manager, embedding_generator, vector_store):
        """Initialize chatbot with structured output support"""
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.sharepoint_manager = sharepoint_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM with structured output
        self.llm = AzureChatOpenAI(
            api_key=Config.AZURE_OPENAI_LLM_API_KEY,
            api_version=Config.AZURE_OPENAI_LLM_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_LLM_API_BASE,
            azure_deployment=Config.AZURE_OPENAI_LLM_MODEL,
            temperature=0.7,
            max_tokens=2000,
        )
        
        # Create structured output LLM
        self.structured_llm = self.llm.with_structured_output(ChatbotResponse)
        
        logger.info("✅ SuccessStoryChatBot initialized with structured output")
    
    def _build_system_prompt_structured(self, query: str, context: str, search_results: List[Dict]) -> str:
        """
        Build system prompt that instructs LLM to return structured JSON
        with sequential citation numbers and unique file mapping
        """
        # Build UNIQUE file mapping with sequential numbers
        file_mapping = {}
        seen_files = set()
        citation_num = 1
        
        for result in search_results:
            file_name = result.get('file_name', 'Unknown')
            download_url = result.get('source', '')
            
            # Only add if we haven't seen this file before
            if file_name and download_url and file_name not in seen_files:
                file_mapping[citation_num] = {
                    'file_name': file_name,
                    'url': download_url
                }
                seen_files.add(file_name)
                citation_num += 1
        
        # Convert mapping to string for prompt
        mapping_str = "\n".join([
            f"[{num}] {info['file_name']} -> {info['url']}"
            for num, info in file_mapping.items()
        ])
        
        logger.info(f"📚 Created {len(file_mapping)} unique file mappings with sequential citations")
        # print(f"\n📋 File Mappings:\n{mapping_str}\n")

        return f"""You are an AI assistant specialized in generating client success stories. Your task is to return a JSON object in the following exact structure:

    {{
        "response": "Provide a complete markdown-formatted answer with inline citations at the sentence level.",
        "sources": [{{"file_name": "[1] filename.pdf", "downloadable_link": "source_link"}}]
    }}

    ### Instructions:

    1. **Role & Objective:**
    - Deliver clear, accurate information about past client engagements and success stories based on the provided context.
    - Reference specific projects, technologies, and outcomes from the given context.
    - Format responses professionally using Markdown.
    - Cite sources immediately after each factual statement using inline citations like `[1]`.
    - If information is not available, politely state this and offer to search more broadly.

    2. **Response Guidelines:**
    - **For greetings (e.g., hi, hello):**
        - Start with `[GREETING]`.
        - Respond warmly and explain what you can help with.
        - Do NOT reference any success stories in greetings.
        - Set sources to empty array: []
        - Example: `[GREETING] Hello! Welcome, I can help you with client success stories and related details.`

    - **For specific success story questions:**
        - Begin with a brief summary.
        - Provide detailed points using bullet lists for accomplishments, technologies, and outcomes.
        - Include metrics and outcomes where available.
        - End with a takeaway (do NOT cite takeaways).
        - **Critical:** Every claim, metric, or outcome must have its own inline citation immediately after the statement.

    3. **Citation Rules:**
    - **IMPORTANT:** Use ONLY citation numbers [1], [2], [3], etc. that are listed in the "Available Source Files" section below
    - Cite at the sentence level using `[n]`
    - Repeat citation numbers for repeated references to the same file
    - **Example:**
        - ✅ CORRECT: `The project replaced AKKA with Spring Web Flux [2]. This eliminated costly license fees [2].`
        -   INCORRECT: `The project replaced AKKA with Spring Web Flux, eliminated costly license fees [2].` (at end of paragraph)
    
    - **For bullet points:**
        ```
        - Replaced AKKA with Spring Web Flux [2]
        - Eliminated costly license fees [2]
        - Upgraded to Spring Boot 3 and Java 17 [2]
        ```
    
    - **For metrics:**
        ```
        - 80% of wealth advisors migrated [2]
        - Cost savings of $2M annually [3]
        ```

    4. **Available Source Files for Citation:**
    **CRITICAL:** Use ONLY these citation numbers in your response. Each number maps to a unique file.
    
    {mapping_str}

    5. **Sources Array Rules:**
    - Include ONLY the files you actually cited in your response
    - Use the EXACT citation numbers from the "Available Source Files" list above
    - Each file should appear only ONCE in the sources array
    - Format: {{"file_name": "[n] filename", "downloadable_link": "url"}}
    
    **Example:**
    If you cited [1] and [3] in your response, your sources array should be:
    ```
    "sources": [
        {{"file_name": "[1] File-Name-One.pdf", "downloadable_link": "https://sharepoint.com/..."}},
        {{"file_name": "[3] File-Name-Three.pdf", "downloadable_link": "https://sharepoint.com/..."}}
    ]
    ```

    6. **Tone & Style:**
    - Professional yet conversational
    - Concise and well-structured
    - Use Markdown headers, bullet points, and bold text
    - Always be transparent about citations

    ### Example Output Format:

    **Question:** Tell me about wealth management modernization projects.

    **Expected JSON Structure:**
    ```json
    {{
        "response": "## Wealth Management Platform Modernization\\n\\n**Summary:**\\nThe monolithic legacy framework hindered time to market [1]. Security vulnerabilities were a major concern [1].\\n\\n**Solution:**\\n- Replaced AKKA with Spring Web Flux [1]\\n- Eliminated costly license fees [1]\\n- Upgraded to Spring Boot 3 and Java 17 [1]\\n\\n**Outcomes:**\\n- Successfully migrated 80% of wealth advisors [1]\\n\\n**Takeaway:**\\nModernizing legacy platforms improves scalability and cost efficiency.",
        "sources": [
            {{"file_name": "[1] Wealth_Management_Modernization.pdf", "downloadable_link": "https://sharepoint.com/sites/client-success/Wealth_Management_Modernization.pdf"}}
        ]
    }}
    ```

    **REMEMBER:**
    - Citation numbers must be sequential: [1], [2], [3], etc.
    - Each file gets ONE unique citation number
    - Only cite files from the "Available Source Files" list
    - Include only cited files in the sources array
    - For greetings, return empty sources array []

    **Context:**
    {context}

    **Question:** {query}

    Now provide your JSON response following the structure exactly."""
       
    def generate_response_structured(
        self, 
        query: str, 
        search_results: List[Dict]
    ) -> Dict[str, any]:
        """
        Generate structured response using OpenAI function calling
        
        Args:
            query: User question
            search_results: Search results from vector store
            
        Returns:
            Dictionary with exact structure:
            {
                "response": str,
                "sources": [{"file_name": str, "downloadable_link": str}]
            }
        """
        try:
            # Format context
            context = self._format_context_new(search_results)
            
            # Build structured prompt
            system_prompt = self._build_system_prompt_structured(query, context, search_results)
            
            messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {query}\n\nContext: {context}")
            ]
            
            # Get structured response from LLM
            logger.info("💬 Generating structured response...")
            structured_response: ChatbotResponse = self.structured_llm.invoke(messages)
            
            # print()
            # Convert Pydantic model to dict
            response_dict = {
                "response": structured_response.response,
                "sources": [
                    {
                        "file_name": source.file_name,
                        "downloadable_link": source.downloadable_link
                    }
                    for source in structured_response.sources
                ]
            }
            
            logger.info(f"✅ Structured response generated with {len(response_dict['sources'])} sources")
            
            return response_dict
            
        except Exception as e:
            logger.error(f"  Error generating structured response: {str(e)}", exc_info=True)
            return {
                "response": f"Sorry, an error occurred: {str(e)}",
                "sources": []
            }
    
    def _format_context_new(self, results: List[Dict]) -> str:
        """Format search results into context string"""
        if not results:
            return "No relevant success stories found."
        
        context_parts = []
        
        for idx, result in enumerate(results, 1):
            context_parts.append(f"\n[Source {idx}] {result['file_name']}")
            context_parts.append(f"Category: {result.get('category', 'N/A')}")
            context_parts.append(f"Relevance: {result['similarity']:.2%}")
            context_parts.append(f"\nContent:\n{result['chunk_text']}\n")
            context_parts.append("─" * 80)
        
        return "\n".join(context_parts)
    
    def search(self, 
               query: str, 
               top_k: int = 5, 
               similarity_threshold: float = 0.5,
               category_filter: Optional[str] = None) -> List[Dict]:
        """
        Search for stories matching the query.
        
        Args:
            query: User query/question
            top_k: Number of stories to return
            similarity_threshold: Minimum similarity score (0-1)
            category_filter: Optional category to filter by (e.g., 'insurance', 'banking')
            
        Returns:
            List of matching stories with download links
        """
        self.logger.info("Searching...")       
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)
        
        # Search in vector store with optional category filter
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,            
            category_filter=category_filter,
            similarity_threshold=similarity_threshold
        )
        
        return results
    
    def index_all_stories(self, folder_path: Optional[str] = None) -> Dict:
        """
        Index all stories from SharePoint.
        This is the main function to process and store all documents.
        Args:
            folder_path: Optional SharePoint folder path to filter files
        Returns:
            Summary statistics
        """
        logger.info("Starting SharePoint story indexing pipeline...")

        # Use SharePointManager to extract all documents
        documents = self.sharepoint_manager.extract_data(folder_path or "")
        logger.info(f"📁 Found {len(documents)} SharePoint files to process")

        stats = {
            'total_files': len(documents),
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'total_stories': 0,
            'failed_files_list': []
        }

        for doc in documents:
            file_name = doc.get('name', '')
            story_id = doc.get('id', '')
            try:
                logger.info(f"📄 Processing: {file_name}")

                # Step 1: Chunk and embed document (returns chunk_texts, embeddings, metadata_list)
                chunked = self.embedding_generator.chunk_and_embed_document(doc)
                chunk_texts = chunked['chunk_texts']
                embeddings = chunked['embeddings']
                metadata_list = chunked['metadata_list']

                if not chunk_texts or not embeddings:
                    logger.warning(f"⚠️ Skipping {file_name} - no chunks or embeddings generated")
                    stats['failed_files'] += 1
                    stats['failed_files_list'].append(file_name)
                    continue

                # Step 2: Prepare file-level metadata
                file_metadata = doc.get('metadata', {})
                category = file_metadata.get('category', 'general')
                source_file = file_name
                graph_url = file_metadata.get('graph_url', '')  # SharePoint download link
                file_type = file_metadata.get('file_type', '')

                # Step 3: Store in PostgreSQL
                self.vector_store.insert_story(
                    story_id=story_id,
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

                logger.info(f"✅ Indexed '{file_name}' with {len(chunk_texts)} chunks")

            except Exception as e:
                logger.error(f"  Failed to process {file_name}: {e}")
                stats['failed_files'] += 1
                stats['failed_files_list'].append(file_name)

        logger.info(f"\n🎉 Indexing complete!")
        logger.info(f"   Processed: {stats['processed_files']}/{stats['total_files']} files")
        logger.info(f"   Total stories: {stats['total_stories']}")
        logger.info(f"   Total chunks: {stats['total_chunks']}")
        logger.info(f"   Failed: {stats['failed_files']}")

        return stats
    
    def index_all_stories_new(self, folder_path: Optional[str] = None) -> Dict:
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
            files = self.sharepoint_manager._get_all_files_recursive(folder_path)
            logger.info(f"📁 Found {len(files)} SharePoint files to process")

            stats = {
                'total_files': len(files),
                'processed_files': 0,
                'failed_files': 0,
                'total_chunks': 0,
                'total_stories': 0,
                'failed_files_list': []
            }

            for doc in files:
                file_name = doc.get('name', '')
                file_id = doc.get('id', '')

                # Download content
                try:
                    content = self.sharepoint_manager.download_file(file_id)
                    if not content:
                        logger.warning(f"⚠️ Could not download {file_name}")
                        stats['failed_files'] += 1
                        stats['failed_files_list'].append(file_name)
                        continue
                except Exception as e:
                    logger.error(f"  Download failed for {file_name}: {e}")
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
                    logger.error(f"  Text extraction failed for {file_name}: {e}")
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
                        'source': doc.get('webUrl', ''),
                        'file_type': extension,
                        'size': doc.get('size', 0),
                        'created_at': doc.get('createdDateTime', ''),
                        'modified_at': doc.get('lastModifiedDateTime', ''),
                        'url': doc.get('webUrl', ''),
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
            logger.error(f"  Failed to process {file_name}: {e}")
            stats['failed_files'] += 1
            stats['failed_files_list'].append(file_name)

            logger.info(f"\n🎉 Indexing complete!")
            logger.info(f"   Processed: {stats['processed_files']}/{stats['total_files']} files")
            logger.info(f"   Total stories: {stats['total_stories']}")
            logger.info(f"   Total chunks: {stats['total_chunks']}")
            logger.info(f"   Failed: {stats['failed_files']}")

        return stats
