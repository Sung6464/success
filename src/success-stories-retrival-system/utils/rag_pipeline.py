from typing import List, Dict, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import logging
from config import Config

logger = logging.getLogger(__name__)


class SourceReference(BaseModel):
    """Schema for individual source reference"""
    file_name: str = Field(description="Citation number and filename, e.g., '[1] filename.pdf'")
    downloadable_link: str = Field(description="SharePoint download URL")


class ChatbotResponse(BaseModel):
    """Schema for structured chatbot response"""
    response: str = Field(description="The main text response without References section")
    sources: List[SourceReference] = Field(default=[], description="List of cited sources with download links")


class SuccessStoryChatBot:
    
    def __init__(self, embedding_generator, vector_store):
        """Initialize chatbot with structured output support"""
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
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
        
        # Follow-up keywords for detecting continuation intent
        self.followup_keywords = [
            "more", "detail", "elaborate", "continue", "above", "same", "that",
            "explain", "tell me more", "what about", "how about", "give me",
            "from that", "from the", "from same", "same story", "this story"
        ]
        
        logger.info("✅ SuccessStoryChatBot initialized with structured output")
    
    def _is_followup_query(self, query: str, conversation_history: List[Dict]) -> bool:
        """
        Detect if query is a follow-up question requesting more details.
        
        Returns:
            True if query references previous content ("more details", "same story", etc.)
        """
        if not conversation_history:
            return False
        
        query_lower = query.lower()
        
        # Check for follow-up keywords
        for keyword in self.followup_keywords:
            if keyword in query_lower:
                logger.info(f"Follow-up detected (keyword: '{keyword}')")
                return True
        
        # Short queries with pronouns are likely follow-ups
        if len(query.split()) <= 5 and any(word in query_lower for word in ["it", "that", "this", "these"]):
            logger.info("Follow-up detected (short query with pronoun)")
            return True
        
        return False
    
    # def _build_system_prompt_structured(self, query: str, context: str, search_results: List[Dict]) -> str:
    #     """
    #     Build system prompt that instructs LLM to return structured JSON
    #     with sequential citation numbers and unique file mapping
    #     """
    #     # Build UNIQUE file mapping with sequential numbers
    #     file_mapping = {}
    #     seen_files = set()
    #     citation_num = 1
        
    #     for result in search_results:
    #         file_name = result.get('file_name', 'Unknown')
    #         download_url = result.get('source', '')
            
    #         # Only add if we haven't seen this file before
    #         if file_name and download_url and file_name not in seen_files:
    #             file_mapping[citation_num] = {
    #                 'file_name': file_name,
    #                 'url': download_url
    #             }
    #             seen_files.add(file_name)
    #             citation_num += 1
        
    #     # Convert mapping to string for prompt
    #     mapping_str = "\n".join([
    #         f"[{num}] {info['file_name']} -> {info['url']}"
    #         for num, info in file_mapping.items()
    #     ])
        
    #     logger.info(f"📚 Created {len(file_mapping)} unique file mappings with sequential citations")
    #     # print(f"\n📋 File Mappings:\n{mapping_str}\n")

    #     return f"""You are an AI assistant specialized in generating client success stories. Your task is to return a JSON object in the following exact structure:

    #     {{
    #         "response": "Provide a complete markdown-formatted answer with inline citations at the sentence level.",
    #         "sources": [{{"file_name": "[1] filename.pdf", "downloadable_link": "source_link"}}]
    #     }}

    #     ### Instructions:

    #     1. **Role & Objective:**
    #     - Deliver clear, accurate information about past client engagements and success stories based on the provided context.
    #     - Reference specific projects, technologies, and outcomes from the given context.
    #     - Format responses professionally using Markdown.
    #     - Cite sources immediately after each factual statement using inline citations like `[1]`.
    #     - If information is not available, politely state this and offer to search more broadly.

    #     2. **Response Guidelines:**
    #     - **For greetings (e.g., hi, hello):**
    #         - Start with `[GREETING]`.
    #         - Respond warmly and explain what you can help with.
    #         - Do NOT reference any success stories in greetings.
    #         - Set sources to empty array: []
    #         - Example: `[GREETING] Hello! Welcome, I can help you with client success stories and related details.`

    #     - **For specific success story questions:**
    #         - Begin with a brief summary.
    #         - Provide detailed points using bullet lists for accomplishments, technologies, and outcomes.
    #         - Include metrics and outcomes where available.
    #         - End with a takeaway (do NOT cite takeaways).
    #         - **Critical:** Every claim, metric, or outcome must have its own inline citation immediately after the statement.

    #     3. **Citation Rules:**
    #     - **IMPORTANT:** Use ONLY citation numbers [1], [2], [3], etc. that are listed in the "Available Source Files" section below
    #     - Cite at the sentence level using `[n]`
    #     - Repeat citation numbers for repeated references to the same file
    #     - **Example:**
    #         - ✅ CORRECT: `The project replaced AKKA with Spring Web Flux [2]. This eliminated costly license fees [2].`
    #         - ❌ INCORRECT: `The project replaced AKKA with Spring Web Flux, eliminated costly license fees [2].` (at end of paragraph)
        
    #     - **For bullet points:**
    #         ```
    #         - Replaced AKKA with Spring Web Flux [2]
    #         - Eliminated costly license fees [2]
    #         - Upgraded to Spring Boot 3 and Java 17 [2]
    #         ```
        
    #     - **For metrics:**
    #         ```
    #         - 80% of wealth advisors migrated [2]
    #         - Cost savings of $2M annually [3]
    #         ```

    #     4. **Available Source Files for Citation:**
    #     **CRITICAL:** Use ONLY these citation numbers in your response. Each number maps to a unique file.
        
    #     {mapping_str}

    #     5. **Sources Array Rules:**
    #     - Include ONLY the files you actually cited in your response
    #     - Use the EXACT citation numbers from the "Available Source Files" list above
    #     - Each file should appear only ONCE in the sources array
    #     - Format: {{"file_name": "[n] filename", "downloadable_link": "url"}}
        
    #     **Example:**
    #     If you cited [1] and [3] in your response, your sources array should be:
    #     ```
    #     "sources": [
    #         {{"file_name": "[1] File-Name-One.pdf", "downloadable_link": "https://sharepoint.com/..."}},
    #         {{"file_name": "[3] File-Name-Three.pdf", "downloadable_link": "https://sharepoint.com/..."}}
    #     ]
    #     ```

    #     6. **Tone & Style:**
    #     - Professional yet conversational
    #     - Concise and well-structured
    #     - Use Markdown headers, bullet points, and bold text
    #     - Always be transparent about citations

    #     ### Example Output Format:

    #     **Question:** Tell me about wealth management modernization projects.

    #     **Expected JSON Structure:**
    #     ```json
    #     {{
    #         "response": "## Wealth Management Platform Modernization\\n\\n**Summary:**\\nThe monolithic legacy framework hindered time to market [1]. Security vulnerabilities were a major concern [1].\\n\\n**Solution:**\\n- Replaced AKKA with Spring Web Flux [1]\\n- Eliminated costly license fees [1]\\n- Upgraded to Spring Boot 3 and Java 17 [1]\\n\\n**Outcomes:**\\n- Successfully migrated 80% of wealth advisors [1]\\n\\n**Takeaway:**\\nModernizing legacy platforms improves scalability and cost efficiency.",
    #         "sources": [
    #             {{"file_name": "[1] Wealth_Management_Modernization.pdf", "downloadable_link": "https://sharepoint.com/sites/client-success/Wealth_Management_Modernization.pdf"}}
    #         ]
    #     }}
    #     ```

    #     **REMEMBER:**
    #     - Citation numbers must be sequential: [1], [2], [3], etc.
    #     - Each file gets ONE unique citation number
    #     - Only cite files from the "Available Source Files" list
    #     - Include only cited files in the sources array
    #     - For greetings, return empty sources array []

    #     **Context:**
    #     {context}

    #     **Question:** {query}

    #     Now provide your JSON response following the structure exactly."""

    def _build_system_prompt_structured_new(self, query: str, search_results: List[Dict]) -> str:
        """
        Build system prompt with UNIQUE file citations (one number per file)
        
        Key Logic:
        1. Each unique document gets ONE citation number [1], [2], [3]...
        2. Multiple chunks from same document reuse the SAME citation number
        3. Context clearly shows which citation maps to which file
        """
        # Build UNIQUE file mapping - ONE citation number per unique file
        file_mapping = {}  # {citation_num: {file_name, url}}
        file_to_citation = {}  # {file_name: citation_num} - for reverse lookup
        citation_num = 1
        
        for result in search_results:
            file_name = result.get('file_name', 'Unknown')
            download_url = result.get('source', '')
            
            # Skip if we've already assigned a citation to this file
            if file_name in file_to_citation:
                logger.debug(f"📄 Reusing citation [{file_to_citation[file_name]}] for {file_name}")
                continue
            
            # Assign new citation number to this unique file
            if file_name and download_url:
                file_mapping[citation_num] = {
                    'file_name': file_name,
                    'url': download_url
                }
                file_to_citation[file_name] = citation_num
                logger.debug(f"📄 Assigned citation [{citation_num}] to {file_name}")
                citation_num += 1
        
        # Convert mapping to string for prompt
        mapping_str = "\n".join([
            f"[{num}] {info['file_name']} -> {info['url']}"
            for num, info in file_mapping.items()
        ])
        
        # Build context with proper citation numbers (multiple chunks can have same citation)
        context_with_citations = self._format_context_with_citations(search_results, file_to_citation)
        
        logger.info(f"Created {len(file_mapping)} unique file citations from {len(search_results)} search results")

        return f"""You are an AI assistant specialized in generating client success stories with conversation memory.
        
        **YOUR RESPONSE STRUCTURE:**
        {{
            "response": "Markdown-formatted answer with inline citations [1], [2], etc.",
            "sources": [{{"file_name": "[1] filename.pdf", "downloadable_link": "url"}}]
        }}

        ### Instructions:

        1. **Role & Objective:**
        - Deliver clear, accurate information about past client engagements and success stories.
        - **MAINTAIN CONVERSATION CONTEXT**: Reference previous messages when relevant (e.g., "As mentioned earlier...", "Building on that...")
        - **CRITICAL**: Use ONLY the citation numbers from "Available Source Files for Citation" section below
        - **STORY COHERENCE**: When telling a SINGLE story, prefer citing from ONE document. Only use multiple documents if discussing different stories.
        - **CRITICAL**: If you reference information from previous conversation that is NOT in current search results, do NOT add citations
        - Format responses professionally using Markdown
        - Cite sources immediately after factual statements ONLY when the information comes from current search results

        2. **Response Guidelines:**
        - **For greetings (e.g., hi, hello):**
            - Start with `[GREETING]`.
            - Respond warmly and explain what you can help with.
            - Do NOT reference any success stories in greetings.
            - Set sources to empty array: []
            - Example: `[GREETING] Hello! Welcome, I can help you with client success stories and related details.`

        - **For specific success story questions:**
            - Each numbered story (1., 2., 3.) should come from a DIFFERENT document
            - **CRITICAL**: If multiple chunks are from the same document, they are ONE story, not separate stories
            - Use numbered lists (1., 2., 3.) ONLY when discussing multiple DIFFERENT projects/clients
            - For a SINGLE story with multiple aspects, use:
              - Section headers (Challenge, Solution, Outcome)
              - Bullet points under each section
            - Include metrics and outcomes where available
            - **Critical:** Every claim, metric, or outcome must have its own inline citation immediately after the statement.

        3. **Citation Rules (CRITICAL FOR FOLLOW-UP QUESTIONS):**
        
        **RULE 1: ONLY cite documents from "Available Source Files for Citation" below**
        - **NEVER** cite documents from previous conversation that aren't in current search results
        - If referencing previous information NOT in current results, use phrases like:
          - "As I mentioned earlier (without new information to cite)"
          - "From our previous discussion"
          - "Building on what we discussed"
        
        **RULE 2: How to cite current search results:**
        - Use ONLY citation numbers [1], [2], [3], etc. listed in "Available Source Files" below
        - Cite at the sentence level using `[n]`
        - Repeat citation numbers for repeated references to the same file
        
        **Examples:**
        
        ✅ CORRECT (Follow-up with new sources):
        ```
        "Building on the banking project we discussed, here are additional details:
        - Cost optimization saved $2M annually [1]
        - Processing time reduced by 40% [1]"
        
        Sources: [1] Banking_Details.pdf (from CURRENT search)
        ```
        
        ✅ CORRECT (Follow-up WITHOUT matching sources):
        ```
        "I mentioned banking projects earlier, but I don't have additional 
        documents in the current search results to provide more details.
        Would you like me to search for specific aspects?"
        
        Sources: [] (empty - no sources from current search)
        ```
        
        ❌ INCORRECT (Citing documents not in current search):
        ```
        "The banking project I mentioned earlier [1] had these outcomes..."
        Sources: [1] Banking_Story.pdf (from PREVIOUS search, not current!)
        ```
        
        **RULE 3: Sentence-level citations:**
        - ✅ CORRECT: `The project replaced AKKA [2]. This eliminated costly fees [2].`
        - ❌ INCORRECT: `The project replaced AKKA, eliminated costly fees [2].` (cite at end)

        4. **Available Source Files for Citation:**
        **CRITICAL:** Use ONLY these citation numbers in your response. Each number maps to a unique file.
        
        {mapping_str}

        5. **Sources Array Rules:**
        - Include ONLY the files you actually cited in your response
        - Use the EXACT citation numbers from the "Available Source Files" list above
        - Each file should appear only ONCE in the sources array
        - Format: {{"file_name": "[n] filename", "downloadable_link": "url"}}

        6. **Tone & Style:**
        - Professional yet conversational
        - Concise and well-structured
        - Use Markdown headers, bullet points, and bold text
        - Always be transparent about citations

        ### Example Output Formats:

        **Example 1 - SINGLE Story (all from one document):**
        Question: "Tell me about the airline PSS migration"
        Available Sources: [1] Airline_PSS_Migration.pdf
        
        ```json
        {{
            "response": "## Airline PSS Migration Project\\n\\n**Challenge:**\\nThe airline needed to migrate its Passenger Service System to Amadeus Altea [1].\\n\\n**Solution:**\\n- Delivered end-to-end integration across 20+ domains [1]\\n- Covered reservations, ticketing, and frequent flyer [1]\\n\\n**Outcomes:**\\n- Achieved 100% on-time delivery [1]\\n- Migrated 67 airports successfully [1]\\n- Transferred 1.5M+ PNRs [1]",
            "sources": [
                {{"file_name": "[1] Airline_PSS_Migration.pdf", "downloadable_link": "https://..."}}
            ]
        }}
        ```
        **Key: ONE story = ONE document = ONE citation [1] used throughout**
        
        **Example 2 - MULTIPLE Different Stories (from different documents):**
        Question: "Show me airline modernization projects"
        Available Sources: [1] Airline_PSS_Migration.pdf, [2] Airline_Mobile_App.pdf, [3] Airline_Cloud_Migration.pdf
        
        ```json
        {{
            "response": "## Airline Modernization Success Stories\\n\\n### 1. PSS Migration Project\\n**Challenge:** Migrating legacy PSS to Amadeus Altea [1]\\n**Outcome:** Successfully migrated 67 airports [1]\\n\\n### 2. Customer Mobile Application\\n**Challenge:** Building customer-facing mobile app [2]\\n**Outcome:** Achieved 4.8-star rating with 1M+ downloads [2]\\n\\n### 3. Cloud Infrastructure Migration\\n**Challenge:** Migrating on-premise systems to Azure [3]\\n**Outcome:** Reduced infrastructure costs by 40% [3]",
            "sources": [
                {{"file_name": "[1] Airline_PSS_Migration.pdf", "downloadable_link": "https://..."}},
                {{"file_name": "[2] Airline_Mobile_App.pdf", "downloadable_link": "https://..."}},
                {{"file_name": "[3] Airline_Cloud_Migration.pdf", "downloadable_link": "https://..."}}
            ]
        }}
        ```
        **Key: THREE stories = THREE documents = THREE different citations [1], [2], [3]**
        **NEVER use [1] for story 1 AND story 3 - that's impossible!**
        
        **Example 2 - Follow-up Question (Same document in current search):**
        Previous: "Tell me about wealth management..."
        Current: "What were the cost savings?"
        Current Search Results: Wealth_Management_Modernization.pdf found
        
        ```json
        {{
            "response": "Regarding the wealth management project, here are the cost details:\\n\\n**Cost Savings:**\\n- Eliminated AKKA licensing fees: $500K annually [1]\\n- Reduced infrastructure costs by 40% [1]\\n- Saved $2M over 3 years [1]",
            "sources": [
                {{"file_name": "[1] Wealth_Management_Modernization.pdf", "downloadable_link": "https://sharepoint.com/..."}}
            ]
        }}
        ```
        
        **Example 3 - Follow-up Question (Document NOT in current search):**
        Previous: "Tell me about wealth management..."
        Current: "What were the cost savings?"
        Current Search Results: Different documents found, NOT the wealth management one
        
        ```json
        {{
            "response": "I discussed a wealth management project earlier, but the current search results don't include that specific document. The results show:\\n\\n**Related Cost Optimization Projects:**\\n- Banking platform reduced costs by 30% [1]\\n- Insurance modernization saved $1.5M [2]\\n\\nWould you like me to search specifically for wealth management cost details?",
            "sources": [
                {{"file_name": "[1] Banking_Platform.pdf", "downloadable_link": "https://sharepoint.com/..."}},
                {{"file_name": "[2] Insurance_Modern.pdf", "downloadable_link": "https://sharepoint.com/..."}}
            ]
        }}
        ```

        **REMEMBER:**
        - Citation numbers are SEQUENTIAL: [1], [2], [3]
        - **ONE DOCUMENT = ONE STORY**: Each numbered story (1., 2., 3.) must come from a DIFFERENT document
        - If you see multiple chunks with same file name, they are ONE story (not separate stories)
        - **NEVER** cite the same document for two different numbered stories
        - **CRITICAL**: Only cite files from "Available Source Files" list above
        - **NEVER** cite documents from conversation history that aren't in current search
        - Include only cited files (from current search) in the sources array
        - For greetings, return empty sources array []
        - If no current sources match the question, return empty sources array []

        **Current Search Results (CITE ONLY FROM HERE):**
        {context_with_citations}

        **Current Question:** {query}

        Now provide your JSON response. Remember: sources array should contain ONLY documents from the "Available Source Files for Citation" section above that you actually cited in your response."""  
    
    def _format_context_with_citations(self, results: List[Dict], file_to_citation: Dict[str, int]) -> str:
        """
        Format search results with REUSABLE citation numbers
        
        Key: Multiple chunks from the same document will show the SAME citation number,
        making it clear that [1] refers to ONE document, not one chunk.
        
        Args:
            results: Search results from vector store
            file_to_citation: Mapping of file names to their UNIQUE citation numbers
            
        Returns:
            Formatted context string where same files share citation numbers
        """
        if not results:
            return "No relevant success stories found."
        
        context_parts = []
        
        for idx, result in enumerate(results, 1):
            file_name = result.get('file_name', 'Unknown')
            citation_num = file_to_citation.get(file_name, '?')
            
            # Check if this is from previous cache
            from_previous = result.get('from_previous', False)
            cache_indicator = " [CACHED]" if from_previous else ""
            
            # Show both chunk number and citation number for clarity
            context_parts.append(f"\n[Chunk {idx} - Use Citation [{citation_num}]]{cache_indicator} From: {file_name}")
            context_parts.append(f"Category: {result.get('category', 'N/A')}")
            context_parts.append(f"Relevance: {result.get('similarity', 0):.2%}")
            
            # Handle missing chunk_text (for cached sources)
            chunk_text = result.get('chunk_text', result.get('content', 'Content not available'))
            context_parts.append(f"\nContent:\n{chunk_text}\n")
            context_parts.append("─" * 80)
        
        return "\n".join(context_parts)

    def generate_response_structured(
        self, 
        query: str, 
        search_results: List[Dict],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, any]:
        """
        Generate structured response using OpenAI function calling with conversation context
        
        Args:
            query: User question
            search_results: Search results from vector store
            conversation_history: Previous conversation messages [{"role": str, "content": str}]
            
        Returns:
            Dictionary with exact structure:
            {
                "response": str,
                "sources": [{"file_name": str, "downloadable_link": str}]
            }
        """
        try:
            system_prompt = self._build_system_prompt_structured_new(query, search_results)
        
            # Build message list with conversation history
            messages = [SystemMessage(content=system_prompt)]
            
            # Add conversation history if available
            if conversation_history:
                logger.info(f"📜 Including {len(conversation_history)} previous messages for context")
                for msg in conversation_history:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:  # assistant
                        messages.append(SystemMessage(content=f"Previous Assistant Response: {msg['content']}"))
            
            # Add current query
            messages.append(HumanMessage(content=f"Question: {query}"))
            
            # Get structured response from LLM
            logger.info("💬 Generating structured response...")
            structured_response: ChatbotResponse = self.structured_llm.invoke(messages)
            

            # Format context
            # context = self._format_context_new(search_results)
            
            # Build structured prompt
            # system_prompt = self._build_system_prompt_structured(query, context, search_results)
            
            # messages = [
            # SystemMessage(content=system_prompt),
            # HumanMessage(content=f"Question: {query}\n\nContext: {context}")
            # ]
            
            # Get structured response from LLM
            logger.info("💬 Generating structured response...")
            # structured_response: ChatbotResponse = self.structured_llm.invoke(messages)
            
            # print()
            # Convert Pydantic model to dict
            response_dict = {
                "response": structured_response.response,
                "sources": [
                    {
                        "file_name": source.file_name,
                        "download_url": source.downloadable_link
                    }
                    for source in structured_response.sources
                ]
            }
            
            logger.info(f"✅ Structured response generated with {len(response_dict['sources'])} sources")
            
            return response_dict
            
        except Exception as e:
            logger.error(f"❌ Error generating structured response: {str(e)}", exc_info=True)
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
               category_filter: Optional[str] = None,
               previous_sources: Optional[List[Dict]] = None,
               is_followup: bool = False) -> List[Dict]:
        """
        Search for stories matching the query.
        
        Args:
            query: User query/question
            top_k: Number of stories to return
            similarity_threshold: Minimum similarity score (0-1)
            category_filter: Optional category to filter by (e.g., 'insurance', 'banking')
            previous_sources: Sources from last assistant response (for follow-ups)
            is_followup: Whether this is a follow-up query
            
        Returns:
            List of matching stories with download links
        """
        self.logger.info("Searching...")       
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)
        
        # Search in vector store with optional category filter
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 3,  # Get more results for grouping
            category_filter=category_filter,
            similarity_threshold=similarity_threshold
        )
        
        # For follow-up queries, merge with previous sources
        if is_followup and previous_sources:
            logger.info(f"🔗 Follow-up query: Merging {len(previous_sources)} previous sources with {len(results)} new results")
            
            # Get file names from new results
            new_file_names = {r.get('file_name') for r in results}
            
            # Add previous sources that aren't in new results (with lower similarity)
            for prev_source in previous_sources:
                if prev_source.get('file_name') not in new_file_names:
                    # Mark as from previous context
                    prev_source['similarity'] = prev_source.get('similarity', 0.85)  # Keep original or default
                    prev_source['from_previous'] = True
                    results.append(prev_source)
                    logger.info(f"📌 Included previous source: {prev_source.get('file_name')}")
        
        # STORY COHERENCE: Group chunks by document and prioritize complete stories
        results = self._ensure_story_coherence(results, top_k)
        
        return results
    
    def _ensure_story_coherence(self, results: List[Dict], top_k: int) -> List[Dict]:
        """
        Ensure story coherence: ONE document = ONE story.
        
        Strategy:
        1. Group chunks by document (all chunks from same doc = same story)
        2. Score each document group
        3. Select top 2-3 DIFFERENT documents (each becomes a separate story)
        4. Mark chunks so LLM knows they're from different stories
        
        This ensures:
        - Story 1 → Document A → Citation [1]
        - Story 2 → Document B → Citation [2]
        - Story 3 → Document C → Citation [3]
        
        NOT: Story 1 and 3 from same Document A (impossible!)
        """
        if not results:
            return []
        
        # Group by document
        from collections import defaultdict
        doc_groups = defaultdict(list)
        
        for result in results:
            file_name = result.get('file_name', 'Unknown')
            doc_groups[file_name].append(result)
        
        # Score each document group
        doc_scores = []
        for file_name, chunks in doc_groups.items():
            avg_similarity = sum(c.get('similarity', 0) for c in chunks) / len(chunks)
            # Bonus for having multiple chunks (more complete story)
            completeness_bonus = min(len(chunks) * 0.05, 0.15)  # Up to 15% bonus
            score = avg_similarity + completeness_bonus
            
            # Find highest similarity chunk (for sorting)
            max_similarity = max(c.get('similarity', 0) for c in chunks)
            
            doc_scores.append({
                'file_name': file_name,
                'chunks': chunks,
                'score': score,
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'chunk_count': len(chunks)
            })
        
        # Sort by score (highest first)
        doc_scores.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"📚 Story Coherence: Grouped {len(results)} chunks into {len(doc_scores)} unique stories/documents")
        for i, doc in enumerate(doc_scores[:3], 1):
            logger.info(f"  Story {i}: {doc['file_name']} - {doc['chunk_count']} chunks, avg: {doc['avg_similarity']:.2%}, score: {doc['score']:.2%}")
        
        # Select top 2-3 DIFFERENT documents (each is a separate story)
        final_results = []
        max_stories = 3  # Limit to 3 different stories to avoid overwhelming user
        
        for story_num, doc_info in enumerate(doc_scores[:max_stories], 1):
            # Mark all chunks from this document as belonging to same story
            for chunk in doc_info['chunks']:
                chunk['story_number'] = story_num  # For debugging
                chunk['story_document'] = doc_info['file_name']
                final_results.append(chunk)
            
            logger.info(f"✅ Story {story_num}: {doc_info['file_name']} ({doc_info['chunk_count']} chunks, score: {doc_info['score']:.2%})")
        
        # Sort chunks within results by story number, then similarity
        final_results.sort(key=lambda x: (x.get('story_number', 999), -x.get('similarity', 0)))
        
        logger.info(f"📖 Final: {len(final_results)} chunks from {min(len(doc_scores), max_stories)} different stories")
        
        return final_results