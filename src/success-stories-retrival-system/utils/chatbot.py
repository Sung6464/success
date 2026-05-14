"""
Success Stories ChatBot v1
Provides conversational AI responses with RAG
"""
from typing import List, Dict, Optional, Generator
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import logging
from datetime import datetime

from config import Config
from utils.rag_pipeline import RAGPipeline


class SuccessStoryChatBot:
    """
    Production-ready chatbot for Success Stories
    Provides conversational AI responses with RAG-based context
    """
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize the chatbot with RAG pipeline and Azure OpenAI
        
        Args:
            rag_pipeline: RAGPipeline instance for document retrieval
        """
        self.rag_pipeline = rag_pipeline
        self.logger = logging.getLogger(__name__)
        
        # Initialize Azure OpenAI Chat
        self.llm = AzureChatOpenAI(
            api_key=Config.AZURE_OPENAI_LLM_API_KEY,
            api_version=Config.AZURE_OPENAI_LLM_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_LLM_API_BASE,
            azure_deployment=Config.AZURE_OPENAI_LLM_MODEL,
            temperature=0.7,
            max_tokens=1500,
            streaming=False
        )
        
        self.conversation_history = []
        
        self.logger.info("✅ SuccessStoryChatBot initialized")

    # def _build_system_prompt(self, query: str, context: str) -> str:
    #     """Build the system prompt for the chatbot with strict grounding rules"""
        
    #     # Check if context is actually empty or placeholder
    #     has_valid_context = context and context.strip() and context != "No relevant success stories found."
    
    #     return f"""You are an AI assistant specialized in analyzing and presenting client success stories.

    #         **CRITICAL RULES - YOU MUST FOLLOW THESE STRICTLY:**
    #         1. **ONLY use information from the provided context below** - Do NOT generate, infer, or hallucinate any information from external knowledge
    #         2. **If context is empty or irrelevant** - Explicitly state that no relevant success stories were found
    #         3. **For greetings ONLY** - Respond with a welcoming message WITHOUT referencing any success stories
    #         4. **Never make up projects, clients, or outcomes** - If it's not in the context, don't mention it

    #         ---

    #         **YOUR RESPONSE GUIDELINES:**

    #         **SCENARIO 1 - Greeting Detection (hi, hello, hey, good morning, how are you, etc.):**
    #         - **Detect if the question is a casual greeting or general inquiry**
    #         - Start response with the tag: `[GREETING]`
    #         - Provide ONLY this generic welcome message:
            
    #         "[GREETING] Hello! 👋 I'm here to help you explore our client success stories. 
            
    #         I can assist you with information about:
    #         • Industry-specific projects (Banking, Healthcare, Retail, Manufacturing, etc.)
    #         • Technology implementations (Cloud, AI/ML, Data Analytics, IoT, etc.)
    #         • Digital transformation and modernization initiatives
    #         • Migration and integration case studies
            
    #         What would you like to know about?"
            
    #         - **DO NOT** reference any specific success stories, projects, or clients
    #         - **DO NOT** use the context provided above for greeting responses
    #         - **DO NOT** mention any technical details or outcomes

    #         **SCENARIO 2 - Specific Query WITH Available Context:**
    #         - **Only proceed if context contains actual success story information (not "[NO DATA AVAILABLE]")**
    #         - Provide a clear, structured response using **ONLY** the context above
    #         - Format your response as:
            
    #         **Summary:**
    #         [2-3 sentence overview based on context]
            
    #         **Key Details:**
    #         • [Detail 1 from context]
    #         • [Detail 2 from context]
    #         • [Detail 3 from context]
            
    #         **Technologies/Outcomes:**
    #         • [Technology/outcome 1 from context]
    #         • [Technology/outcome 2 from context]
            
    #         **Source:** [Story/Project name from context]
            
    #         - Include specific metrics or achievements **ONLY if explicitly stated in context**
    #         - Always cite which success story you're referencing
    #         - If context has limited information, acknowledge it: "Based on available information..."

    #         **SCENARIO 3 - Specific Query BUT No Relevant Context:**
    #         - **This applies when context is "[NO DATA AVAILABLE]" or empty**
    #         - Respond with:
            
    #         "I couldn't find any success stories matching '{query}' in our current database.
            
    #         **Suggestions to refine your search:**
    #         • Try broader terms (e.g., 'cloud' instead of 'Azure Kubernetes')
    #         • Search by industry (Banking, Healthcare, Retail, etc.)
    #         • Search by technology category (Cloud, AI, Analytics, etc.)
            
    #         **Example queries that might help:**
    #         • 'Show me cloud migration projects'
    #         • 'Tell me about banking sector implementations'
    #         • 'What AI/ML success stories do you have?'
    #         • 'Show me digital transformation case studies'
            
    #         Would you like to try a different search?"
            
    #         - **DO NOT** make up or generate any project information
    #         - **DO NOT** use generic industry knowledge or examples
    #         - **DO NOT** reference projects not in the context

    #         **FORMATTING RULES:**
    #         - Use markdown formatting (headers, bullet points, bold text)
    #         - Keep responses concise but informative
    #         - Use professional yet conversational tone
    #         - Break up text with appropriate spacing

    #         **VALIDATION CHECKLIST (Before responding, verify):**
    #         ☑️ Am I only using information from the context above?
    #         ☑️ If it's a greeting, am I avoiding specific success story references?
    #         ☑️ If there's no context, am I being honest about it?
    #         ☑️ Am I citing sources when referencing specific stories?

    #         **REMEMBER:** You are a search assistant for an existing database. You can ONLY work with the success stories provided in the context above. If information isn't there, acknowledge it honestly and guide the user to refine their search. Never invent or assume information.
    #         """
    
    # def _build_system_prompt(self, query: str, context: str) -> str:
    #     """Build the system prompt for the chatbot with strict grounding rules"""
    #     return f"""You are an AI assistant specialized in analyzing and presenting client success stories.

    #         **CRITICAL RULES - YOU MUST FOLLOW THESE STRICTLY:**
    #         1. **ONLY use information from the provided context below** - Do NOT generate, infer, or hallucinate any information from external knowledge
    #         2. **If context is empty or irrelevant** - Explicitly state that no relevant success stories were found
    #         3. **For greetings ONLY** - Respond with a welcoming message WITHOUT referencing any success stories
    #         4. **Never make up projects, clients, or outcomes** - If it's not in the context, don't mention it

    #         ---

    #         **USER QUESTION:**
    #         {query}

    #         **AVAILABLE SUCCESS STORIES CONTEXT:**
    #         {context if context and context != "No relevant success stories found." else "[NO DATA AVAILABLE]"}

    #         ---

    #         **YOUR RESPONSE GUIDELINES:**

    #         **SCENARIO 1 - Greeting Detection (hi, hello, hey, good morning, etc.):**
    #         - Start response with the tag: `[GREETING]`
    #         - Provide ONLY this generic welcome message:
    #         "Hello! 👋 I'm here to help you explore our client success stories. 
            
    #         I can assist you with information about:
    #         • Industry-specific projects (Banking, Healthcare, Retail, etc.)
    #         • Technology implementations (Cloud, AI/ML, Data Analytics, etc.)
    #         • Digital transformation initiatives
    #         • Legacy modernization case studies
            
    #         What would you like to know about?"
    #         - **DO NOT** reference any specific success stories, projects, or clients
    #         - **DO NOT** use the context provided above for greeting responses

    #         **SCENARIO 2 - Specific Query WITH Available Context:**
    #         - Provide a clear, structured response using ONLY the context above
    #         - Format your response as:
    #         1. **Brief Summary** (2-3 sentences)
    #         2. **Key Details** from relevant success stories (use bullet points)
    #         3. **Technologies/Outcomes** mentioned in the context
    #         4. **Source Citation** (mention which success story/project you're referencing)
    #         - Use markdown formatting for readability
    #         - Include specific metrics, outcomes, or achievements ONLY if present in the context
    #         - Always cite the source story/project name when referencing information

    #         **SCENARIO 3 - Specific Query BUT No Relevant Context (context is empty or "[NO DATA AVAILABLE]"):**
    #         - Start response with: "I couldn't find any success stories matching your query in our current database."
    #         - Suggest rephrasing with broader terms
    #         - Provide example queries:
    #         • "Show me cloud migration projects"
    #         • "Tell me about banking sector implementations"
    #         • "What AI/ML success stories do you have?"
    #         - **DO NOT** make up or generate any project information
    #         - **DO NOT** reference generic industry knowledge

    #         **TONE & STYLE:**
    #         - Professional yet conversational
    #         - Concise and well-structured
    #         - Honest about limitations (if info isn't in context, say so clearly)
    #         - Use markdown formatting (headers, bullet points, bold text)

    #         **REMEMBER:** You can ONLY work with the context provided above. If information isn't there, acknowledge it honestly and guide the user to refine their search.
    #         """
    
    def _build_system_prompt(self, query:str, context:str) -> str:
        """Build the system prompt for the chatbot"""
        return """You are an AI assistant specialized in client success stories. 

                    Your role is to:
                    1. Provide clear, accurate information about past client engagements and success stories from the provided context
                    2. Reference specific projects, technologies, and outcomes from the provided context
                    3. Format responses in a professional, easy-to-read manner using markdown
                    4. Cite sources by mentioning the story/project name when referencing information
                    5. If information isn't in the provided context, politely say so and offer to search more broadly

                    Response Guidelines:
                    - **For greetings (hi, hello, hey, etc.):** Start your response with "[GREETING]" tag, then respond warmly and briefly explain what you can help with. DO NOT reference any specific success stories or context provided. Keep it generic and welcoming. Example: "[GREETING] Hello! Welcome..."
                    - **For specific questions about success stories:** Start with a brief summary answer, provide specific details from relevant success stories, use bullet points for key accomplishments or features, include relevant metrics and outcomes when available, end with a brief takeaway or recommendation, and always cite which success story/project you're referencing.
                    
                    Be conversational, professional, and helpful.
                """
                    
    
    def _build_user_prompt(self, query:str, context:str) -> str:
        """Build the system prompt for the chatbot"""
        
        return f"""Based on the following success stories, please answer this question:

                    **Question:** {query}

                    **Relevant Success Stories:**
                    {context}

                    Please provide a comprehensive answer using the information above. Include:
                    1. A brief summary
                    2. Specific details from relevant projects
                    3. Key accomplishments or technologies
                    4. Citations to the specific success stories you reference"""

    def _format_context(self, search_results: List[Dict]) -> str:
        """Format search results into context for the LLM"""
        if not search_results:
            return "No relevant success stories found."
        
        # Group by story
        stories = {}
        for result in search_results:
            story_id = result['story_id']
            if story_id not in stories:
                stories[story_id] = {
                    'category': result.get('category', 'General'),
                    'source_file': result.get('source_file', 'Unknown'),
                    'chunks': []
                }
            stories[story_id]['chunks'].append({
                'content': result.get('chunk_text', ''),
                'similarity': result.get('similarity', 0.0)
            })
        
        # Format context
        context_parts = []
        for story_id, data in stories.items():
            # Extract readable title from source file
            title = data['source_file'].replace('.pdf', '').replace('.docx', '').replace('_', ' ')
            title = ' - '.join(title.split('/')[-1].split('-')[-2:]) if '-' in title else title
            
            context_parts.append(f"\n## 📄 Success Story: {title}")
            context_parts.append(f"**Category:** {data['category']}")
            context_parts.append(f"**Source:** {story_id}\n")
            
            # Add top chunks
            top_chunks = sorted(data['chunks'], key=lambda x: x['similarity'], reverse=True)[:2]
            for i, chunk in enumerate(top_chunks, 1):
                context_parts.append(f"### Excerpt {i}:")
                context_parts.append(chunk['content'].strip())
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def ask(self, query: str, top_k: int = 5, stream: bool = False) -> str:
        """
        Ask a question and get a conversational response
        
        Args:
            query: User's question
            top_k: Number of relevant stories to retrieve
            stream: Whether to stream the response (for real-time display)
            
        Returns:
            Formatted response string
        """
        try:
            # Step 1: Retrieve relevant context
            self.logger.info(f"🔍 Searching for: '{query}'")
            search_results = self.rag_pipeline.search(query, top_k=top_k)
            
            if not search_results:
                return self._no_results_response(query)
            
            # Step 2: Format context
            context = self._format_context(search_results)
            
            # Step 3: Build prompt
            system_prompt = self._build_system_prompt()
            
            user_prompt = f"""Based on the following success stories, please answer this question:

                                **Question:** {query}

                                **Relevant Success Stories:**
                                {context}

                                Please provide a comprehensive answer using the information above. Include:
                                1. A brief summary
                                2. Specific details from relevant projects
                                3. Key accomplishments or technologies
                                4. Citations to the specific success stories you reference"""
            
            # Step 4: Generate response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            # Step 5: Add to conversation history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'answer': answer,
                'num_sources': len(search_results)
            })
            
            # Step 6: Format final response
            formatted_response = self._format_response(answer, search_results)
            
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"❌ Error generating response: {e}")
            return f"❌ Sorry, I encountered an error: {str(e)}"
    
    def _no_results_response(self, query: str) -> str:
        """Generate response when no relevant stories are found"""
        return f"""
                ╔══════════════════════════════════════════════════════════════════════════════╗
                ║                          🔍 NO RESULTS FOUND                                  ║
                ╚══════════════════════════════════════════════════════════════════════════════╝

                I couldn't find any success stories directly related to **"{query}"** in our database.

                **💡 Suggestions:**
                - Try rephrasing your question with different keywords
                - Use broader terms (e.g., "cloud migration" instead of "AWS EC2 migration")
                - Ask about general categories like:
                - Banking & Financial Services
                - Healthcare & Life Sciences
                - Retail & E-commerce
                - Manufacturing & IoT
                - Cloud Migration & Modernization

                **🤔 Example questions:**
                - "Show me success stories about cloud migration"
                - "What projects have we done in the banking sector?"
                - "Tell me about Azure implementations"

                Would you like to try a different search?
                """
    
    def _format_response(self, answer: str, sources: List[Dict]) -> str:
        """Format the final response with sources"""
        
        # Check if this is a greeting response
        is_greeting = answer.startswith('[GREETING]')
        if is_greeting:
            # Remove the [GREETING] tag
            answer = answer.replace('[GREETING]', '').strip()
        
        # Extract unique stories
        unique_stories = {}
        for source in sources:
            story_id = source['story_id']
            if story_id not in unique_stories:
                title = source.get('source_file', story_id).replace('.pdf', '').replace('.docx', '')
                title = ' - '.join(title.split('/')[-1].split('-')[-2:]) if '-' in title else title
                unique_stories[story_id] = {
                    'title': title,
                    'category': source.get('category', 'General'),
                    'relevance': source.get('similarity', 0) * 100
                }
        
        # Format response
        formatted = []
        formatted.append("╔" + "═" * 78 + "╗")
        formatted.append("║" + " " * 25 + "💬 RESPONSE" + " " * 42 + "║")
        formatted.append("╚" + "═" * 78 + "╝\n")
        
        # Add answer with proper formatting
        for line in answer.split('\n'):
            if line.strip():
                # Handle markdown headers
                if line.startswith('#'):
                    formatted.append(f"\n{line}\n")
                # Handle bullet points
                elif line.strip().startswith(('- ', '* ', '• ')):
                    formatted.append(f"  {line}")
                # Handle numbered lists
                elif len(line) > 0 and line[0].isdigit() and '. ' in line[:4]:
                    formatted.append(f"  {line}")
                else:
                    formatted.append(line)
        
        # Add sources only if NOT a greeting
        if not is_greeting:
            formatted.append("\n\n" + "─" * 80)
            formatted.append("\n📚 **SOURCES** (Based on the following success stories):\n")
            
            for idx, (story_id, info) in enumerate(sorted(
                unique_stories.items(), 
                key=lambda x: x[1]['relevance'], 
                reverse=True
            ), 1):
                formatted.append(f"{idx}. **{info['title']}**")
                formatted.append(f"   Category: {info['category']} | Relevance: {info['relevance']:.1f}%")
            
            formatted.append("\n" + "─" * 80 + "\n")
        else:
            formatted.append("\n")
        
        return "\n".join(formatted)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.logger.info("🗑️  Conversation history cleared")
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def display_history(self):
        """Display conversation history"""
        if not self.conversation_history:
            print("📭 No conversation history yet.")
            return
        
        print("\n" + "═" * 80)
        print("📜 CONVERSATION HISTORY")
        print("═" * 80 + "\n")
        
        for i, entry in enumerate(self.conversation_history, 1):
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i}. [{timestamp}] 🙋 Question: {entry['query']}")
            print(f"   📊 Found {entry['num_sources']} relevant sources")
            print()


# Helper function to create chatbot instance
def create_chatbot(rag_pipeline: RAGPipeline) -> SuccessStoryChatBot:
    """Create and return a chatbot instance"""
    return SuccessStoryChatBot(rag_pipeline)