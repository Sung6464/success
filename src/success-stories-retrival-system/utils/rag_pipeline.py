"""Adapted RAG pipeline wrapper using Proxy RAG."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional

from utils.proxy_rag import MMRagBot, config
from utils.proxy_rag.openai_client import generate_multimodal

logger = logging.getLogger(__name__)


class MockVectorStore:
    def close_all_connections(self):
        pass


class SuccessStoryChatBot:
    def __init__(self, embedding_generator=None, vector_store=None):
        """Initialize adapted chatbot wrapping MMRagBot."""
        self.bot = MMRagBot()
        self.vector_store = vector_store or MockVectorStore()
        logger.info("✅ SuccessStoryChatBot wrapper initialized with Proxy RAG")

    def _is_followup_query(self, query: str, conversation_history: List[Dict]) -> bool:
        """Detect if query is a follow-up question."""
        return self.bot._is_followup_query(query, conversation_history)

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.4,
        previous_sources: Optional[List[Dict]] = None,
        is_followup: bool = False,
    ) -> List[Dict]:
        """Retrieve matching sections from local FAISS index."""
        cands = self.bot._recall(query)
        finalists = self.bot._rerank(query, cands)

        results = []
        for f in finalists:
            results.append({
                'story_id': f.get('doc_id') or 'Unknown',
                'file_name': f.get('doc_id') or 'Unknown',
                'similarity': f.get('score') or 0.85,
                'category': f.get('category') or 'general',
                'chunk_text': f.get('text') or '',
                'content': f.get('text') or '',
                'source': f.get('download_url') or f.get('url') or '',
                'breadcrumb': f.get('breadcrumb'),
                'images': f.get('images', []),
                'doc_id': f.get('doc_id'),
            })
        return results

    def generate_response_structured(
        self,
        query: str,
        search_results: List[Dict],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, any]:
        """Generate response utilizing multimodal vision synthesis."""
        context_blocks = []
        image_paths = []
        filename_to_path = {}
        file_mapping = {}
        citation_num = 1

        seen_files = set()
        for f in search_results:
            doc_id = f.get('doc_id') or 'Unknown'
            if doc_id not in seen_files:
                file_mapping[citation_num] = {
                    'file_name': doc_id,
                    'url': f.get('source') or ''
                }
                seen_files.add(doc_id)
                citation_num += 1

        doc_to_citation = {info['file_name']: num for num, info in file_mapping.items()}

        for f in search_results:
            doc_id = f.get('doc_id') or 'Unknown'
            citation_num_for_doc = doc_to_citation.get(doc_id, '?')

            block = [f"### [Citation [{citation_num_for_doc}]] {f.get('breadcrumb')}", f.get('chunk_text', '')]

            images = f.get('images', [])
            if images:
                names = []
                for rel in images:
                    p = config.PAPERS_DIR / doc_id / rel
                    fname = Path(rel).name
                    filename_to_path[fname] = p
                    if p.exists() and p not in image_paths and len(image_paths) < 12:
                        image_paths.append(p)
                    names.append(fname)
                block.append(f"[Images available in this section: {', '.join(names)}]")
            context_blocks.append("\n".join(block))

        context = "\n\n---\n\n".join(context_blocks)

        mapping_str = "\n".join([
            f"[{num}] {info['file_name']} -> {info['url']}"
            for num, info in file_mapping.items()
        ])

        system_prompt = f"""You are an AI assistant specialized in client success stories. Respond to the user's question using ONLY the provided document sections and images.
        
        Inline citations:
        - Cite sources immediately after factual statements using inline citation numbers [1], [2], etc., matching the "Available Source Files" list.
        - Set inline citation numbers to match the document [Citation [n]] header of the section from which the information is taken.
        - Ensure every metric or outcome is cited.
        
        Inline figure rendering:
        - When a provided image supports your answer, cite/show it inline by emitting a directive on its own line in EXACTLY this format:
          [SHOW: <image_filename> | <short caption of what it shows>]
          Use the bare filename (e.g. img_4.png), not a path. Only SHOW images that appear in the provided context.
          
        Available Source Files for Citation:
        {mapping_str}
        
        Respond with a professional, concise markdown answer.
        """

        prompt = (
            f"QUESTION:\n{query}\n\n"
            f"DOCUMENT SECTIONS:\n\n{context}\n\n"
            f"Attached below are the actual images referenced above. Write the answer now."
        )

        history_str = ""
        if conversation_history:
            history_str = "Conversation History (for context):\n"
            for msg in conversation_history[-5:]:
                history_str += f"- {msg['role']}: {msg['content']}\n"
            history_str += "\n"

        prompt = history_str + prompt

        answer_text = generate_multimodal(prompt, image_paths, system=system_prompt)

        sources_list = []
        for num, info in file_mapping.items():
            sources_list.append({
                'file_name': f"[{num}] {info['file_name']}",
                'download_url': info['url']
            })

        return {
            'response': answer_text,
            'sources': sources_list
        }