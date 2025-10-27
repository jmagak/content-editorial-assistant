"""
Next-Generation Taxonomy Classifier

Uses semantic similarity with sentence transformers as primary method,
with robust fallback to keyword-based classification for reliability.
"""

import logging
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
import re

logger = logging.getLogger(__name__)

# Try to import sentence transformers for semantic classification
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.info("sentence-transformers not available, using keyword-based classification")

# Try to import spacy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class NextGenTaxonomyClassifier:
    """
    Next-generation taxonomy classifier using semantic similarity.
    
    Primary approach: Semantic similarity using sentence transformers
    Fallback approach: Keyword-based classification for reliability
    """
    
    def __init__(self, model_manager=None):
        """
        Initialize the classifier with disk-cached embeddings.
        
        Args:
            model_manager: Optional ModelManager for integration
        """
        self.model_manager = model_manager
        self.sentence_transformer = None
        self.category_embeddings = {}
        self.fallback_classifier = LegacyTaxonomyClassifier()
        
        # World-class: Set up embedding cache directory
        self.embedding_cache_dir = self._get_embedding_cache_dir()
        
        self._initialize_sentence_transformer()
        
        # World-class: Load pre-computed category embeddings from disk
        self._load_cached_embeddings()
    
    def classify_taxonomy(self, content: str, spacy_doc, keywords: List[str], 
                         taxonomy_config: Dict) -> List[Dict[str, Any]]:
        """
        Classify content taxonomy using semantic similarity.
        
        Args:
            content: Raw document text
            spacy_doc: Parsed spaCy document (may be None)
            keywords: Extracted keywords
            taxonomy_config: Taxonomy configuration
            
        Returns:
            List of taxonomy classifications with confidence scores
        """
        try:
            # PRIMARY: Semantic similarity approach
            if self.sentence_transformer and SENTENCE_TRANSFORMERS_AVAILABLE:
                return self._classify_with_semantic_similarity(
                    content, spacy_doc, keywords, taxonomy_config
                )
            else:
                logger.info("Sentence transformer not available, using fallback classification")
                return self.fallback_classifier.classify_taxonomy(
                    content, spacy_doc, keywords, taxonomy_config
                )
                
        except Exception as e:
            logger.error(f"Semantic classification failed: {e}")
            # Graceful fallback to keyword-based approach
            return self.fallback_classifier.classify_taxonomy(
                content, spacy_doc, keywords, taxonomy_config
            )
    
    def _classify_with_semantic_similarity(self, content: str, spacy_doc, keywords: List[str], 
                                         taxonomy_config: Dict) -> List[Dict[str, Any]]:
        """Semantic similarity classification using sentence transformers."""
        
        # Generate document summary for embedding
        document_summary = self._generate_semantic_summary(content, spacy_doc, keywords)
        
        # Create document embedding
        try:
            document_embedding = self.sentence_transformer.encode(document_summary)
        except Exception as e:
            logger.error(f"Failed to encode document: {e}")
            # Fall back to keyword classification
            return self.fallback_classifier.classify_taxonomy(
                content, spacy_doc, keywords, taxonomy_config
            )
        
        taxonomy_scores = {}
        
        for category, config in taxonomy_config.items():
            try:
                # Get precomputed category embedding (with disk caching)
                if category not in self.category_embeddings:
                    # World-class: Try to load from disk cache first
                    cached_embedding = self._load_category_embedding_from_cache(category)
                    if cached_embedding is not None:
                        self.category_embeddings[category] = cached_embedding
                        logger.debug(f"Loaded cached embedding for category: {category}")
                    else:
                        # Compute and save to cache
                        embedding = self._create_category_embedding(config)
                        self.category_embeddings[category] = embedding
                        self._save_category_embedding_to_cache(category, embedding)
                        logger.debug(f"Computed and cached embedding for category: {category}")
                
                category_embedding = self.category_embeddings[category]
                
                # Calculate semantic similarity (PRIMARY SCORE)
                semantic_similarity = self._cosine_similarity(document_embedding, category_embedding)
                
                # Keyword boost (SUPPLEMENTARY)
                keyword_boost = self._calculate_keyword_boost(keywords, config.get('indicators', []))
                
                # Structure analysis boost (SUPPLEMENTARY) - using spacy if available
                structure_boost = self._analyze_document_structure(spacy_doc, config.get('structure_hints', []))
                
                # Hybrid scoring: semantic similarity as primary, others as boosts
                base_score = max(semantic_similarity * 0.7, 0)  # 70% semantic, ensure non-negative
                boosted_score = base_score + (keyword_boost * 0.2) + (structure_boost * 0.1)
                
                taxonomy_scores[category] = {
                    'category': category,
                    'score': min(boosted_score, 1.0),
                    'confidence': self._calculate_confidence(semantic_similarity, keyword_boost, structure_boost),
                    'semantic_similarity': semantic_similarity,
                    'keyword_boost': keyword_boost,
                    'structure_boost': structure_boost,
                    'classification_method': 'semantic_similarity'
                }
                
            except Exception as e:
                logger.warning(f"Failed to process category {category}: {e}")
                continue
        
        # Filter by confidence threshold and return top matches
        high_confidence = [cat for cat in taxonomy_scores.values() if cat['confidence'] >= 0.2]
        return sorted(high_confidence, key=lambda x: x['confidence'], reverse=True)[:3]
    
    def _generate_semantic_summary(self, content: str, spacy_doc, keywords: List[str]) -> str:
        """
        Generate a rich semantic summary for embedding.
        
        World-class enhancement: Include title, key sentences, section headings,
        technical terms, and positional context for maximum accuracy.
        """
        components = []
        
        # 1. Extract probable title (highest weight)
        title = self._extract_probable_title(content)
        if title:
            components.append(f"Title: {title}")
        
        # 2. Extract section headings (structural signals)
        headings = self._extract_section_headings(content)
        if headings:
            components.append(f"Sections: {' | '.join(headings[:3])}")
        
        # 3. Extract key sentences (opening + closing for context)
        key_sentences = self._extract_key_sentences_enhanced(content, max_sentences=3)
        if key_sentences:
            components.append(' '.join(key_sentences))
        
        # 4. Use top keywords with technical term boosting
        if keywords:
            # Separate technical terms from regular keywords
            technical_keywords = [kw for kw in keywords[:8] if self._is_technical_term(kw)]
            regular_keywords = [kw for kw in keywords[:8] if kw not in technical_keywords]
            
            if technical_keywords:
                components.append(f"Technical: {' '.join(technical_keywords[:5])}")
            if regular_keywords:
                components.append(f"Keywords: {' '.join(regular_keywords[:5])}")
        
        # 5. Add content type indicators if detectable
        content_indicators = self._detect_content_indicators(content)
        if content_indicators:
            components.append(f"Type: {content_indicators}")
        
        # Combine all components into rich semantic representation
        summary = '. '.join(components)
        
        # Limit to reasonable length for embedding (optimal: 200-400 chars)
        if len(summary) > 400:
            summary = summary[:397] + '...'
        
        return summary
    
    def _extract_probable_title(self, content: str) -> str:
        """Extract a probable title from content."""
        # Look for heading patterns
        lines = content.split('\n')[:10]  # First 10 lines
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for markdown/asciidoc heading
            if (line.startswith('#') or line.startswith('=') or 
                (len(line) < 100 and line[0].isupper())):
                title = re.sub(r'^[#=\s]+', '', line).strip()
                if title:
                    return title
        
        # Fallback to first sentence
        sentences = content.split('.')[:1]
        if sentences:
            return sentences[0].strip()[:100]
        
        return "Document"
    
    def _extract_section_headings(self, content: str) -> List[str]:
        """Extract section headings for structural signals."""
        headings = []
        
        # AsciiDoc headings (== Section)
        asciidoc_headings = re.findall(r'^==+\s+(.+)$', content, re.MULTILINE)
        headings.extend(asciidoc_headings[:5])
        
        # Markdown headings (## Section)
        if not headings:
            markdown_headings = re.findall(r'^#{2,6}\s+(.+)$', content, re.MULTILINE)
            headings.extend(markdown_headings[:5])
        
        return headings
    
    def _extract_key_sentences_enhanced(self, content: str, max_sentences: int = 3) -> List[str]:
        """
        Extract key sentences with positional awareness.
        
        World-class: First sentence + middle key sentence + last sentence
        provides beginning, context, and conclusion signals.
        """
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if 10 <= len(s.strip()) <= 200]
        
        if not sentences:
            return []
        
        if len(sentences) <= max_sentences:
            return sentences
        
        # World-class approach: beginning + middle + end
        key_sentences = []
        
        # First sentence (introduction)
        key_sentences.append(sentences[0])
        
        # Middle sentence (context) if we have room
        if max_sentences >= 2 and len(sentences) > 2:
            middle_idx = len(sentences) // 2
            key_sentences.append(sentences[middle_idx])
        
        # Last sentence (conclusion) if we have room and enough sentences
        if max_sentences >= 3 and len(sentences) > 4:
            key_sentences.append(sentences[-1])
        
        return key_sentences
    
    def _is_technical_term(self, term: str) -> bool:
        """Check if a term is technical (APIs, tools, technologies)."""
        technical_patterns = [
            r'^[A-Z][a-z]+[A-Z]',  # CamelCase (e.g., JavaScript, OpenShift)
            r'^[A-Z]{2,}$',        # Acronyms (e.g., API, REST, JSON)
            r'(?i)^(api|sdk|cli|http|https|ssh|tcp|ip|dns|url|json|xml|yaml|html|css|sql)$',
            r'(?i).*(server|client|protocol|framework|library|database|container|cluster)$'
        ]
        
        return any(re.match(pattern, term) for pattern in technical_patterns)
    
    def _detect_content_indicators(self, content: str) -> str:
        """Detect content type indicators for classification boost."""
        content_lower = content[:500].lower()  # First 500 chars
        
        indicators = []
        
        # Procedural indicators
        if re.search(r'\b(?:install|configure|setup|deploy)\b', content_lower):
            indicators.append('procedural')
        
        # Conceptual indicators
        if re.search(r'\b(?:understanding|overview|introduction|concept)\b', content_lower):
            indicators.append('conceptual')
        
        # Reference indicators
        if re.search(r'\b(?:reference|parameters?|options?|api|syntax)\b', content_lower):
            indicators.append('reference')
        
        return ' + '.join(indicators) if indicators else ''
    
    def _extract_key_sentences(self, content: str, max_sentences: int = 3) -> List[str]:
        """Extract key sentences for semantic analysis (legacy method)."""
        # Simple sentence extraction
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if 10 <= len(s.strip()) <= 200]
        
        # Return first few sentences as key sentences
        return sentences[:max_sentences]
    
    def _create_category_embedding(self, category_config: Dict) -> np.ndarray:
        """Create semantic embedding for a taxonomy category."""
        try:
            indicators = category_config.get('indicators', [])
            description = category_config.get('description', '')
            examples = category_config.get('examples', [])
            
            # Create rich semantic description
            semantic_description = f"{description} "
            if indicators:
                semantic_description += f"Keywords: {' '.join(indicators)} "
            if examples:
                semantic_description += f"Examples: {' '.join(examples)}"
            
            return self.sentence_transformer.encode(semantic_description)
            
        except Exception as e:
            logger.error(f"Failed to create category embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(384)  # Default dimension for all-MiniLM-L6-v2
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norm_a = np.linalg.norm(vec1)
            norm_b = np.linalg.norm(vec2)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return float(dot_product / (norm_a * norm_b))
        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_keyword_boost(self, keywords: List[str], indicators: List[str]) -> float:
        """Calculate keyword-based boost score."""
        if not keywords or not indicators:
            return 0.0
        
        # Count keyword matches with indicators
        keyword_text = ' '.join(keywords).lower()
        matches = sum(1 for indicator in indicators if indicator.lower() in keyword_text)
        
        # Normalize by number of indicators
        return min(matches / len(indicators), 1.0) if indicators else 0.0
    
    def _analyze_document_structure(self, spacy_doc, structure_hints: List[str]) -> float:
        """Analyze document structure for category hints."""
        if not structure_hints:
            return 0.0
        
        # Simple structure analysis based on content patterns
        structure_score = 0.0
        
        # This is a simplified version - in production, you'd analyze structural_blocks
        # For now, use simple text pattern matching
        if spacy_doc and SPACY_AVAILABLE:
            text = spacy_doc.text.lower()
            
            # Look for structure indicators
            if 'numbered_list' in structure_hints and bool(re.search(r'\d+\.|\d+\)', text)):
                structure_score += 0.2
            
            if 'code_blocks' in structure_hints and ('```' in text or '    ' in text):
                structure_score += 0.3
            
            if 'procedure_steps' in structure_hints and bool(re.search(r'\b(step|first|next|then|finally)\b', text)):
                structure_score += 0.2
            
            if 'warning_blocks' in structure_hints and bool(re.search(r'\b(warning|caution|important|note)\b', text)):
                structure_score += 0.2
        
        return min(structure_score, 1.0)
    
    def _calculate_confidence(self, semantic_sim: float, keyword_boost: float, structure_boost: float) -> float:
        """Calculate overall confidence score."""
        # Base confidence from semantic similarity
        base_confidence = max(semantic_sim, 0)
        
        # Keyword and structure agreement boosts confidence
        if keyword_boost > 0.5 and structure_boost > 0.3:
            base_confidence *= 1.2  # Agreement boost
        elif keyword_boost > 0.3 or structure_boost > 0.2:
            base_confidence *= 1.1  # Partial agreement boost
        
        return min(base_confidence, 1.0)
    
    def _initialize_sentence_transformer(self):
        """Initialize sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info("sentence-transformers not available")
            return
        
        try:
            # Use a lightweight, fast model that works well for classification
            model_name = 'all-MiniLM-L6-v2'  # 384 dimensions, fast, good quality
            
            logger.info(f"Initializing sentence transformer: {model_name}")
            self.sentence_transformer = SentenceTransformer(model_name)
            logger.info("Sentence transformer initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize sentence transformer: {e}")
            self.sentence_transformer = None
    
    def _get_embedding_cache_dir(self) -> Path:
        """Get directory for embedding cache (OpenShift-compatible)."""
        # Check for OpenShift/Docker environment
        if os.path.exists('/app/.cache'):
            cache_dir = Path('/app/.cache/taxonomy_embeddings')
        else:
            # Development environment
            cache_dir = Path.home() / '.cache' / 'metadata_assistant' / 'taxonomy_embeddings'
        
        # Create directory if it doesn't exist
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Fallback to /tmp if permissions issue
            cache_dir = Path('/tmp/taxonomy_embeddings')
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Using fallback cache directory: {cache_dir}")
        
        return cache_dir
    
    def _load_cached_embeddings(self):
        """Load all cached category embeddings from disk."""
        if not self.sentence_transformer:
            return
        
        try:
            cache_file = self.embedding_cache_dir / 'category_embeddings.pkl'
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Verify cache version matches current model
                    if cached_data.get('model') == 'all-MiniLM-L6-v2':
                        self.category_embeddings = cached_data.get('embeddings', {})
                        logger.info(f"Loaded {len(self.category_embeddings)} cached category embeddings from disk")
                    else:
                        logger.info("Cache model version mismatch, will recompute embeddings")
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}")
    
    def _save_all_embeddings_to_cache(self):
        """Save all category embeddings to disk cache."""
        if not self.sentence_transformer or not self.category_embeddings:
            return
        
        try:
            cache_file = self.embedding_cache_dir / 'category_embeddings.pkl'
            cache_data = {
                'model': 'all-MiniLM-L6-v2',
                'embeddings': self.category_embeddings,
                'version': '1.0'
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Saved {len(self.category_embeddings)} category embeddings to disk cache")
        except Exception as e:
            logger.warning(f"Failed to save embeddings to cache: {e}")
    
    def _load_category_embedding_from_cache(self, category: str) -> Optional[np.ndarray]:
        """Load a specific category embedding from disk cache."""
        # First check in-memory cache
        if category in self.category_embeddings:
            return self.category_embeddings[category]
        
        # Then check disk cache (already loaded in __init__)
        return None
    
    def _save_category_embedding_to_cache(self, category: str, embedding: np.ndarray):
        """Save a specific category embedding and update disk cache."""
        # Save all embeddings together (more efficient)
        self._save_all_embeddings_to_cache()


class LegacyTaxonomyClassifier:
    """Fallback keyword-based classifier for reliability."""
    
    def classify_taxonomy(self, content: str, spacy_doc, keywords: List[str], 
                         taxonomy_config: Dict) -> List[Dict[str, Any]]:
        """
        Legacy keyword-based classification as reliable fallback.
        
        Args:
            content: Raw document text
            spacy_doc: Parsed spaCy document (may be None)
            keywords: Extracted keywords
            taxonomy_config: Taxonomy configuration
            
        Returns:
            List of taxonomy classifications
        """
        taxonomy_scores = {}
        content_lower = content.lower()
        
        # Process each taxonomy category
        for category, config in taxonomy_config.items():
            try:
                indicators = config.get('indicators', [])
                required_keywords = config.get('required_keywords', [])
                structure_hints = config.get('structure_hints', [])
                confidence_boost = config.get('confidence_boost', 1.0)
                
                score = 0.0
                
                # Keyword matching score (40% of total)
                if keywords and indicators:
                    keyword_matches = 0
                    for kw in keywords:
                        kw_lower = kw.lower()
                        for ind in indicators:
                            if ind.lower() in kw_lower:
                                keyword_matches += 1
                                break
                    
                    keyword_score = keyword_matches / len(indicators) if indicators else 0
                    score += keyword_score * 0.4
                
                # Content similarity score (30% of total)
                if indicators:
                    content_matches = sum(1 for ind in indicators if ind.lower() in content_lower)
                    content_score = content_matches / len(indicators)
                    score += content_score * 0.3
                
                # Structure analysis score (20% of total)
                structure_score = self._analyze_document_structure_legacy(content_lower, structure_hints)
                score += structure_score * 0.2
                
                # Required keywords boost/penalty (10% of total)
                if required_keywords:
                    required_found = sum(1 for req in required_keywords if req.lower() in content_lower)
                    if required_found == 0:
                        score *= 0.5  # Strong penalty if no required keywords found
                    else:
                        score += 0.1  # Small boost if required keywords present
                
                # Apply category-specific confidence boost
                score *= confidence_boost
                
                taxonomy_scores[category] = {
                    'category': category,
                    'score': min(score, 1.0),
                    'confidence': min(score, 1.0),
                    'matched_indicators': [ind for ind in indicators if ind.lower() in content_lower],
                    'classification_method': 'keyword_based',
                    'keyword_boost': keyword_score if 'keyword_score' in locals() else 0.0,
                    'content_matches': content_matches if 'content_matches' in locals() else 0,
                    'structure_boost': structure_score
                }
                
            except Exception as e:
                logger.warning(f"Failed to process category {category}: {e}")
                continue
        
        # Filter by confidence threshold and return top matches
        high_confidence = [cat for cat in taxonomy_scores.values() if cat['confidence'] >= 0.2]
        return sorted(high_confidence, key=lambda x: x['confidence'], reverse=True)[:3]
    
    def _analyze_document_structure_legacy(self, content_lower: str, structure_hints: List[str]) -> float:
        """Analyze document structure using simple pattern matching."""
        if not structure_hints:
            return 0.0
        
        structure_score = 0.0
        hints_found = 0
        
        for hint in structure_hints:
            found = False
            
            if hint == 'numbered_list':
                if re.search(r'\d+\.|^\s*\d+\)', content_lower, re.MULTILINE):
                    found = True
            
            elif hint == 'procedure_steps':
                step_patterns = [r'\bstep\s+\d+', r'\b(first|second|third|next|then|finally)\b']
                if any(re.search(pattern, content_lower) for pattern in step_patterns):
                    found = True
            
            elif hint == 'code_blocks':
                if '```' in content_lower or re.search(r'^\s{4,}', content_lower, re.MULTILINE):
                    found = True
            
            elif hint == 'parameter_tables':
                if re.search(r'\|\s*\w+\s*\||\w+\s*:\s*\w+', content_lower):
                    found = True
            
            elif hint == 'warning_blocks':
                warning_words = ['warning', 'caution', 'important', 'note', 'attention']
                if any(word in content_lower for word in warning_words):
                    found = True
            
            elif hint == 'example_requests':
                if re.search(r'\b(get|post|put|delete|example)\b.*\{.*\}', content_lower, re.DOTALL):
                    found = True
            
            elif hint == 'error_messages':
                if re.search(r'\berror\b.*:|exception.*:|fault.*:', content_lower):
                    found = True
            
            # Add more structure hint patterns as needed
            
            if found:
                hints_found += 1
        
        # Return proportion of structure hints found
        return hints_found / len(structure_hints) if structure_hints else 0.0
