"""
Literature Review Module
Uses GPT-5 with web search to conduct literature review before training function generation
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import OpenAI
from config import config
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LiteratureReview:
    """Literature review result"""
    query: str
    review_text: str
    key_findings: List[str]
    recommended_approaches: List[str]
    recent_papers: List[Dict[str, str]]
    confidence: float
    timestamp: int

class LiteratureReviewGenerator:
    """Literature review generator using GPT-5 with web search"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.openai_api_key
        self.base_url = base_url or config.openai_base_url
        self.model = model or "gpt-5"  # Default to GPT-5 for web search capabilities

        if not self.api_key:
            raise ValueError("OpenAI API key is required but not configured.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Create directory for storing literature reviews
        self.review_storage_dir = Path("literature_reviews")
        self.review_storage_dir.mkdir(exist_ok=True)

    def _create_literature_query(self, data_profile: Dict[str, Any], input_shape: tuple, num_classes: int) -> str:
        """Create research query based on data characteristics"""

        data_type = data_profile.get('data_type', 'tabular')
        num_samples = data_profile.get('sample_count', data_profile.get('num_samples', 'unknown'))
        is_sequence = data_profile.get('is_sequence', False)

        # Build specific query based on data characteristics
        query_parts = []
        
        # Add dataset-specific information from environment
        dataset_name = config.dataset_name
        if dataset_name and dataset_name != "Unknown Dataset":
            if "MIT-BIH" in dataset_name or "ECG" in dataset_name.upper():
                query_parts.extend(["ECG classification", "arrhythmia detection", "heart rhythm analysis"])
            elif "EEG" in dataset_name.upper():
                query_parts.extend(["EEG classification", "brain signal analysis"])
            elif "EMG" in dataset_name.upper():
                query_parts.extend(["EMG classification", "muscle signal analysis"])
            else:
                # Extract key terms from dataset name
                query_parts.append(dataset_name.replace(" Database", "").replace(" Dataset", ""))

        if is_sequence or len(input_shape) > 1:
            query_parts.append("sequence classification")
            query_parts.append("time series machine learning")
            if num_classes > 2:
                query_parts.append("multiclass sequence classification")
        else:
            query_parts.append("tabular data classification")
            if num_classes > 2:
                query_parts.append("multiclass classification")

        # Add data size considerations
        if isinstance(num_samples, int):
            if num_samples < 1000:
                query_parts.append("small dataset machine learning")
            elif num_samples < 10000:
                query_parts.append("medium dataset classification")
            else:
                query_parts.append("large dataset classification")

        # Add current year for recent research
        query_parts.append("2024 2025")
        query_parts.append("PyTorch implementation")
        query_parts.append("state-of-the-art methods")

        return " ".join(query_parts)

    def _call_gpt5_with_web_search(self, query: str, research_prompt: str) -> str:
        """Call GPT-5 with web search using the responses.create method"""
        logger.info(f"Making GPT-5 literature review call with query: {query}")

        # Use the GPT-5 responses.create method with web search
        response = self.client.responses.create(
            model=self.model,
            tools=[
                {"type": "web_search"}
            ],
            input=f"Research Query: {query}\n\nTask: {research_prompt}"
        )

        # Extract the output text
        if hasattr(response, 'output_text'):
            result = response.output_text
        elif hasattr(response, 'choices') and len(response.choices) > 0:
            result = response.choices[0].message.content
        elif hasattr(response, 'text'):
            result = response.text
        else:
            logger.warning("Unexpected response format, trying to extract content")
            result = str(response)

        logger.info("Successfully completed GPT-5 literature review with web search")
        return result

    def generate_literature_review(self, data_profile: Dict[str, Any], input_shape: tuple, num_classes: int) -> LiteratureReview:
        """Generate comprehensive literature review for the given ML problem"""

        # Create research query
        query = self._create_literature_query(data_profile, input_shape, num_classes)

        # Create detailed research prompt with dataset context
        dataset_context = ""
        if config.dataset_name and config.dataset_name != "Unknown Dataset":
            dataset_context = f"""
        Dataset: {config.dataset_name}
        Source: {config.dataset_source}"""
        
        research_prompt = f"""
        Conduct a comprehensive literature review for a machine learning classification problem with the following characteristics:

        Data Type: {data_profile.get('data_type', 'unknown')}
        Input Shape: {input_shape}
        Number of Classes: {num_classes}
        Number of Samples: {data_profile.get('num_samples', 'unknown')}
        Is Sequence Data: {data_profile.get('is_sequence', False)}{dataset_context}

        Please provide a structured literature review including:

        1. RECENT DEVELOPMENTS (2023-2025):
           - Latest state-of-the-art methods for this type of problem
           - Recent breakthrough papers and their key contributions
           - Emerging architectures and techniques

        2. RECOMMENDED APPROACHES:
           - Top 3-5 most effective methods for this specific problem type
           - Rationale for why these approaches work well
           - Typical hyperparameter ranges and architecture choices

        3. KEY FINDINGS:
           - Best practices for this type of data and problem size
           - Common pitfalls to avoid
           - Performance benchmarks and expected accuracy ranges

        4. IMPLEMENTATION CONSIDERATIONS:
           - PyTorch-specific implementation tips
           - Memory and computational requirements
           - Regularization and optimization strategies

        Format your response as structured JSON with the following fields:
        - review_text: comprehensive review (2-3 paragraphs)
        - key_findings: list of 5-7 key findings
        - recommended_approaches: list of 3-5 recommended methods with brief descriptions
        - recent_papers: list of recent relevant papers with titles and key contributions
        - confidence: confidence score (0.0-1.0)
        """

        # Get literature review from GPT-5 with web search
        response = self._call_gpt5_with_web_search(query, research_prompt)

        # Parse the response
        review = self._parse_literature_review(response, query)

        logger.info(f"Literature review completed with confidence: {review.confidence:.2f}")
        logger.info(f"Found {len(review.recommended_approaches)} recommended approaches")

        return review

    def _parse_literature_review(self, response: str, query: str) -> LiteratureReview:
        """Parse the literature review response"""
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]

            # Find JSON boundaries
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx != -1 and end_idx > 0:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)

                return LiteratureReview(
                    query=query,
                    review_text=data.get("review_text", response),
                    key_findings=data.get("key_findings", []),
                    recommended_approaches=data.get("recommended_approaches", []),
                    recent_papers=data.get("recent_papers", []),
                    confidence=float(data.get("confidence", 0.8)),
                    timestamp=int(time.time())
                )
            else:
                # Fallback: parse as plain text
                logger.warning("Could not parse JSON response, using plain text format")
                return LiteratureReview(
                    query=query,
                    review_text=response,
                    key_findings=self._extract_key_points(response),
                    recommended_approaches=self._extract_recommendations(response),
                    recent_papers=[],
                    confidence=0.7,
                    timestamp=int(time.time())
                )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Return plain text review
            return LiteratureReview(
                query=query,
                review_text=response,
                key_findings=self._extract_key_points(response),
                recommended_approaches=self._extract_recommendations(response),
                recent_papers=[],
                confidence=0.6,
                timestamp=int(time.time())
            )

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from plain text review"""
        lines = text.split('\n')
        key_points = []

        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('• ') or line.startswith('* '):
                key_points.append(line[2:])
            elif line and any(keyword in line.lower() for keyword in ['key', 'important', 'finding', 'recommendation']):
                key_points.append(line)

        return key_points[:7]  # Limit to 7 key points

    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from plain text review"""
        lines = text.split('\n')
        recommendations = []

        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'approach', 'method', 'technique']):
                if line and len(line) > 10:  # Filter out very short lines
                    recommendations.append(line)

        return recommendations[:5]  # Limit to 5 recommendations

    def save_literature_review(self, review: LiteratureReview, data_profile: Dict[str, Any]) -> str:
        """Save literature review to text file"""

        # Create filename based on data characteristics and timestamp
        data_type = data_profile.get('data_type', 'unknown')
        filename = f"literature_review_{data_type}_{review.timestamp}.txt"
        filepath = self.review_storage_dir / filename

        # Format the review for text file
        review_content = f"""LITERATURE REVIEW
=================

Query: {review.query}
Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(review.timestamp))}
Confidence: {review.confidence:.2f}

DATA PROFILE:
{json.dumps(data_profile, indent=2)}

COMPREHENSIVE REVIEW:
{review.review_text}

KEY FINDINGS:
"""

        for i, finding in enumerate(review.key_findings, 1):
            finding_str = str(finding) if finding else 'No finding'
            review_content += f"{i}. {finding_str}\n"

        review_content += f"\nRECOMMENDED APPROACHES:\n"
        for i, approach in enumerate(review.recommended_approaches, 1):
            approach_str = str(approach) if approach else 'No approach'
            review_content += f"{i}. {approach_str}\n"

        if review.recent_papers:
            review_content += f"\nRECENT PAPERS:\n"
            for paper in review.recent_papers:
                title = str(paper.get('title', 'Unknown title')) if paper.get('title') else 'Unknown title'
                contribution = str(paper.get('contribution', 'No description')) if paper.get('contribution') else 'No description'
                review_content += f"- {title}: {contribution}\n"

        review_content += f"\n" + "="*50 + "\n"

        # Save to text file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(review_content)

        logger.info(f"Literature review saved to: {filepath}")
        return str(filepath)

    def load_literature_review(self, filepath: str) -> str:
        """Load literature review from text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

# Global instance
literature_review_generator = LiteratureReviewGenerator()

def generate_literature_review_for_data(data_profile: Dict[str, Any], input_shape: tuple, num_classes: int) -> LiteratureReview:
    """
    Convenience function: Generate literature review for data
    """
    return literature_review_generator.generate_literature_review(data_profile, input_shape, num_classes)