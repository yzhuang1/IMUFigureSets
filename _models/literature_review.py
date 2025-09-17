"""
Literature Review Module
Uses GPT-5 with web search to conduct literature review before training function generation
"""

import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import OpenAI
from config import config
import os
from pathlib import Path

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
            pass
            result = str(response)

        return result

    def generate_literature_review(self, data_profile: Dict[str, Any], input_shape: tuple, num_classes: int) -> LiteratureReview:
        """Generate comprehensive literature review for the given ML problem"""
        print("Literature review started")

        # Create research query
        query = self._create_literature_query(data_profile, input_shape, num_classes)

        # Create detailed research prompt with dataset context
        dataset_context = ""
        if config.dataset_name and config.dataset_name != "Unknown Dataset":
            dataset_context = f"""
        Dataset: {config.dataset_name}
        Source: {config.dataset_source}"""
        
        research_prompt = f"""
        SYSTEMATIC LITERATURE REVIEW TASK:
        
        STEP 1 - PROBLEM ANALYSIS:
        Data Type: {data_profile.get('data_type', 'unknown')}
        Input Shape: {input_shape}
        Number of Classes: {num_classes}
        Number of Samples: {data_profile.get('num_samples', 'unknown')}
        Is Sequence Data: {data_profile.get('is_sequence', False)}{dataset_context}
        
        STEP 2 - RESEARCH METHODOLOGY:
        1. Search for state-of-the-art papers specifically addressing this problem type
        2. Focus on papers with similar data characteristics (shape, size, domain)
        3. Prioritize papers with empirical results and benchmark comparisons
        4. Look for recent surveys or meta-analyses in this domain
        
        STEP 3 - INFORMATION EXTRACTION:
        For each relevant paper, extract:
        - Model architecture details
        - Performance metrics on similar datasets
        - Key innovations or techniques
        - Computational requirements
        
        STEP 4 - SYNTHESIS AND RECOMMENDATION:
        Based on the research, provide ONE BEST model recommendation that:
        - Matches the data characteristics
        - Has proven performance on similar problems
        - Is implementable in PyTorch
        - Balances accuracy with computational efficiency
        
        REQUIRED OUTPUT FORMAT (JSON):
        {{
            "review_text": "Comprehensive summary of research findings with specific paper citations and performance metrics",
            "key_findings": [
                "Specific finding with paper reference and metrics",
                "Another key insight with quantitative evidence",
                "Third finding with architectural details"
            ],
            "recommended_approaches": ["ONE specific model architecture with justification"],
            "recent_papers": [
                {{"title": "Paper Title", "contribution": "Specific contribution and results"}},
                {{"title": "Paper Title 2", "contribution": "Key innovation and performance"}}
            ],
            "confidence": 0.0-1.0
        }}
        
        FOCUS: You can only output a json format.
        """

        # Get literature review from GPT-5 with web search
        response = self._call_gpt5_with_web_search(query, research_prompt)

        # Parse the response
        review = self._parse_literature_review(response, query)
        
        print("Literature review finished")
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
                print(f"Literature review response missing required JSON structure")
                print(f"Response content: {response}")
                raise ValueError(f"Literature review must return valid JSON with required fields. Got: {data}")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Literature review response is not valid JSON format: {e}")
            print(f"Response content: {response}")
            raise ValueError(f"Literature review must return valid JSON format. Got invalid response: {e}")

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
                title = paper.get('title', 'Unknown title') if isinstance(paper, dict) else 'Unknown title'
                contribution = paper.get('contribution', 'No description') if isinstance(paper, dict) else 'No description'
                review_content += f"- {title}: {contribution}\n"

        review_content += f"\n" + "="*50 + "\n"

        # Save to text file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(review_content)

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