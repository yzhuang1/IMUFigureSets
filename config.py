"""
Configuration management for the ML pipeline
Handles secure loading of API keys and other sensitive configuration
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for managing API keys and settings"""
    
    def __init__(self):
        self.openai_api_key = self._get_openai_api_key()
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-5")
        
        # Dataset information for GPT-5 context
        self.dataset_name = os.getenv("DATASET_NAME", "Unknown Dataset")
        self.dataset_source = os.getenv("DATASET_SOURCE", "Unknown Source")
        
        # Literature review configuration
        self.skip_literature_review = os.getenv("SKIP_LITERATURE_REVIEW", "false").lower() == "true"
        
        # API call limits to prevent infinite loops and excessive costs
        # These values should be set in .env file, fallback values provided as backup
        self.max_bo_trials = int(os.getenv("MAX_BO_TRIALS") or "40")
        self.debug_chances = int(os.getenv("DEBUG_CHANCES") or "4")
    
    def _get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment variables or .env file"""
        # Try to get from environment variable first
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            # Try to get from .env file
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            logger = logging.getLogger(__name__)
            logger.warning("OPENAI_API_KEY not found in environment variables or .env file")
            logger.warning("Please set OPENAI_API_KEY in your environment or create a .env file")
        
        return api_key
    
    def is_openai_configured(self) -> bool:
        """Check if OpenAI is properly configured"""
        return self.openai_api_key is not None and len(self.openai_api_key.strip()) > 0

# Global configuration instance
config = Config()
