"""
Data Processor
==============
Clean and validate scraped SHL assessment data.
"""

import json
import logging
import re
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and clean scraped assessment data."""
    
    def __init__(self, input_file: str, output_file: str):
        """
        Initialize data processor.
        
        Args:
            input_file: Path to raw scraped data
            output_file: Path to save processed data
        """
        self.input_file = input_file
        self.output_file = output_file
        self.assessments = []
        
    def load_data(self) -> List[Dict]:
        """Load raw scraped data."""
        logger.info(f"Loading data from: {self.input_file}")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} assessments")
        return data
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?-]', '', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def validate_assessment(self, assessment: Dict) -> bool:
        """
        Validate assessment has required fields.
        
        Args:
            assessment: Assessment dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['assessment_name', 'assessment_url', 'description', 'test_type']
        
        for field in required_fields:
            if not assessment.get(field):
                logger.warning(f"Missing required field '{field}' in: {assessment.get('assessment_url', 'UNKNOWN')}")
                return False
        
        # Validate URL format
        url = assessment['assessment_url']
        if not url.startswith('https://'):
            logger.warning(f"Invalid URL format: {url}")
            return False
        
        return True
    
    def deduplicate(self, assessments: List[Dict]) -> List[Dict]:
        """
        Remove duplicate assessments based on URL.
        
        Args:
            assessments: List of assessments
            
        Returns:
            Deduplicated list
        """
        seen_urls = set()
        unique_assessments = []
        
        for assessment in assessments:
            url = assessment['assessment_url']
            if url not in seen_urls:
                seen_urls.add(url)
                unique_assessments.append(assessment)
            else:
                logger.debug(f"Removing duplicate: {url}")
        
        logger.info(f"Removed {len(assessments) - len(unique_assessments)} duplicates")
        return unique_assessments
    
    def process(self) -> List[Dict]:
        """
        Main processing workflow.
        
        Returns:
            Processed assessments
        """
        logger.info("=" * 60)
        logger.info("DATA PROCESSING STARTED")
        logger.info("=" * 60)
        
        # Load data
        raw_data = self.load_data()
        
        # Process each assessment
        processed_assessments = []
        invalid_count = 0
        
        for assessment in raw_data:
            # Clean fields
            assessment['assessment_name'] = self.clean_text(assessment.get('assessment_name', ''))
            assessment['description'] = self.clean_text(assessment.get('description', ''))
            assessment['category'] = self.clean_text(assessment.get('category', ''))
            assessment['skills'] = self.clean_text(assessment.get('skills', ''))
            assessment['job_roles'] = self.clean_text(assessment.get('job_roles', ''))
            
            # Validate
            if self.validate_assessment(assessment):
                processed_assessments.append(assessment)
            else:
                invalid_count += 1
        
        logger.info(f"Valid assessments: {len(processed_assessments)}")
        logger.info(f"Invalid assessments: {invalid_count}")
        
        # Deduplicate
        processed_assessments = self.deduplicate(processed_assessments)
        
        # Save processed data
        self.save_data(processed_assessments)
        
        logger.info("=" * 60)
        logger.info(f"DATA PROCESSING COMPLETE: {len(processed_assessments)} assessments")
        logger.info("=" * 60)
        
        return processed_assessments
    
    def save_data(self, assessments: List[Dict]) -> None:
        """Save processed data to file."""
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(assessments, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed data to: {self.output_file}")


def main():
    """Run data processing."""
    processor = DataProcessor(
        input_file="data/shl_catalog.json",
        output_file="data/processed/shl_assessments_clean.json"
    )
    processor.process()


if __name__ == "__main__":
    main()
