#!/usr/bin/env python
"""
Simple Report Generator for RAG system.
Takes questions from JSON file and generates markdown Q&A report.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from ..rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class SimpleReportGenerator:
    """Simple report generator that converts questions to Q&A markdown."""
    
    def __init__(self, config_path: str = "conf"):
        """Initialize the report generator with RAG pipeline."""
        self.rag_pipeline = RAGPipeline(config_path=config_path)
        logger.info("Report generator initialized")

    def load_questions(self, questions_file: str) -> List[str]:
        """
        Load questions from JSON file.
        
        Expected format: {"questions": ["Question 1?", "Question 2?"]}
        
        Args:
            questions_file: Path to JSON file with questions
            
        Returns:
            List of question strings
        """
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'questions' in data:
                questions = data['questions']
            elif isinstance(data, list):
                questions = data
            else:
                raise ValueError("JSON must contain 'questions' key with list of questions, or be a list directly")
            
            if not questions:
                raise ValueError("No questions found in file")
            
            logger.info(f"Loaded {len(questions)} questions from {questions_file}")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to load questions from {questions_file}: {e}")
            raise

    def generate_report(self, questions: List[str], output_file: str) -> str:
        """
        Generate markdown report from questions.
        
        Args:
            questions: List of questions to answer
            output_file: Path to output markdown file
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating report with {len(questions)} questions")
        
        # Prepare markdown content
        md_lines = []
        
        # Header
        md_lines.append("# RAG System Report")
        md_lines.append("")
        md_lines.append(f"*Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*")
        md_lines.append("")
        md_lines.append(f"This report contains {len(questions)} questions answered using the RAG system.")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # Process each question
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}: {question}")
            
            try:
                # Query RAG pipeline
                result = self.rag_pipeline.query(question=question)
                
                answer = result.get('answer', 'No response generated.')
                source_docs = result.get('sources', [])
                
                # Add Q&A to markdown
                md_lines.append(f"## Question {i}")
                md_lines.append("")
                md_lines.append(f"**Q: {question}**")
                md_lines.append("")
                md_lines.append(f"**A:** {answer}")
                md_lines.append("")
                
                # Add sources if available
                if source_docs:
                    md_lines.append("**Sources:**")
                    for j, source_info in enumerate(source_docs[:3], 1):  # Limit to top 3 sources
                        citation = source_info.get('citation', 'Unknown source')
                        md_lines.append(f"{j}. {citation}")
                    md_lines.append("")
                
                md_lines.append("---")
                md_lines.append("")
                
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                md_lines.append(f"**Q: {question}**")
                md_lines.append("")
                md_lines.append(f"**A:** Error generating response: {str(e)}")
                md_lines.append("")
                md_lines.append("---")
                md_lines.append("")
        
        # Write report
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        
        logger.info(f"Report generated successfully: {output_file}")
        return output_file


def main():
    """Command line interface for simple report generation."""
    parser = argparse.ArgumentParser(description="Generate Q&A markdown report from questions")
    
    parser.add_argument('--questions', '-q', required=True,
                       help='JSON file containing questions')
    
    parser.add_argument('--output', '-o',
                       default='reports/report.md',
                       help='Output markdown file (default: reports/report.md)')
    
    parser.add_argument('--config-path',
                       default='conf',
                       help='Path to configuration directory (default: conf)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize generator
        generator = SimpleReportGenerator(config_path=args.config_path)
        
        # Load questions
        questions = generator.load_questions(args.questions)
        
        # Generate report
        output_file = generator.generate_report(questions, args.output)
        
        print(f"✅ Report generated: {output_file}")
        print(f"📊 Processed {len(questions)} questions")
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())