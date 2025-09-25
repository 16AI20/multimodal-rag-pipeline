"""HTML parser for extracting structured text from web content.

This module provides functionality to parse HTML files and extract meaningful
text content for vector embedding and search. It handles smart content area
detection, formatting preservation, and canonical URL extraction.
"""

import json
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Optional


class HTMLParser:
    """Parse HTML files and extract text content for ChromaDB embedding."""
    
    def __init__(self, corpus_path: str = "corpus/html") -> None:
        """Initialize HTMLParser with corpus directory path.
        
        Args:
            corpus_path: Path to directory containing HTML files to parse.
        """
        self.corpus_path = Path(corpus_path)
        self.logger = logging.getLogger(__name__)
        
    def extract_text_from_html(self, file_path: Path) -> Optional[Dict[str, str]]:
        """Extract meaningful text content from an HTML file.
        
        Args:
            file_path: Path to the HTML file to process.
            
        Returns:
            Dictionary containing extracted content with keys: file_path, title,
            description, content, url, source. Returns None if processing fails.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
            
            # Remove non-content elements
            for element in soup(["script", "style", "nav", "footer", "aside", "header", 
                               "iframe", "object", "embed"]):
                element.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ""
            
            # Extract main content areas
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.post', '.entry-content', '.article-content'
            ]
            
            main_content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    main_content = self._extract_formatted_text(content_elem)
                    break
            
            # If no main content found, extract from body
            if not main_content:
                body = soup.find('body')
                if body:
                    main_content = self._extract_formatted_text(body)
            
            # Clean up whitespace
            main_content = ' '.join(main_content.split())
            
            # Create source identifier for citations
            url = self._extract_canonical_url(soup)
            source = self._create_source_identifier(url, title_text, file_path)
            
            return {
                'file_path': str(file_path),
                'title': title_text,
                'description': description,
                'content': main_content,
                'url': url,
                'source': source
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _extract_formatted_text(self, element: BeautifulSoup) -> str:
        """Extract text with special handling for headings, tables, lists, and code blocks.
        
        Args:
            element: BeautifulSoup element to extract text from.
            
        Returns:
            Formatted text string with preserved structure.
        """
        text_parts = []
        
        # Create a copy of children to avoid modification during iteration
        children_to_process = list(element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table', 'ul', 'ol', 'pre', 'code']))
        
        for child in children_to_process:
            if not child or not hasattr(child, 'name') or child.name is None:
                continue
                
            if child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Preserve heading structure with markdown-style formatting
                level = int(child.name[1])
                heading_text = child.get_text(strip=True)
                if heading_text:
                    text_parts.append(f"{'#' * level} {heading_text}")
                child.extract()
            elif child.name == 'table':
                # Extract table content with basic structure
                table_text = []
                for row in child.find_all('tr'):
                    cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                    if cells:
                        table_text.append(' | '.join(cells))
                if table_text:
                    text_parts.append('\n'.join(table_text))
                child.extract()
            elif child.name in ['ul', 'ol']:
                # Extract list items with bullets/numbers
                list_items = []
                for li in child.find_all('li'):
                    item_text = li.get_text(strip=True)
                    if item_text:
                        prefix = '• ' if child.name == 'ul' else f"{len(list_items) + 1}. "
                        list_items.append(f"{prefix}{item_text}")
                if list_items:
                    text_parts.append('\n'.join(list_items))
                child.extract()
            elif child.name in ['pre', 'code']:
                # Preserve code blocks with minimal formatting
                code_text = child.get_text(strip=True)
                if code_text:
                    text_parts.append(f"[CODE] {code_text}")
                child.extract()
        
        # Get remaining text
        remaining_text = element.get_text(separator=' ', strip=True)
        if remaining_text:
            text_parts.append(remaining_text)
        
        return ' '.join(text_parts)
    
    def _extract_canonical_url(self, soup: BeautifulSoup) -> str:
        """Extract canonical URL from HTML head.
        
        Args:
            soup: BeautifulSoup object of the HTML document.
            
        Returns:
            Canonical URL string if found, empty string otherwise.
        """
        canonical = soup.find('link', attrs={'rel': 'canonical'})
        return canonical.get('href', '') if canonical else ""
    
    def _create_source_identifier(self, url: str, title: str, file_path: Path) -> str:
        """Create a user-friendly source identifier for citations.
        
        Args:
            url: Canonical URL from the HTML.
            title: Page title from the HTML.
            file_path: Path to the HTML file.
            
        Returns:
            User-friendly source identifier string for citations.
        """
        # Priority 1: Use URL if available
        if url and url.strip():
            return url.strip()
        
        # Priority 2: Use title if available
        if title and title.strip():
            return title.strip()
        
        # Priority 3: Create readable name from file path
        # Convert path like "corpus/html/faqs/what-is-program/index.html" 
        # to "Program FAQs - What is Program"
        path_parts = file_path.parts
        
        if len(path_parts) >= 3:  # corpus/html/something
            # Skip corpus/html, use the rest
            relevant_parts = path_parts[2:]
            
            # Remove index.html if present
            if relevant_parts[-1] == 'index.html':
                relevant_parts = relevant_parts[:-1]
            
            if relevant_parts:
                # Convert kebab-case to title case and join
                readable_parts = []
                for part in relevant_parts:
                    # Convert "what-is-program" to "What Is Program"
                    formatted = part.replace('-', ' ').title()
                    readable_parts.append(formatted)
                
                return ' - '.join(readable_parts)
        
        # Final fallback: just the filename
        return file_path.stem
    
    def parse_all_html_files(self) -> List[Dict[str, str]]:
        """Parse all HTML files in the corpus directory.
        
        Returns:
            List of dictionaries containing extracted content from each HTML file.
        """
        documents = []
        
        for html_file in self.corpus_path.rglob("*.html"):
            if html_file.is_file():
                try:
                    doc = self.extract_text_from_html(html_file)
                    if doc and doc['content'].strip():
                        documents.append(doc)
                        self.logger.info(f"Processed: {html_file}")
                except Exception as e:
                    self.logger.error(f"Error processing {html_file}: {e}")
                    continue
        
        return documents
    
    def save_parsed_data(self, documents: List[Dict[str, str]], output_file: str = "parsed_html_data.json") -> None:
        """Save parsed documents to JSON file.
        
        Args:
            documents: List of parsed document dictionaries to save.
            output_file: Output file path for the JSON data.
        """
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved {len(documents)} documents to {output_path}")


def main() -> None:
    """Main function to parse HTML files for ChromaDB embedding."""
    logging.basicConfig(level=logging.INFO)
    parser = HTMLParser()
    documents = parser.parse_all_html_files()
    
    logger = logging.getLogger(__name__)
    if documents:
        parser.save_parsed_data(documents)
        logger.info(f"Successfully parsed {len(documents)} HTML documents")
    else:
        logger.warning("No documents were parsed successfully")


if __name__ == "__main__":
    main()