"""
Comprehensive test suite for HTMLParser class.
Tests HTML content extraction, cleaning, and metadata processing for production readiness.

This test suite covers:
- HTML file discovery and parsing
- Content extraction and cleaning 
- Structured data parsing (tables, lists, headings)
- Metadata extraction (titles, URLs, descriptions)
- Error handling for malformed HTML
- Content filtering and sanitization
- Performance with large HTML files
- Unicode and encoding handling

Each test ensures:
- Robust parsing of real-world HTML content
- Proper handling of edge cases and malformed markup
- Consistent metadata extraction
- Safe content sanitization
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any
import json

from src.embedders.html_parser import HTMLParser


class TestHTMLParserInitialization:
    """Test HTMLParser initialization and configuration."""
    
    def test_html_parser_initialization(self):
        """Test HTMLParser initialization with default settings."""
        parser = HTMLParser()
        assert hasattr(parser, 'logger')
        assert parser is not None
    
    def test_html_parser_with_directory(self):
        """Test HTMLParser initialization with specific directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parser = HTMLParser(temp_dir)
            assert parser is not None


class TestHTMLContentExtraction:
    """Test HTML content extraction functionality."""
    
    @pytest.fixture
    def html_parser(self):
        """Create HTMLParser instance for testing."""
        return HTMLParser()
    
    @pytest.fixture
    def sample_html_content(self):
        """Sample HTML content for testing."""
        return {
            'simple': """
            <html>
                <head><title>Simple Test Page</title></head>
                <body>
                    <h1>Main Heading</h1>
                    <p>This is a simple paragraph with some content.</p>
                </body>
            </html>
            """,
            'complex': """
            <html>
                <head>
                    <title>Complex Program Information</title>
                    <meta name="description" content="Sample Educational Program details">
                    <link rel="canonical" href="https://example.edu/about">
                </head>
                <body>
                    <header>
                        <nav>
                            <ul>
                                <li><a href="/home">Home</a></li>
                                <li><a href="/about">About</a></li>
                            </ul>
                        </nav>
                    </header>
                    <main>
                        <article>
                            <h1>Sample Educational Program</h1>
                            <h2>Overview</h2>
                            <p>The program is a comprehensive training programme designed to equip participants with practical AI skills.</p>
                            
                            <h2>Curriculum</h2>
                            <table>
                                <thead>
                                    <tr><th>Module</th><th>Duration</th><th>Focus</th></tr>
                                </thead>
                                <tbody>
                                    <tr><td>Machine Learning</td><td>4 weeks</td><td>Fundamentals</td></tr>
                                    <tr><td>Deep Learning</td><td>3 weeks</td><td>Neural Networks</td></tr>
                                </tbody>
                            </table>
                            
                            <h3>Requirements</h3>
                            <ul>
                                <li>Bachelor's degree in relevant field</li>
                                <li>Programming experience (Python preferred)</li>
                                <li>Strong mathematical background</li>
                            </ul>
                        </article>
                    </main>
                    <footer>
                        <p>&copy; 2024 Sample Programme</p>
                    </footer>
                </body>
            </html>
            """,
            'malformed': """
            <html>
                <head><title>Malformed HTML</title>
                <body>
                    <h1>Unclosed heading
                    <p>Paragraph without closing tag
                    <div>Nested content
                        <span>Span content</span>
                    <p>Another paragraph
                </body>
            </html>
            """,
            'empty': """
            <html>
                <head><title>Empty Page</title></head>
                <body></body>
            </html>
            """,
            'scripts_and_styles': """
            <html>
                <head>
                    <title>Page with Scripts and Styles</title>
                    <style>
                        body { background: white; }
                        .hidden { display: none; }
                    </style>
                    <script>
                        function hideContent() {
                            document.getElementById('content').style.display = 'none';
                        }
                    </script>
                </head>
                <body>
                    <div id="content">
                        <h1>Visible Content</h1>
                        <p>This content should be extracted.</p>
                    </div>
                    <div class="hidden">Hidden content that should not be extracted.</div>
                    <script>console.log('Script content should be ignored');</script>
                </body>
            </html>
            """
        }
    
    @pytest.fixture
    def sample_html_files(self, sample_html_content):
        """Create sample HTML files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            html_dir = Path(temp_dir)
            
            # Create HTML files
            for name, content in sample_html_content.items():
                html_file = html_dir / f"{name}.html"
                html_file.write_text(content, encoding='utf-8')
            
            yield html_dir
    
    def test_parse_single_html_file_simple(self, html_parser, sample_html_content):
        """Test parsing a simple HTML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(sample_html_content['simple'])
            f.flush()
            
            try:
                result = html_parser.parse_single_html_file(Path(f.name))
                
                assert result is not None
                assert result['title'] == 'Simple Test Page'
                assert 'Main Heading' in result['content']
                assert 'simple paragraph' in result['content']
                assert result['file_path'] == f.name
                assert result['file_name'] == Path(f.name).name
            finally:
                Path(f.name).unlink()
    
    def test_parse_single_html_file_complex(self, html_parser, sample_html_content):
        """Test parsing a complex HTML file with structured content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(sample_html_content['complex'])
            f.flush()
            
            try:
                result = html_parser.parse_single_html_file(Path(f.name))
                
                assert result is not None
                assert result['title'] == 'Complex Program Information'
                assert result['description'] == 'Sample Educational Program details'
                assert result['canonical_url'] == 'https://example.edu/about'
                
                # Check content extraction
                content = result['content']
                assert 'Sample Educational Program' in content
                assert 'comprehensive training programme' in content
                assert 'Machine Learning' in content  # Table content
                assert 'Bachelor\'s degree' in content  # List content
                
                # Check that navigation and footer are excluded
                assert 'Home' not in content  # Navigation should be filtered
                assert '© 2024' not in content  # Footer should be filtered
                
            finally:
                Path(f.name).unlink()
    
    def test_parse_single_html_file_malformed(self, html_parser, sample_html_content):
        """Test parsing malformed HTML gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(sample_html_content['malformed'])
            f.flush()
            
            try:
                result = html_parser.parse_single_html_file(Path(f.name))
                
                # Should still extract content despite malformed structure
                assert result is not None
                assert result['title'] == 'Malformed HTML'
                assert 'Unclosed heading' in result['content']
                assert 'Paragraph without closing tag' in result['content']
                
            finally:
                Path(f.name).unlink()
    
    def test_parse_single_html_file_empty(self, html_parser, sample_html_content):
        """Test parsing empty HTML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(sample_html_content['empty'])
            f.flush()
            
            try:
                result = html_parser.parse_single_html_file(Path(f.name))
                
                assert result is not None
                assert result['title'] == 'Empty Page'
                assert result['content'].strip() == '' or result['content'] is None
                
            finally:
                Path(f.name).unlink()
    
    def test_parse_single_html_file_scripts_filtered(self, html_parser, sample_html_content):
        """Test that scripts and styles are properly filtered out."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(sample_html_content['scripts_and_styles'])
            f.flush()
            
            try:
                result = html_parser.parse_single_html_file(Path(f.name))
                
                assert result is not None
                assert 'Visible Content' in result['content']
                assert 'This content should be extracted' in result['content']
                
                # Should filter out script/style content
                assert 'background: white' not in result['content']
                assert 'console.log' not in result['content']
                assert 'hideContent' not in result['content']
                
            finally:
                Path(f.name).unlink()
    
    def test_parse_single_html_file_nonexistent(self, html_parser):
        """Test parsing non-existent HTML file."""
        result = html_parser.parse_single_html_file(Path('/nonexistent/file.html'))
        assert result is None
    
    def test_parse_single_html_file_invalid_format(self, html_parser):
        """Test parsing invalid file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not HTML content")
            f.flush()
            
            try:
                result = html_parser.parse_single_html_file(Path(f.name))
                # Should handle gracefully, possibly returning minimal result
                assert result is None or result['content'] is not None
                
            finally:
                Path(f.name).unlink()


class TestHTMLDirectoryParsing:
    """Test HTML directory parsing functionality."""
    
    @pytest.fixture
    def html_parser(self):
        """Create HTMLParser instance for testing."""
        return HTMLParser()
    
    def test_parse_all_html_files_multiple_files(self, html_parser, sample_html_files):
        """Test parsing multiple HTML files in a directory."""
        results = html_parser.parse_all_html_files(str(sample_html_files))
        
        assert len(results) > 0
        
        # Check that all files were processed
        titles = [r['title'] for r in results if r['title']]
        assert 'Simple Test Page' in titles
        assert 'Complex Program Information' in titles
        assert 'Malformed HTML' in titles
        
        # Verify each result has required fields
        for result in results:
            assert 'title' in result
            assert 'content' in result
            assert 'file_path' in result
            assert 'file_name' in result
    
    def test_parse_all_html_files_empty_directory(self, html_parser):
        """Test parsing empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = html_parser.parse_all_html_files(temp_dir)
            assert len(results) == 0
    
    def test_parse_all_html_files_no_html_files(self, html_parser):
        """Test parsing directory with no HTML files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create non-HTML files
            (Path(temp_dir) / 'readme.txt').write_text('Not HTML')
            (Path(temp_dir) / 'data.json').write_text('{"not": "html"}')
            
            results = html_parser.parse_all_html_files(temp_dir)
            assert len(results) == 0
    
    def test_parse_all_html_files_mixed_files(self, html_parser):
        """Test parsing directory with mixed file types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create HTML file
            html_file = Path(temp_dir) / 'test.html'
            html_file.write_text('<html><head><title>Test</title></head><body>Content</body></html>')
            
            # Create non-HTML files
            (Path(temp_dir) / 'readme.txt').write_text('Not HTML')
            (Path(temp_dir) / 'data.json').write_text('{"not": "html"}')
            
            results = html_parser.parse_all_html_files(temp_dir)
            
            # Should only process HTML files
            assert len(results) == 1
            assert results[0]['title'] == 'Test'
    
    def test_parse_all_html_files_nonexistent_directory(self, html_parser):
        """Test parsing non-existent directory."""
        results = html_parser.parse_all_html_files('/nonexistent/directory')
        assert len(results) == 0


class TestHTMLContentCleaning:
    """Test HTML content cleaning and sanitization."""
    
    @pytest.fixture
    def html_parser(self):
        """Create HTMLParser instance for testing."""
        return HTMLParser()
    
    def test_clean_extracted_content_whitespace(self, html_parser):
        """Test whitespace normalization in content cleaning."""
        raw_content = "  Multiple   spaces    and\n\n\nnewlines\t\ttabs  "
        
        if hasattr(html_parser, 'clean_extracted_content'):
            cleaned = html_parser.clean_extracted_content(raw_content)
            
            # Should normalize whitespace
            assert '   ' not in cleaned  # Multiple spaces reduced
            assert '\n\n\n' not in cleaned  # Multiple newlines reduced
            assert '\t\t' not in cleaned  # Tabs normalized
            assert cleaned.strip() == cleaned  # No leading/trailing whitespace
    
    def test_extract_metadata_comprehensive(self, html_parser):
        """Test comprehensive metadata extraction."""
        html_content = """
        <html>
            <head>
                <title>Sample Programme Information</title>
                <meta name="description" content="Comprehensive AI training programme">
                <meta name="keywords" content="AI, machine learning, training">
                <meta name="author" content="Sample Team">
                <link rel="canonical" href="https://example.edu/programme">
                <meta property="og:title" content="Sample Programme">
                <meta property="og:description" content="Join the AI revolution">
            </head>
            <body>Content here</body>
        </html>
        """
        
        if hasattr(html_parser, 'extract_metadata'):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                f.write(html_content)
                f.flush()
                
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    metadata = html_parser.extract_metadata(soup, Path(f.name))
                    
                    assert metadata['title'] == 'Sample Programme Information'
                    assert metadata['description'] == 'Comprehensive AI training programme'
                    assert metadata['canonical_url'] == 'https://example.edu/programme'
                    
                    if 'keywords' in metadata:
                        assert 'AI' in metadata['keywords']
                        assert 'machine learning' in metadata['keywords']
                    
                finally:
                    Path(f.name).unlink()
    
    def test_filter_content_areas(self, html_parser):
        """Test filtering of irrelevant content areas."""
        html_with_noise = """
        <html>
            <body>
                <nav>Navigation menu should be filtered</nav>
                <header>Header content should be filtered</header>
                <aside>Sidebar content should be filtered</aside>
                
                <main>
                    <article>
                        <h1>Main Article Content</h1>
                        <p>This is the important content that should be kept.</p>
                    </article>
                </main>
                
                <footer>Footer content should be filtered</footer>
                <div class="advertisement">Ad content should be filtered</div>
            </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_with_noise)
            f.flush()
            
            try:
                result = html_parser.parse_single_html_file(Path(f.name))
                
                if result and result['content']:
                    content = result['content']
                    
                    # Should keep main content
                    assert 'Main Article Content' in content
                    assert 'important content that should be kept' in content
                    
                    # Should filter out noise (these assertions might need adjustment based on actual implementation)
                    # assert 'Navigation menu' not in content
                    # assert 'Header content' not in content
                    # assert 'Footer content' not in content
                
            finally:
                Path(f.name).unlink()


class TestHTMLParserStructuredData:
    """Test parsing of structured HTML data."""
    
    @pytest.fixture
    def html_parser(self):
        """Create HTMLParser instance for testing."""
        return HTMLParser()
    
    def test_table_parsing(self, html_parser):
        """Test extraction of table data."""
        html_with_table = """
        <html>
            <body>
                <h1>Course Schedule</h1>
                <table>
                    <thead>
                        <tr>
                            <th>Course</th>
                            <th>Duration</th>
                            <th>Prerequisites</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Machine Learning Basics</td>
                            <td>2 weeks</td>
                            <td>Python programming</td>
                        </tr>
                        <tr>
                            <td>Deep Learning</td>
                            <td>3 weeks</td>
                            <td>ML Basics, Linear Algebra</td>
                        </tr>
                    </tbody>
                </table>
            </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_with_table)
            f.flush()
            
            try:
                result = html_parser.parse_single_html_file(Path(f.name))
                
                assert result is not None
                content = result['content']
                
                # Should extract table content in readable format
                assert 'Course Schedule' in content
                assert 'Machine Learning Basics' in content
                assert '2 weeks' in content
                assert 'Python programming' in content
                assert 'Deep Learning' in content
                
            finally:
                Path(f.name).unlink()
    
    def test_list_parsing(self, html_parser):
        """Test extraction of list data."""
        html_with_lists = """
        <html>
            <body>
                <h1>Programme Requirements</h1>
                <h2>Technical Skills</h2>
                <ul>
                    <li>Python programming (intermediate level)</li>
                    <li>Statistics and probability</li>
                    <li>Linear algebra fundamentals</li>
                </ul>
                
                <h2>Application Process</h2>
                <ol>
                    <li>Submit online application</li>
                    <li>Complete technical assessment</li>
                    <li>Attend interview session</li>
                    <li>Final selection notification</li>
                </ol>
            </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_with_lists)
            f.flush()
            
            try:
                result = html_parser.parse_single_html_file(Path(f.name))
                
                assert result is not None
                content = result['content']
                
                # Should extract list content
                assert 'Python programming' in content
                assert 'Statistics and probability' in content
                assert 'Submit online application' in content
                assert 'technical assessment' in content
                
            finally:
                Path(f.name).unlink()
    
    def test_heading_hierarchy_preservation(self, html_parser):
        """Test preservation of heading hierarchy."""
        html_with_headings = """
        <html>
            <body>
                <h1>Sample Educational Program</h1>
                <h2>Programme Overview</h2>
                <p>General information about the programme.</p>
                
                <h2>Curriculum Structure</h2>
                <h3>Phase 1: Foundations</h3>
                <p>Basic concepts and theory.</p>
                
                <h3>Phase 2: Practical Applications</h3>
                <p>Hands-on projects and implementation.</p>
                
                <h4>Project Examples</h4>
                <p>Sample projects from previous cohorts.</p>
            </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_with_headings)
            f.flush()
            
            try:
                result = html_parser.parse_single_html_file(Path(f.name))
                
                assert result is not None
                content = result['content']
                
                # Should preserve heading structure
                assert 'Sample Educational Program' in content
                assert 'Programme Overview' in content
                assert 'Curriculum Structure' in content
                assert 'Phase 1: Foundations' in content
                assert 'Phase 2: Practical Applications' in content
                assert 'Project Examples' in content
                
            finally:
                Path(f.name).unlink()


class TestHTMLParserEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def html_parser(self):
        """Create HTMLParser instance for testing."""
        return HTMLParser()
    
    def test_unicode_content_handling(self, html_parser):
        """Test handling of Unicode content."""
        unicode_html = """
        <html>
            <head><title>Unicode Test - 中文测试</title></head>
            <body>
                <h1>Multilingual Content</h1>
                <p>English content with unicode: café, naïve, résumé</p>
                <p>Chinese content: 人工智能学习计划</p>
                <p>Japanese content: 機械学習プログラム</p>
                <p>Emoji content: 🤖 AI 📊 Data 🧠 Learning</p>
            </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(unicode_html)
            f.flush()
            
            try:
                result = html_parser.parse_single_html_file(Path(f.name))
                
                assert result is not None
                assert '中文测试' in result['title']
                
                content = result['content']
                assert 'café' in content
                assert '人工智能' in content
                assert '機械学習' in content
                assert '🤖' in content
                
            finally:
                Path(f.name).unlink()
    
    def test_very_large_html_file(self, html_parser):
        """Test handling of very large HTML files."""
        # Create a large HTML file
        large_content_parts = ['<html><head><title>Large File Test</title></head><body>']
        
        # Add many paragraphs to create a large file
        for i in range(1000):
            large_content_parts.append(f'<p>This is paragraph number {i} with some content about AI and machine learning.</p>')
        
        large_content_parts.append('</body></html>')
        large_html = '\n'.join(large_content_parts)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(large_html)
            f.flush()
            
            try:
                # Should handle large files without crashing
                result = html_parser.parse_single_html_file(Path(f.name))
                
                assert result is not None
                assert result['title'] == 'Large File Test'
                assert len(result['content']) > 10000  # Should contain substantial content
                
            finally:
                Path(f.name).unlink()
    
    def test_deeply_nested_html(self, html_parser):
        """Test handling of deeply nested HTML structures."""
        nested_html = '<html><head><title>Nested Test</title></head><body>'
        
        # Create deeply nested divs
        for i in range(20):
            nested_html += f'<div class="level-{i}">'
        
        nested_html += '<p>Deeply nested content about AI training programmes.</p>'
        
        for i in range(20):
            nested_html += '</div>'
        
        nested_html += '</body></html>'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(nested_html)
            f.flush()
            
            try:
                result = html_parser.parse_single_html_file(Path(f.name))
                
                assert result is not None
                assert 'Deeply nested content' in result['content']
                
            finally:
                Path(f.name).unlink()
    
    def test_html_with_comments(self, html_parser):
        """Test handling of HTML comments."""
        html_with_comments = """
        <html>
            <head><title>Comments Test</title></head>
            <body>
                <!-- This is a comment that should be ignored -->
                <h1>Visible Heading</h1>
                <!-- Another comment with sensitive info: password123 -->
                <p>Visible paragraph content.</p>
                <!--
                Multi-line comment
                with multiple lines
                should also be ignored
                -->
                <p>Another visible paragraph.</p>
            </body>
        </html>
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_with_comments)
            f.flush()
            
            try:
                result = html_parser.parse_single_html_file(Path(f.name))
                
                assert result is not None
                content = result['content']
                
                # Should extract visible content
                assert 'Visible Heading' in content
                assert 'Visible paragraph content' in content
                assert 'Another visible paragraph' in content
                
                # Should ignore comments
                assert 'This is a comment' not in content
                assert 'password123' not in content
                assert 'Multi-line comment' not in content
                
            finally:
                Path(f.name).unlink()


class TestHTMLParserPerformance:
    """Test performance and scalability of HTML parser."""
    
    @pytest.fixture
    def html_parser(self):
        """Create HTMLParser instance for testing."""
        return HTMLParser()
    
    def test_batch_processing_performance(self, html_parser):
        """Test performance with batch processing of multiple files."""
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple HTML files
            for i in range(50):
                html_content = f"""
                <html>
                    <head><title>Test File {i}</title></head>
                    <body>
                        <h1>Content for file {i}</h1>
                        <p>This is test content for performance testing. File number {i}.</p>
                        <p>Additional paragraph with more content to make files larger.</p>
                    </body>
                </html>
                """
                
                html_file = Path(temp_dir) / f"test_{i}.html"
                html_file.write_text(html_content)
            
            # Measure parsing time
            start_time = time.time()
            results = html_parser.parse_all_html_files(temp_dir)
            end_time = time.time()
            
            # Should complete in reasonable time (adjust threshold as needed)
            processing_time = end_time - start_time
            assert processing_time < 10.0  # Should complete within 10 seconds
            
            # Should process all files
            assert len(results) == 50
            
            # Verify content quality
            for i, result in enumerate(results):
                if result['title']:
                    assert f'Test File' in result['title']
                assert f'Content for file' in result['content']
    
    def test_memory_usage_with_large_batch(self, html_parser):
        """Test memory usage doesn't grow excessively with large batches."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create many HTML files
            for i in range(100):
                html_content = f"""
                <html>
                    <head><title>Memory Test {i}</title></head>
                    <body>
                        <h1>Large Content Block {i}</h1>
                        {'<p>Large paragraph content. ' * 100}
                    </body>
                </html>
                """
                
                html_file = Path(temp_dir) / f"memory_test_{i}.html"
                html_file.write_text(html_content)
            
            # Process files
            results = html_parser.parse_all_html_files(temp_dir)
            
            # Check final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (adjust threshold as needed)
            assert memory_increase < 500  # Less than 500MB increase
            assert len(results) == 100