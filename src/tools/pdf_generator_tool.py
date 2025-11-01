"""
PDF Generator Tool for converting markdown reports to PDF format.

This tool provides functionality to convert markdown reports to professional PDF format
with proper styling, formatting, and layout.
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import markdown
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

class PDFGeneratorTool:
    """
    PDF Generator Tool for converting markdown reports to PDF format.
    
    Provides functionality to:
    - Convert markdown to HTML
    - Apply professional styling
    - Generate PDF with proper formatting
    - Handle tables, charts, and complex layouts
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the PDF Generator Tool.
        
        Args:
            output_dir: Directory to save generated PDFs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure markdown extensions
        self.markdown_extensions = [
            'markdown.extensions.extra',
            'markdown.extensions.codehilite',
            'markdown.extensions.tables',
            'markdown.extensions.toc',
            'markdown.extensions.fenced_code'
        ]
        
        # Initialize reportlab styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for the PDF report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=HexColor('#2c3e50'),
            alignment=TA_CENTER,
            borderWidth=1,
            borderColor=HexColor('#3498db'),
            borderPadding=10
        ))
        
        # Heading 1 style
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=12,
            spaceBefore=20,
            textColor=HexColor('#34495e'),
            borderWidth=1,
            borderColor=HexColor('#ecf0f1'),
            borderPadding=5
        ))
        
        # Heading 2 style
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=HexColor('#2c3e50')
        ))
        
        # Heading 3 style
        self.styles.add(ParagraphStyle(
            name='CustomHeading3',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=12,
            textColor=HexColor('#34495e')
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        # Executive summary style
        self.styles.add(ParagraphStyle(
            name='ExecutiveSummary',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leftIndent=20,
            rightIndent=20,
            borderWidth=1,
            borderColor=HexColor('#3498db'),
            borderPadding=10,
            backColor=HexColor('#f8f9fa')
        ))
        
        # Financial metrics style
        self.styles.add(ParagraphStyle(
            name='FinancialMetrics',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leftIndent=20,
            rightIndent=20,
            borderWidth=1,
            borderColor=HexColor('#27ae60'),
            borderPadding=10,
            backColor=HexColor('#e8f5e8')
        ))
        
        # Risk section style
        self.styles.add(ParagraphStyle(
            name='RiskSection',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leftIndent=20,
            rightIndent=20,
            borderWidth=1,
            borderColor=HexColor('#e74c3c'),
            borderPadding=10,
            backColor=HexColor('#fdf2e9')
        ))
    
    def _parse_markdown_to_elements(self, markdown_content: str):
        """Parse markdown content and convert to reportlab elements."""
        elements = []
        lines = markdown_content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
                
            # Handle headers
            if line.startswith('# '):
                elements.append(Paragraph(line[2:], self.styles['CustomTitle']))
                elements.append(Spacer(1, 12))
            elif line.startswith('## '):
                elements.append(Paragraph(line[3:], self.styles['CustomHeading1']))
                elements.append(Spacer(1, 6))
            elif line.startswith('### '):
                elements.append(Paragraph(line[4:], self.styles['CustomHeading2']))
                elements.append(Spacer(1, 6))
            elif line.startswith('#### '):
                elements.append(Paragraph(line[5:], self.styles['CustomHeading3']))
                elements.append(Spacer(1, 6))
            
            # Handle bullet points
            elif line.startswith('- ') or line.startswith('* '):
                bullet_text = line[2:]
                # Process any bold text in bullet points
                bullet_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', bullet_text)
                elements.append(Paragraph(f"• {bullet_text}", self.styles['CustomBody']))
            
            # Handle numbered lists
            elif re.match(r'^\d+\. ', line):
                # Process any bold text in numbered lists
                formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                elements.append(Paragraph(formatted_line, self.styles['CustomBody']))
            
            # Handle regular paragraphs with potential bold text
            else:
                if line and not line.startswith('---'):
                    # Process any bold text in regular paragraphs
                    formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                    elements.append(Paragraph(formatted_line, self.styles['CustomBody']))
            
            i += 1
        
        return elements
    
    def markdown_to_html(self, markdown_content: str) -> str:
        """
        Convert markdown content to HTML with extensions.
        
        Args:
            markdown_content: Raw markdown content
            
        Returns:
            HTML content
        """
        try:
            # Convert markdown to HTML
            html = markdown.markdown(
                markdown_content,
                extensions=self.markdown_extensions,
                extension_configs={
                    'markdown.extensions.codehilite': {
                        'css_class': 'highlight'
                    }
                }
            )
            
            # Wrap in HTML document structure
            html_document = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Equity Research Report</title>
            </head>
            <body>
                {html}
                <div class="footer-note">
                    This report was generated using AI-powered multi-agent analysis on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
                </div>
            </body>
            </html>
            """
            
            return html_document
            
        except Exception as e:
            logger.error(f"Error converting markdown to HTML: {e}")
            raise
    
    def generate_pdf(
        self, 
        markdown_content: str, 
        output_filename: Optional[str] = None,
        stock_symbol: Optional[str] = None
    ) -> str:
        """
        Generate PDF from markdown content.
        
        Args:
            markdown_content: Markdown content to convert
            output_filename: Optional custom filename
            stock_symbol: Stock symbol for filename generation
            
        Returns:
            Path to generated PDF file
        """
        try:
            logger.info("Starting PDF generation...")
            
            # Generate filename if not provided
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if stock_symbol:
                    output_filename = f"stock_report_{stock_symbol}_{timestamp}.pdf"
                else:
                    output_filename = f"stock_report_{timestamp}.pdf"
            
            # Ensure .pdf extension
            if not output_filename.endswith('.pdf'):
                output_filename += '.pdf'
            
            # Full path to output file
            pdf_path = os.path.join(self.output_dir, output_filename)
            
            # Parse markdown to reportlab elements
            elements = self._parse_markdown_to_elements(markdown_content)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build PDF
            logger.info(f"Generating PDF: {pdf_path}")
            doc.build(elements)
            
            logger.info(f"PDF generated successfully: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            raise
    
    def generate_pdf_from_file(self, markdown_file_path: str) -> str:
        """
        Generate PDF from an existing markdown file.
        
        Args:
            markdown_file_path: Path to markdown file
            
        Returns:
            Path to generated PDF file
        """
        try:
            # Read markdown file
            with open(markdown_file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Extract stock symbol from filename if possible
            filename = os.path.basename(markdown_file_path)
            stock_symbol = None
            if 'stock_report_' in filename:
                parts = filename.split('_')
                if len(parts) >= 3:
                    stock_symbol = parts[2]
            
            # Generate PDF filename
            pdf_filename = filename.replace('.md', '.pdf')
            
            # Generate PDF
            return self.generate_pdf(
                markdown_content=markdown_content,
                output_filename=pdf_filename,
                stock_symbol=stock_symbol
            )
            
        except Exception as e:
            logger.error(f"Error generating PDF from file {markdown_file_path}: {e}")
            raise
    
    def batch_generate_pdfs(self, markdown_files: list) -> Dict[str, str]:
        """
        Generate PDFs from multiple markdown files.
        
        Args:
            markdown_files: List of markdown file paths
            
        Returns:
            Dictionary mapping markdown files to generated PDF paths
        """
        results = {}
        
        for markdown_file in markdown_files:
            try:
                pdf_path = self.generate_pdf_from_file(markdown_file)
                results[markdown_file] = pdf_path
                logger.info(f"Generated PDF for {markdown_file}: {pdf_path}")
            except Exception as e:
                logger.error(f"Failed to generate PDF for {markdown_file}: {e}")
                results[markdown_file] = None
        
        return results
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Get information about a generated PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with PDF information
        """
        try:
            if not os.path.exists(pdf_path):
                return {"error": "PDF file not found"}
            
            file_stats = os.stat(pdf_path)
            
            return {
                "file_path": pdf_path,
                "file_size": file_stats.st_size,
                "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "exists": True
            }
            
        except Exception as e:
            logger.error(f"Error getting PDF info: {e}")
            return {"error": str(e)}

# Global PDF generator instance
_pdf_generator = PDFGeneratorTool()

@tool(
    description="Convert markdown content to professional PDF format. Generates professionally formatted PDF documents from markdown text with proper styling, headers, sections, and layout. Essential for creating final stock research reports in PDF format.",
    infer_schema=True,
    parse_docstring=False
)
def generate_pdf_from_markdown(
    markdown_content: str,
    output_filename: Optional[str] = None,
    stock_symbol: Optional[str] = None,
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Convert markdown content to professional PDF format.
    
    Generates professionally formatted PDF documents from markdown text with proper styling,
    headers, sections, and layout.
    
    Args:
        markdown_content: Markdown text content to convert to PDF.
        output_filename: Optional custom filename for the PDF.
        stock_symbol: Optional stock symbol to include in auto-generated filename.
        output_dir: Directory where PDF will be saved (default: "reports").
    
    Returns:
        Dictionary containing success, pdf_path, filename, output_dir, and error (if failed).
    """
    try:
        generator = PDFGeneratorTool(output_dir=output_dir)
        pdf_path = generator.generate_pdf(markdown_content, output_filename, stock_symbol)
        
        return {
            "success": True,
            "pdf_path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "output_dir": output_dir
        }
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return {
            "success": False,
            "error": f"PDF generation failed: {str(e)}",
            "pdf_path": None,
            "filename": None,
            "output_dir": output_dir
        }
