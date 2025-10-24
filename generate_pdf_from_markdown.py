#!/usr/bin/env python3
"""
Utility script to generate PDF from existing markdown reports.
This script can be used to convert any markdown report to PDF format.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tools.pdf_generator_tool import PDFGeneratorTool

def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF from markdown stock reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_pdf_from_markdown.py reports/stock_report_ICICIBANK_20251021_200913.md
  python generate_pdf_from_markdown.py --batch reports/
  python generate_pdf_from_markdown.py --output-dir my_pdfs reports/stock_report_ICICIBANK_20251021_200913.md
        """
    )
    
    parser.add_argument(
        "input",
        nargs='?',
        help="Markdown file to convert to PDF"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Convert all .md files in the specified directory"
    )
    
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for PDF files (default: reports)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if not args.input and not args.batch:
        parser.error("Please specify a markdown file or use --batch for directory processing")
    
    # Initialize PDF generator
    pdf_generator = PDFGeneratorTool(output_dir=args.output_dir)
    
    try:
        if args.batch:
            # Process all markdown files in directory
            input_dir = args.input if args.input else "reports"
            markdown_files = []
            
            for file_path in Path(input_dir).glob("*.md"):
                markdown_files.append(str(file_path))
            
            if not markdown_files:
                print(f"No markdown files found in {input_dir}")
                return
            
            print(f"Found {len(markdown_files)} markdown files to convert...")
            
            results = pdf_generator.batch_generate_pdfs(markdown_files)
            
            print(f"\n📊 Batch conversion results:")
            for markdown_file, pdf_path in results.items():
                if pdf_path:
                    print(f"✅ {markdown_file} → {pdf_path}")
                else:
                    print(f"❌ Failed to convert {markdown_file}")
        
        else:
            # Process single file
            if not os.path.exists(args.input):
                print(f"❌ File not found: {args.input}")
                return
            
            print(f"Converting {args.input} to PDF...")
            pdf_path = pdf_generator.generate_pdf_from_file(args.input)
            
            print(f"✅ PDF generated successfully: {pdf_path}")
            
            # Show PDF info
            pdf_info = pdf_generator.get_pdf_info(pdf_path)
            if pdf_info.get('exists'):
                print(f"📄 File size: {pdf_info['file_size']} bytes")
                print(f"📅 Created: {pdf_info['created_at']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
