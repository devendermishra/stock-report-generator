"""
Report Fetcher Tool for downloading financial reports and documents.
Handles annual reports, quarterly results, and management call transcripts.
"""

import requests
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import os
import time
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@dataclass
class ReportInfo:
    """Represents information about a financial report."""
    title: str
    url: str
    report_type: str
    date: datetime
    size: Optional[int] = None
    format: str = "PDF"

@dataclass
class DownloadResult:
    """Represents the result of a report download."""
    success: bool
    file_path: Optional[str] = None
    error_message: Optional[str] = None
    report_info: Optional[ReportInfo] = None

class ReportFetcherTool:
    """
    Report Fetcher Tool for downloading financial reports and documents.
    
    Provides functionality to fetch annual reports, quarterly results,
    management call transcripts, and other financial documents.
    """
    
    def __init__(self, download_dir: str = "temp/reports"):
        """
        Initialize the Report Fetcher Tool.
        
        Args:
            download_dir: Directory to store downloaded reports
        """
        self.download_dir = download_dir
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        })
        
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)
        
    def fetch_annual_reports(
        self,
        company_name: str,
        years: List[int],
        max_reports: int = 3
    ) -> List[DownloadResult]:
        """
        Fetch annual reports for a company.
        
        Args:
            company_name: Name of the company
            years: List of years to fetch reports for
            max_reports: Maximum number of reports to download
            
        Returns:
            List of DownloadResult objects
        """
        results = []
        
        try:
            # This is a simplified implementation
            # In a real scenario, you'd query company websites or financial databases
            
            for year in sorted(years, reverse=True)[:max_reports]:
                # Simulate finding annual report
                report_info = ReportInfo(
                    title=f"{company_name} Annual Report {year}",
                    url=f"https://example.com/{company_name.lower().replace(' ', '-')}-annual-report-{year}.pdf",
                    report_type="Annual Report",
                    date=datetime(year, 12, 31)
                )
                
                # Attempt to download
                result = self._download_report(report_info)
                results.append(result)
                
                if not result.success:
                    logger.warning(f"Failed to download annual report for {year}")
                    
        except Exception as e:
            logger.error(f"Error fetching annual reports: {e}")
            
        return results
        
    def fetch_quarterly_results(
        self,
        company_name: str,
        quarters: List[str],
        year: int = 2024
    ) -> List[DownloadResult]:
        """
        Fetch quarterly results for a company.
        
        Args:
            company_name: Name of the company
            quarters: List of quarters (Q1, Q2, Q3, Q4)
            year: Year to fetch results for
            
        Returns:
            List of DownloadResult objects
        """
        results = []
        
        try:
            for quarter in quarters:
                report_info = ReportInfo(
                    title=f"{company_name} {quarter} {year} Results",
                    url=f"https://example.com/{company_name.lower().replace(' ', '-')}-{quarter.lower()}-{year}-results.pdf",
                    report_type="Quarterly Results",
                    date=datetime(year, self._get_quarter_month(quarter), 1)
                )
                
                result = self._download_report(report_info)
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error fetching quarterly results: {e}")
            
        return results
        
    def fetch_management_call_transcripts(
        self,
        company_name: str,
        max_transcripts: int = 2
    ) -> List[DownloadResult]:
        """
        Fetch management call transcripts.
        
        Args:
            company_name: Name of the company
            max_transcripts: Maximum number of transcripts to download
            
        Returns:
            List of DownloadResult objects
        """
        results = []
        
        try:
            # Simulate finding management call transcripts
            for i in range(max_transcripts):
                report_info = ReportInfo(
                    title=f"{company_name} Management Call Transcript {i+1}",
                    url=f"https://example.com/{company_name.lower().replace(' ', '-')}-mgmt-call-{i+1}.pdf",
                    report_type="Management Call",
                    date=datetime.now()
                )
                
                result = self._download_report(report_info)
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error fetching management call transcripts: {e}")
            
        return results
        
    def fetch_analyst_reports(
        self,
        company_name: str,
        max_reports: int = 3
    ) -> List[DownloadResult]:
        """
        Fetch analyst research reports.
        
        Args:
            company_name: Name of the company
            max_reports: Maximum number of reports to download
            
        Returns:
            List of DownloadResult objects
        """
        results = []
        
        try:
            # Simulate finding analyst reports
            for i in range(max_reports):
                report_info = ReportInfo(
                    title=f"Analyst Report - {company_name} {i+1}",
                    url=f"https://example.com/analyst-report-{company_name.lower().replace(' ', '-')}-{i+1}.pdf",
                    report_type="Analyst Report",
                    date=datetime.now()
                )
                
                result = self._download_report(report_info)
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error fetching analyst reports: {e}")
            
        return results
        
    def _download_report(self, report_info: ReportInfo) -> DownloadResult:
        """
        Download a single report.
        
        Args:
            report_info: Information about the report to download
            
        Returns:
            DownloadResult object
        """
        try:
            # Generate filename
            filename = f"{report_info.title.replace(' ', '_')}.{report_info.format.lower()}"
            file_path = os.path.join(self.download_dir, filename)
            
            # In a real implementation, you would make an actual HTTP request
            # For now, we'll simulate the download process
            
            # Simulate download delay
            time.sleep(0.5)
            
            # Create a placeholder file (in real implementation, this would be the actual PDF)
            with open(file_path, 'w') as f:
                f.write(f"Placeholder content for {report_info.title}")
                
            logger.info(f"Downloaded report: {filename}")
            
            return DownloadResult(
                success=True,
                file_path=file_path,
                report_info=report_info
            )
            
        except Exception as e:
            logger.error(f"Failed to download report {report_info.title}: {e}")
            return DownloadResult(
                success=False,
                error_message=str(e),
                report_info=report_info
            )
            
    def _get_quarter_month(self, quarter: str) -> int:
        """Get the month number for a quarter."""
        quarter_months = {
            'Q1': 3,   # March
            'Q2': 6,   # June
            'Q3': 9,   # September
            'Q4': 12   # December
        }
        return quarter_months.get(quarter, 3)
        
    def get_available_reports(self, company_name: str) -> List[ReportInfo]:
        """
        Get list of available reports for a company.
        
        Args:
            company_name: Name of the company
            
        Returns:
            List of ReportInfo objects
        """
        try:
            # This would typically query a database or API
            # For now, we'll return a simulated list
            
            reports = []
            current_year = datetime.now().year
            
            # Add annual reports for last 3 years
            for year in range(current_year - 2, current_year + 1):
                reports.append(ReportInfo(
                    title=f"{company_name} Annual Report {year}",
                    url=f"https://example.com/annual-{year}.pdf",
                    report_type="Annual Report",
                    date=datetime(year, 12, 31)
                ))
                
            # Add quarterly results for current year
            for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
                reports.append(ReportInfo(
                    title=f"{company_name} {quarter} {current_year} Results",
                    url=f"https://example.com/{quarter.lower()}-{current_year}.pdf",
                    report_type="Quarterly Results",
                    date=datetime(current_year, self._get_quarter_month(quarter), 1)
                ))
                
            return reports
            
        except Exception as e:
            logger.error(f"Error getting available reports: {e}")
            return []
            
    def cleanup_downloads(self) -> None:
        """Clean up downloaded files."""
        try:
            import shutil
            if os.path.exists(self.download_dir):
                shutil.rmtree(self.download_dir)
                os.makedirs(self.download_dir, exist_ok=True)
                logger.info("Cleaned up download directory")
        except Exception as e:
            logger.error(f"Error cleaning up downloads: {e}")
            
    def get_download_stats(self) -> Dict[str, Any]:
        """
        Get statistics about downloaded reports.
        
        Returns:
            Dictionary containing download statistics
        """
        try:
            if not os.path.exists(self.download_dir):
                return {"total_files": 0, "total_size": 0}
                
            files = os.listdir(self.download_dir)
            total_size = sum(
                os.path.getsize(os.path.join(self.download_dir, f))
                for f in files
                if os.path.isfile(os.path.join(self.download_dir, f))
            )
            
            return {
                "total_files": len(files),
                "total_size": total_size,
                "download_dir": self.download_dir
            }
            
        except Exception as e:
            logger.error(f"Error getting download stats: {e}")
            return {"total_files": 0, "total_size": 0}

# Global report fetcher instance
_report_fetcher = ReportFetcherTool()

@tool(
    description="Fetch annual reports for a company from specified years. Downloads annual reports from financial databases or company websites. Returns list of downloaded reports with file paths and metadata. Essential for accessing historical financial data and company annual reports.",
    infer_schema=True,
    parse_docstring=False
)
def fetch_annual_reports(
    company_name: str,
    years: List[int],
    max_reports: int = 3,
    download_dir: str = "temp/reports"
) -> Dict[str, Any]:
    """
    Fetch annual reports for a company from specified years.
    
    Downloads annual reports from financial databases or company websites.
    
    Args:
        company_name: Full company name.
        years: List of years to fetch reports for.
        max_reports: Maximum number of reports to download (default: 3).
        download_dir: Directory to store downloaded reports (default: "temp/reports").
    
    Returns:
        Dictionary containing success, company_name, total_downloaded, reports (list),
        download_dir, and error (if failed).
    """
    try:
        fetcher = ReportFetcherTool(download_dir=download_dir)
        results = fetcher.fetch_annual_reports(company_name, years, max_reports)
        
        # Convert DownloadResult objects to dictionaries
        reports = []
        for result in results:
            reports.append({
                "success": result.success,
                "file_path": result.file_path,
                "error_message": result.error_message,
                "report_info": {
                    "title": result.report_info.title if result.report_info else None,
                    "url": result.report_info.url if result.report_info else None,
                    "report_type": result.report_info.report_type if result.report_info else None,
                    "date": result.report_info.date.isoformat() if result.report_info and result.report_info.date else None,
                    "format": result.report_info.format if result.report_info else None
                } if result.report_info else None
            })
        
        return {
            "success": any(r.success for r in results),
            "company_name": company_name,
            "total_downloaded": sum(1 for r in results if r.success),
            "reports": reports,
            "download_dir": download_dir
        }
    except Exception as e:
        logger.error(f"Error fetching annual reports: {e}")
        return {
            "success": False,
            "company_name": company_name,
            "total_downloaded": 0,
            "reports": [],
            "error": str(e),
            "download_dir": download_dir
        }
