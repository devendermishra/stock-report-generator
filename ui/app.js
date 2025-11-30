// API Configuration
const API_BASE_URL = 'http://localhost:8000';
let currentReport = '';
let currentSymbol = '';

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Allow Enter key to trigger report generation
    const symbolInput = document.getElementById('symbolInput');
    symbolInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            generateReport();
        }
    });
    
    // Configure marked.js options
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true,
            gfm: true,
            highlight: function(code, lang) {
                if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (err) {
                        console.error('Highlight error:', err);
                    }
                }
                return code;
            }
        });
    }
});

/**
 * Generate a stock report by calling the API
 */
async function generateReport() {
    const symbolInput = document.getElementById('symbolInput');
    const symbol = symbolInput.value.trim().toUpperCase();
    
    // Validate input
    if (!symbol) {
        showAlert('Please enter a stock symbol', 'warning');
        return;
    }
    
    // Clean symbol (remove .NS suffix if present)
    const cleanSymbol = symbol.replace(/\.NS$/i, '');
    
    currentSymbol = cleanSymbol;
    
    // Hide previous report and alerts
    hideReport();
    hideAlert();
    
    // Show loading spinner
    showLoading();
    
    // Disable generate button
    const generateBtn = document.getElementById('generateBtn');
    generateBtn.disabled = true;
    
    try {
        // Call API
        const response = await fetch(`${API_BASE_URL}/report/${cleanSymbol}`, {
            method: 'GET',
            headers: {
                'Accept': 'text/plain'
            }
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || `HTTP error! status: ${response.status}`);
        }
        
        // Get markdown content
        const markdown = await response.text();
        currentReport = markdown;
        
        // Render markdown
        renderMarkdown(markdown);
        
        // Show report section
        showReport();
        
        // Show success message
        showAlert(`Report generated successfully for ${cleanSymbol}`, 'success');
        
    } catch (error) {
        console.error('Error generating report:', error);
        hideLoading();
        showAlert(
            `Failed to generate report: ${error.message}. Make sure the API server is running on ${API_BASE_URL}`,
            'danger'
        );
    } finally {
        // Re-enable generate button
        generateBtn.disabled = false;
    }
}

/**
 * Render markdown content to HTML
 */
function renderMarkdown(markdown) {
    const reportContainer = document.getElementById('reportContainer');
    
    if (typeof marked !== 'undefined') {
        // Use marked.js to convert markdown to HTML
        const html = marked.parse(markdown);
        reportContainer.innerHTML = html;
        
        // Highlight code blocks if highlight.js is available
        if (typeof hljs !== 'undefined') {
            reportContainer.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }
    } else {
        // Fallback: display as plain text with line breaks
        reportContainer.innerHTML = '<pre>' + escapeHtml(markdown) + '</pre>';
    }
}

/**
 * Download report as Markdown/TXT file
 */
function downloadMarkdown() {
    if (!currentReport) {
        showAlert('No report available to download', 'warning');
        return;
    }
    
    const blob = new Blob([currentReport], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentSymbol}_report_${getTimestamp()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showAlert('Markdown file downloaded successfully', 'success');
}

/**
 * Download report as PDF using API
 */
async function downloadPDF() {
    if (!currentReport) {
        showAlert('No report available to download', 'warning');
        return;
    }
    
    const downloadBtn = document.getElementById('downloadPDFBtn');
    const originalText = downloadBtn.innerHTML;
    downloadBtn.disabled = true;
    downloadBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Generating PDF...';
    
    try {
        // Call API to generate PDF
        const response = await fetch(`${API_BASE_URL}/pdf`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                markdown_content: currentReport,
                stock_symbol: currentSymbol
            })
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || `HTTP error! status: ${response.status}`);
        }
        
        // Get PDF blob
        const blob = await response.blob();
        
        // Create download link
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        // Get filename from Content-Disposition header or use default
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `${currentSymbol}_report_${getTimestamp()}.pdf`;
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename="?(.+)"?/i);
            if (filenameMatch) {
                filename = filenameMatch[1];
            }
        }
        
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showAlert('PDF downloaded successfully', 'success');
        
    } catch (error) {
        console.error('Error generating PDF:', error);
        showAlert(
            `PDF generation failed: ${error.message}. Make sure the API server is running on ${API_BASE_URL}`,
            'danger'
        );
    } finally {
        downloadBtn.disabled = false;
        downloadBtn.innerHTML = originalText;
    }
}


/**
 * Show loading spinner
 */
function showLoading() {
    document.getElementById('loadingSpinner').style.display = 'block';
}

/**
 * Hide loading spinner
 */
function hideLoading() {
    document.getElementById('loadingSpinner').style.display = 'none';
}

/**
 * Show report section
 */
function showReport() {
    hideLoading();
    document.getElementById('reportSection').style.display = 'block';
    // Scroll to report
    document.getElementById('reportSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Hide report section
 */
function hideReport() {
    document.getElementById('reportSection').style.display = 'none';
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info') {
    const alertContainer = document.getElementById('alertContainer');
    const alertId = 'alert-' + Date.now();
    
    const alertHTML = `
        <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
            <i class="bi bi-${getAlertIcon(type)}"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    
    alertContainer.innerHTML = alertHTML;
    
    // Auto-dismiss after 5 seconds for success/info messages
    if (type === 'success' || type === 'info') {
        setTimeout(() => {
            const alertElement = document.getElementById(alertId);
            if (alertElement) {
                const bsAlert = new bootstrap.Alert(alertElement);
                bsAlert.close();
            }
        }, 5000);
    }
}

/**
 * Hide alert
 */
function hideAlert() {
    document.getElementById('alertContainer').innerHTML = '';
}

/**
 * Get icon for alert type
 */
function getAlertIcon(type) {
    const icons = {
        'success': 'check-circle-fill',
        'danger': 'exclamation-triangle-fill',
        'warning': 'exclamation-triangle-fill',
        'info': 'info-circle-fill'
    };
    return icons[type] || 'info-circle-fill';
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Get timestamp for filename
 */
function getTimestamp() {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    return `${year}${month}${day}_${hours}${minutes}`;
}

