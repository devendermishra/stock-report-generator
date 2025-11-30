# Stock Report Generator UI

A modern web interface for generating and viewing stock research reports.

## Features

- **Simple Input**: Enter NSE stock symbols (e.g., RELIANCE, TCS, INFY)
- **Formatted Display**: View reports with properly formatted markdown
- **Download Options**: 
  - Download as Markdown/TXT file (client-side)
  - Download as PDF (server-side generation via API with rate limiting)
- **Modern UI**: Bootstrap 5 with a beautiful blue theme
- **Responsive Design**: Works on desktop and mobile devices
- **Server-Side PDF Generation**: PDFs are generated on the server for better quality and reliability

## Setup

1. Make sure the API server is running:
   ```bash
   # From the project root
   python -m src.api
   # Or using uvicorn
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

2. Open the UI:
   - Simply open `index.html` in a web browser
   - Or serve it using a local web server:
     ```bash
     # Using Python
     cd ui
     python -m http.server 3000
     # Then open http://localhost:3000 in your browser
     
     # Using Node.js
     npx http-server ui -p 3000
     ```

## Configuration

If your API is running on a different host or port, edit `app.js` and change the `API_BASE_URL`:

```javascript
const API_BASE_URL = 'http://localhost:8000'; // Change this to your API URL
```

## Usage

1. Enter an NSE stock symbol in the input field (e.g., `RELIANCE`)
2. Click "Generate Report" or press Enter
3. Wait for the report to be generated (this may take a few moments)
4. View the formatted report in the display area
5. Download the report as Markdown or PDF using the download buttons

## Browser Compatibility

- Modern browsers (Chrome, Firefox, Safari, Edge)
- Requires JavaScript enabled
- PDF generation works best in Chrome and Firefox

## Dependencies

All dependencies are loaded via CDN:
- Bootstrap 5.3.0
- Marked.js (for markdown parsing)
- Highlight.js (for code syntax highlighting)
- jsPDF (for PDF generation)
- html2canvas (for converting HTML to PDF)

No installation required!

