"""
Simple HTTP server to view the generated HTML report.
Run this script to open the report in your default browser.
"""
import http.server
import socketserver
import webbrowser
import os
from pathlib import Path
import time
import threading

PORT = 8080
REPORT_DIR = Path(__file__).parent / "reports"

class ReportHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPORT_DIR), **kwargs)
    
    def log_message(self, format, *args):
        # Suppress server logs for cleaner output
        pass

def open_browser():
    """Open browser after a short delay to ensure server is ready."""
    time.sleep(1)
    
    # Find the latest report
    reports = list(REPORT_DIR.glob("analysis_report_*.html"))
    if reports:
        latest_report = max(reports, key=lambda p: p.stat().st_mtime)
        url = f"http://localhost:{PORT}/{latest_report.name}"
        print(f"\n✅ Opening report in browser: {latest_report.name}")
        webbrowser.open(url)
    else:
        print("\n❌ No reports found in reports/ directory")
        print("   Run verify_full_pipeline.py first to generate a report")

if __name__ == "__main__":
    os.chdir(REPORT_DIR.parent)
    
    print("=" * 60)
    print("🚀 Autonomous Insight Engine - Report Viewer")
    print("=" * 60)
    print(f"\n📂 Serving reports from: {REPORT_DIR}")
    print(f"🌐 Server running at: http://localhost:{PORT}")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    # Start browser in separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start server
    with socketserver.TCPServer(("", PORT), ReportHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n✋ Server stopped")
