import markdown
import os
import webbrowser
from datetime import datetime

# Configuration
MD_FILE = r'C:\Users\khair\.gemini\antigravity\brain\f500f753-5c2b-42e5-95e6-04d3680c6350\explanation_model2_master.md'
OUTPUT_HTML = r'd:\op_ecom\master_model2_audit_report.html'

# Executive CSS Styling (Premium Dark/Gold Theme)
CSS_STYLE = """
<style>
    :root {
        --primary-color: #1a1a1a;
        --secondary-color: #2c3e50;
        --accent-color: #d4af37; /* Gold accent */
        --bg-color: #ffffff;
        --text-color: #333333;
        --border-color: #eeeeee;
    }
    
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        line-height: 1.7;
        color: var(--text-color);
        max-width: 1000px;
        margin: 0 auto;
        padding: 60px;
        background-color: var(--bg-color);
    }
    
    header {
        border-bottom: 5px solid var(--accent-color);
        margin-bottom: 50px;
        padding-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }
    
    h1 { color: var(--primary-color); font-size: 3em; margin: 0; letter-spacing: -2px; line-height: 1; }
    h2 { color: var(--secondary-color); border-bottom: 2px solid var(--border-color); padding-bottom: 10px; margin-top: 60px; font-weight: 700; text-transform: uppercase; font-size: 1.2em; letter-spacing: 2px; }
    h3 { color: var(--accent-color); margin-top: 35px; font-size: 1.5em; font-weight: 600; }
    
    p, li { font-size: 1.1em; margin-bottom: 20px; }
    
    table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 40px 0;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }
    
    th, td { padding: 18px 25px; text-align: left; }
    th { background-color: var(--primary-color); color: #fff; font-weight: 500; text-transform: uppercase; font-size: 0.8em; letter-spacing: 1.5px; }
    tr:nth-child(even) { background-color: #fafafa; }
    
    blockquote {
        border-left: 8px solid var(--accent-color);
        background: #fdfaf2;
        margin: 30px 0;
        padding: 20px 40px;
        font-style: italic;
    }

    .footer {
        margin-top: 100px;
        padding-top: 40px;
        border-top: 1px solid var(--border-color);
        font-size: 0.9em;
        color: #999;
        text-align: center;
    }
    
    @media print {
        body { padding: 0; }
        .no-print { display: none; }
    }
</style>
"""

def generate():
    if not os.path.exists(MD_FILE):
        print("MD file not found.")
        return

    with open(MD_FILE, 'r', encoding='utf-8') as f:
        md_content = f.read()

    html_body = markdown.markdown(md_content, extensions=['extra', 'tables', 'toc'])

    now = datetime.now().strftime("%B %d, %Y")
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Master Model 2 Audit Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        {CSS_STYLE}
    </head>
    <body>
        <header>
            <div>
                <p style="margin: 0; color: var(--accent-color); font-weight: 700; text-transform: uppercase; font-size: 0.8em; letter-spacing: 2px;">OP-ECOM Intelligence Unit</p>
                <h1>Model 2 Master Report</h1>
            </div>
            <p style="margin: 0; color: #999; font-weight: 500;">{now}</p>
        </header>
        
        {html_body}
        
        <div class="footer">
            <p>&copy; 2026 OP-ECOM • Sequential AI Audit Protocol V2.1</p>
        </div>
        
        <div class="no-print" style="position: fixed; bottom: 40px; right: 40px;">
            <button onclick="window.print()" style="background: var(--primary-color); color: white; border: none; padding: 15px 30px; border-radius: 50px; cursor: pointer; font-family: 'Inter'; font-weight: 600; box-shadow: 0 10px 20px rgba(0,0,0,0.2); transition: 0.3s;">
                Download PDF Report
            </button>
        </div>
    </body>
    </html>
    """

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(f"Master report generated: {OUTPUT_HTML}")
    webbrowser.open('file://' + os.path.abspath(OUTPUT_HTML))

if __name__ == "__main__":
    generate()
