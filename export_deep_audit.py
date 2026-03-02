import markdown
import os
import webbrowser
from datetime import datetime

# Files to export
MD_FILE = r'C:\Users\khair\.gemini\antigravity\brain\f500f753-5c2b-42e5-95e6-04d3680c6350\deep_technical_audit.md'
HTML_FILE = r'd:\op_ecom\deep_infrastructure_audit_EXPERT.html'

CSS = """
<style>
    :root {
        --primary: #0f172a;
        --secondary: #334155;
        --accent: #3b82f6;
        --code: #f1f5f9;
        --text: #1e293b;
    }
    body {
        font-family: 'Inter', -apple-system, system-ui, sans-serif;
        line-height: 1.8;
        color: var(--text);
        max-width: 1100px;
        margin: 0 auto;
        padding: 80px 40px;
        background: #f8fafc;
    }
    .container {
        background: white;
        padding: 60px;
        border-radius: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    h1 { color: var(--primary); font-size: 3.5rem; letter-spacing: -0.05em; margin-bottom: 2rem; border-bottom: 8px solid var(--accent); display: inline-block; }
    h2 { color: var(--primary); font-size: 2rem; margin-top: 4rem; display: flex; align-items: center; }
    h2:before { content: '◈'; margin-right: 15px; color: var(--accent); }
    h3 { color: var(--secondary); font-size: 1.5rem; margin-top: 2rem; }
    code { background: var(--code); padding: 4px 8px; border-radius: 6px; font-family: 'Fira Code', monospace; font-size: 0.9em; }
    table { width: 100%; border-collapse: collapse; margin: 30px 0; border-radius: 12px; overflow: hidden; }
    th, td { padding: 20px; text-align: left; border-bottom: 1px solid #e2e8f0; }
    th { background: var(--primary); color: white; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.1em; }
    tr:hover { background: #f1f5f9; }
    .status-expert { color: #059669; font-weight: 900; background: #ecfdf5; padding: 10px 20px; border-radius: 50px; border: 2px solid #059669; }
    @media print { .no-print { display: none; } }
</style>
"""

def generate():
    if not os.path.exists(MD_FILE): return
    with open(MD_FILE, 'r', encoding='utf-8') as f:
        content = markdown.markdown(f.read(), extensions=['extra', 'tables', 'toc'])
    
    html = f"""
    <html><head><title>Deep technical Audit - EXPERT</title>{CSS}</head>
    <body>
        <div class="container">
            <div style="text-align:right; margin-bottom: 20px;">
                <span class="status-expert">SYSTEM RANK: EXPERT AI INFRASTRUCTURE</span>
            </div>
            {content}
            <hr style="margin-top:50px; border:0; border-top:1px solid #eee;">
            <p style="text-align:center; color:#94a3b8; font-size:0.8rem;">
                Audit Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Technical Lead: AI Auditor
            </p>
        </div>
        <div class="no-print" style="position:fixed; bottom:30px; right:30px;">
            <button onclick="window.print()" style="padding:15px 30px; background:var(--primary); color:white; border:none; border-radius:50px; box-shadow:0 10px 15px -3px rgba(0,0,0,0.1); cursor:pointer; font-weight:700;">Export Engineering PDF</button>
        </div>
    </body></html>
    """
    with open(HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Deep Audit Exported: {HTML_FILE}")
    webbrowser.open('file://' + os.path.abspath(HTML_FILE))

if __name__ == "__main__":
    generate()
