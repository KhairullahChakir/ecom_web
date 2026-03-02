import markdown
import os
import webbrowser
from datetime import datetime

# Files to export
REPORTS = [
    {
        'md': r'C:\Users\khair\.gemini\antigravity\brain\f500f753-5c2b-42e5-95e6-04d3680c6350\explanation_model2_deployment.md',
        'html': r'd:\op_ecom\model2_deployment_audit.html',
        'title': 'Model 2 ONNX Deployment Audit',
        'subtitle': 'Research to Production'
    },
    {
        'md': r'C:\Users\khair\.gemini\antigravity\brain\f500f753-5c2b-42e5-95e6-04d3680c6350\explanation_model2_evaluation.md',
        'html': r'd:\op_ecom\model2_final_evaluation_audit.html',
        'title': 'Model 2 Final Evaluation Audit',
        'subtitle': 'Probabilistic Reliability'
    }
]

CSS = """
<style>
    :root { --primary: #2c3e50; --accent: #e67e22; --bg: #ffffff; }
    body { font-family: 'Inter', sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; padding: 50px; }
    header { border-bottom: 4px solid var(--accent); margin-bottom: 30px; }
    h1 { color: var(--primary); font-size: 2.5em; margin-bottom: 0px; }
    h2 { color: var(--primary); border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 40px; }
    code { background: #f4f4f4; padding: 2px 5px; border-radius: 4px; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    th, td { padding: 15px; text-align: left; border-bottom: 1px solid #eee; }
    th { background: var(--primary); color: white; }
    .footer { margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.9em; }
    @media print { .no-print { display: none; } }
</style>
"""

def generate():
    for report in REPORTS:
        if not os.path.exists(report['md']): continue
        with open(report['md'], 'r', encoding='utf-8') as f:
            content = markdown.markdown(f.read(), extensions=['extra', 'tables'])
        
        full_html = f"""
        <html><head><title>{report['title']}</title>{CSS}</head>
        <body>
            <header><p style="text-align:right; color:#7f8c8d;">{report['subtitle']}</p></header>
            {content}
            <div class="footer">Generated on {datetime.now().strftime('%Y-%m-%d')}</div>
            <div class="no-print" style="position:fixed; bottom:20px; right:20px;">
                <button onclick="window.print()" style="padding:10px 20px; background:var(--accent); color:white; border:none; cursor:pointer; border-radius:5px;">Print PDF</button>
            </div>
        </body></html>
        """
        with open(report['html'], 'w', encoding='utf-8') as f:
            f.write(full_html)
        print(f"Exported: {report['html']}")
        webbrowser.open('file://' + os.path.abspath(report['html']))

if __name__ == "__main__":
    generate()
