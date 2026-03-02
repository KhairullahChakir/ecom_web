import markdown
import webbrowser
import os

# Paths
MD_PATH = r'C:\Users\khair\.gemini\antigravity\brain\f500f753-5c2b-42e5-95e6-04d3680c6350\THESIS_REPORT_ULTIMATE.md'
HTML_PATH = os.path.join(os.path.dirname(MD_PATH), 'THESIS_REPORT_ULTIMATE.html')

# Read the markdown file
with open(MD_PATH, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML
html_content = markdown.markdown(
    md_content, 
    extensions=['tables', 'fenced_code', 'toc']
)

# Add professional styling
styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>E-commerce Prediction System - Final Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px;
            color: #333;
            background: #fff;
        }}
        h1 {{
            color: #1E4FA8;
            border-bottom: 3px solid #1E4FA8;
            padding-bottom: 10px;
            font-size: 2.2em;
            text-align: center;
        }}
        h2 {{
            color: #2563EB;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
            font-size: 1.6em;
        }}
        h3 {{
            color: #3B82F6;
            font-size: 1.3em;
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 0.9em;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #1E4FA8;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f8fbff;
        }}
        code {{
            background-color: #f1f5f9;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 0.9em;
            color: #e11d48;
        }}
        pre {{
            background-color: #0f172a;
            color: #f8fafc;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.85em;
            margin: 20px 0;
        }}
        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
        }}
        blockquote {{
            border-left: 5px solid #3B82F6;
            margin: 20px 0;
            padding: 10px 20px;
            background-color: #eff6ff;
            font-style: italic;
        }}
        hr {{
            border: none;
            border-top: 2px solid #e2e8f0;
            margin: 40px 0;
        }}
        .print-instructions {{
            background: linear-gradient(135deg, #1e40af, #3b82f6);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 40px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .print-instructions h3 {{
            color: white;
            margin: 0 0 10px 0;
        }}
        @media print {{
            .print-instructions {{
                display: none;
            }}
            body {{
                padding: 0;
            }}
            @page {{
                margin: 2cm;
            }}
        }}
    </style>
</head>
<body>
    <div class="print-instructions">
        <h3>🎓 Ready to Export Your Thesis Report?</h3>
        <p>1. Press <strong>Ctrl + P</strong> on your keyboard</p>
        <p>2. Set Destination to <strong>"Save as PDF"</strong></p>
        <p>3. Click <strong>Save</strong> and select your desktop!</p>
    </div>
    {html_content}
</body>
</html>
"""

# Save HTML file
with open(HTML_PATH, 'w', encoding='utf-8') as f:
    f.write(styled_html)

print(f"✅ Success! Report generated here: {HTML_PATH}")
print("🌐 Opening in your browser now...")

# Open in browser
webbrowser.open(f'file:///{HTML_PATH}')
