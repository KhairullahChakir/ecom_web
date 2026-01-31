"""
Convert Markdown to HTML (open in browser and print to PDF)
"""
import markdown
import webbrowser
import os

# Read the markdown file
with open('FULL_PROJECT_REPORT.md', 'r', encoding='utf-8') as f:
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
    <title>OP-ECOM Project Report</title>
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
            font-size: 2em;
        }}
        h2 {{
            color: #2563EB;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
            font-size: 1.5em;
        }}
        h3 {{
            color: #3B82F6;
            font-size: 1.2em;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 0.9em;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #1E4FA8;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #1a1a2e;
            color: #eee;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.85em;
        }}
        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
        }}
        blockquote {{
            border-left: 4px solid #1E4FA8;
            margin: 20px 0;
            padding: 10px 20px;
            background-color: #f8f9fa;
        }}
        hr {{
            border: none;
            border-top: 2px solid #1E4FA8;
            margin: 30px 0;
        }}
        strong {{
            color: #1E4FA8;
        }}
        .print-instructions {{
            background: linear-gradient(135deg, #1E4FA8, #3B82F6);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
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
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="print-instructions">
        <h3>📄 To Export as PDF:</h3>
        <p>Press <strong>Ctrl + P</strong> → Select "Save as PDF" → Click Save</p>
    </div>
    {html_content}
</body>
</html>
"""

# Save HTML file
html_path = os.path.abspath('FULL_PROJECT_REPORT.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(styled_html)

print(f"✅ HTML generated: {html_path}")
print("🌐 Opening in browser... Press Ctrl+P to save as PDF")

# Open in browser
webbrowser.open(f'file:///{html_path}')
