import markdown
import webbrowser
import os

# Paths
INPUT_MD = r'C:\Users\khair\.gemini\antigravity\brain\f500f753-5c2b-42e5-95e6-04d3680c6350\explanation_baseline_training.md'
OUTPUT_HTML = 'baseline_training_deep_dive.html'

if not os.path.exists(INPUT_MD):
    print(f"❌ Error: {INPUT_MD} not found!")
    exit(1)

# Read the markdown content
with open(INPUT_MD, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML with professional extensions
html_body = markdown.markdown(
    md_content, 
    extensions=['tables', 'fenced_code', 'toc', 'attr_list']
)

# Professional CSS Styling (Optimized for PDF Printing)
styled_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Dive: Baseline Training Report</title>
    <style>
        :root {{
            --primary: #1E4FA8;
            --secondary: #3B82F6;
            --text: #333;
            --bg: #fff;
            --code-bg: #f8f9fa;
        }}

        @page {{
            margin: 2cm;
            size: A4;
        }}

        body {{
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
            line-height: 1.7;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            color: var(--text);
            background: var(--bg);
        }}

        .report-header {{
            text-align: center;
            margin-bottom: 50px;
            padding-bottom: 20px;
            border-bottom: 4px solid var(--primary);
        }}

        .report-header h1 {{
            color: var(--primary);
            margin: 0;
            font-size: 2.5em;
            letter-spacing: -1px;
        }}

        h2 {{
            color: var(--primary);
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 8px;
            margin-top: 40px;
            font-size: 1.8em;
        }}

        h3 {{
            color: var(--secondary);
            margin-top: 25px;
            font-size: 1.3em;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 30px 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            border-radius: 8px;
        }}

        th, td {{
            border: 1px solid #e5e7eb;
            padding: 12px 15px;
            text-align: left;
        }}

        th {{
            background-color: var(--primary);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
        }}

        tr:nth-child(even) {{
            background-color: #f9fafb;
        }}

        code {{
            background-color: var(--code-bg);
            padding: 3px 6px;
            border-radius: 5px;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 0.9em;
            color: #d63384;
        }}

        pre {{
            background-color: #111827;
            color: #e5e7eb;
            padding: 20px;
            border-radius: 12px;
            overflow-x: auto;
            font-size: 0.9em;
            margin: 20px 0;
        }}

        .no-print {{
            background: linear-gradient(135deg, #1E4FA8 0%, #3B82F6 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 40px;
            text-align: center;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }}

        .btn-print {{
            display: inline-block;
            background: white;
            color: var(--primary);
            padding: 10px 25px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            margin-top: 15px;
            cursor: pointer;
            border: none;
        }}

        @media print {{
            .no-print {{
                display: none;
            }}
            body {{
                padding: 0;
            }}
        }}
    </style>
</head>
<body>
    <div class="no-print">
        <h2>📑 Export Baseline Training Deep Dive</h2>
        <button class="btn-print" onclick="window.print()">🖨️ Click to Save as PDF</button>
    </div>

    <div class="report-header">
        <h1>Deep Dive: Baseline Training Performance</h1>
    </div>

    <div class="content">
        {html_body}
    </div>
</body>
</html>
"""

# Save HTML file
html_abs_path = os.path.abspath(OUTPUT_HTML)
with open(html_abs_path, 'w', encoding='utf-8') as f:
    f.write(styled_html)

print(f"✅ Report generated at: {html_abs_path}")
webbrowser.open(f'file:///{html_abs_path}')
