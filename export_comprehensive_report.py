"""
OP-ECOM: Comprehensive Report Exporter
Conver Markdown to Professional HTML for PDF Printing
"""
import markdown
import webbrowser
import os

# Paths
INPUT_MD = 'COMPREHENSIVE_FINAL_REPORT.md'
OUTPUT_HTML = 'COMPREHENSIVE_FINAL_REPORT.html'

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
    <title>OP-ECOM Comprehensive Final Report</title>
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

        /* Header / Branding */
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

        .report-header p {{
            color: #666;
            font-size: 1.1em;
            margin-top: 10px;
        }}

        /* Typography */
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

        p, li {{
            font-size: 1.05em;
        }}

        /* Tables (UCI/Benchmark Style) */
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

        /* Code & Blocks */
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

        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
        }}

        /* Print Instructions UI */
        .no-print {{
            background: linear-gradient(135deg, #1E4FA8 0%, #3B82F6 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 40px;
            text-align: center;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }}

        .no-print h2 {{
            color: white;
            border: none;
            margin: 0 0 10px 0;
            font-size: 1.4em;
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
            transition: transform 0.2s;
        }}

        .btn-print:hover {{
            transform: scale(1.05);
        }}

        @media print {{
            .no-print {{
                display: none;
            }}
            body {{
                padding: 0;
            }}
            h2 {{
                page-break-before: always;
            }}
            pre, table {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="no-print">
        <h2>📑 Project Report Successfully Built!</h2>
        <p>Ready for high-quality PDF export.</p>
        <button class="btn-print" onclick="window.print()">🖨️ Click to Save as PDF</button>
        <p style="font-size: 0.8em; margin-top: 10px; opacity: 0.8;">Or press <strong>Ctrl + P</strong> and select "Save as PDF"</p>
    </div>

    <div class="report-header">
        <h1>OP-ECOM Integrated Ecosystem</h1>
        <p>Technical Real-Time Acquisition & Intent Classification Platform</p>
    </div>

    <div class="content">
        {html_body}
    </div>

    <footer style="margin-top: 80px; text-align: center; color: #999; font-size: 0.8em; border-top: 1px solid #eee; padding-top: 20px;">
        Generated on 2026-02-08 | OP-ECOM Technical Documentation
    </footer>
</body>
</html>
"""

# Save HTML file
html_abs_path = os.path.abspath(OUTPUT_HTML)
with open(html_abs_path, 'w', encoding='utf-8') as f:
    f.write(styled_html)

print(f"✅ Professional HTML version generated at: {html_abs_path}")
print("🌐 Opening in your browser now... Click 'Save as PDF' in the blue header.")

# Open in default browser
webbrowser.open(f'file:///{html_abs_path}')
