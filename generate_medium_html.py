"""
Generate a styled, self-contained HTML / visual reader for Docs/MEDIUM_ARTICLE.md
with all figures embedded for easy previewing, printing to PDF, or copying to Medium.
"""
import os
import re
import base64

def image_to_base64(img_path):
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(img_path)[1].lower().replace(".", "")
            if ext == "jpg": ext = "jpeg"
            return f"data:image/{ext};base64,{data}"
    return ""

def main():
    md_path = os.path.join("Docs", "MEDIUM_ARTICLE.md")
    html_out = os.path.join("Docs", "MEDIUM_ARTICLE.html")
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Replace local image paths with embedded base64 data URIs
    img_matches = re.findall(r'!\[(.*?)\]\((.*?)\)', md_text)
    for alt, rel_path in img_matches:
        # resolve path relative to Docs/
        abs_img_path = os.path.normpath(os.path.join("Docs", rel_path))
        b64_uri = image_to_base64(abs_img_path)
        if b64_uri:
            md_text = md_text.replace(f"![{alt}]({rel_path})", f'<div class="img-container"><img src="{b64_uri}" alt="{alt}"/><p class="caption">{alt}</p></div>')

    # Convert simple markdown to html
    # Headers
    md_text = re.sub(r'^### (.*)$', r'<h3>\1</h3>', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^## (.*)$', r'<h2>\1</h2>', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^# (.*)$', r'<h1>\1</h1>', md_text, flags=re.MULTILINE)
    
    # Blockquotes
    md_text = re.sub(r'^> (.*)$', r'<blockquote>\1</blockquote>', md_text, flags=re.MULTILINE)
    
    # Bold & Italic
    md_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', md_text)
    md_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', md_text)
    
    # Horizontal rules
    md_text = re.sub(r'^---$', r'<hr/>', md_text, flags=re.MULTILINE)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>We Gave an AI 32 Neurons and Brain-Like Rules. It Built an Internal GPS.</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
        line-height: 1.7;
        color: #242424;
        background-color: #ffffff;
        max-width: 780px;
        margin: 0 auto;
        padding: 40px 20px 80px 20px;
    }}
    h1 {{
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.25;
        margin-bottom: 8px;
        color: #1a1a1a;
    }}
    h3.subtitle {{
        font-size: 1.3rem;
        font-weight: 400;
        color: #6b6b6b;
        margin-top: 0;
        margin-bottom: 24px;
        line-height: 1.4;
    }}
    h2 {{
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 40px;
        margin-bottom: 16px;
        color: #1a1a1a;
    }}
    p {{
        font-size: 1.1rem;
        margin-bottom: 20px;
    }}
    blockquote {{
        border-left: 3px solid #242424;
        padding-left: 20px;
        margin: 24px 0;
        font-style: italic;
        color: #333333;
        font-size: 1.15rem;
    }}
    .img-container {{
        margin: 32px 0;
        text-align: center;
    }}
    img {{
        max-width: 100%;
        height: auto;
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}
    .caption {{
        font-size: 0.9rem;
        color: #6b6b6b;
        margin-top: 8px;
        text-align: center;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 28px 0;
        font-size: 0.95rem;
    }}
    th, td {{
        padding: 12px 14px;
        text-align: left;
        border-bottom: 1px solid #e0e0e0;
    }}
    th {{
        background-color: #f7f7f7;
        font-weight: 600;
    }}
    hr {{
        border: none;
        border-top: 1px solid #eaeaea;
        margin: 36px 0;
    }}
    ul, ol {{
        font-size: 1.1rem;
        margin-bottom: 20px;
        padding-left: 28px;
    }}
    li {{
        margin-bottom: 8px;
    }}
    a {{
        color: #1a8917;
        text-decoration: underline;
    }}
    @media print {{
        body {{ max-width: 100%; padding: 0; }}
        img {{ max-width: 90% !important; }}
    }}
</style>
</head>
<body>
{md_text}
</body>
</html>
"""
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"Generated standalone visual HTML article: {html_out}")

if __name__ == "__main__":
    main()
