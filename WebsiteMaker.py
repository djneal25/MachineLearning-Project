import os
import re
from collections import defaultdict
import random

def tokenize(content):
    tokens = []
    for line in content.splitlines():
        for token in re.findall(r"[<>\w\s{};()]+", line):
            tokens.append(token)
    return tokens

def build_vocabulary(files, file_extension):
    vocabulary = defaultdict(int)
    for filename in files:
        if filename.endswith(file_extension):
            print(f"Processing {file_extension} file: {filename}")
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
                tokens = tokenize(content)
                for token in tokens:
                    vocabulary[token] += 1
    return vocabulary

def generate_file(vocabulary, file_extension, placeholder_image_path):
    generated_file = ""
    for token in vocabulary.keys():
        if token.endswith(file_extension):
            generated_file += re.sub(r'url\([^)]*\)', f'url({placeholder_image_path})', token) + '\n'
        else:
            generated_file += token + '\n'
    return generated_file

def generate_html(vocabulary, n_words, css_content, js_content, placeholder_image_path):
    html = ""
    for _ in range(n_words):
        # Choose a random token from the vocabulary
        token = random.choice(list(vocabulary.keys()))

        if token.startswith("<") and token.endswith(">"):
            if token.startswith("<img"):
                html += f'<img src="{placeholder_image_path}" alt="Placeholder Image">'
            else:
                html += token
        elif token.endswith(".css"):
            html += f'<style>{css_content}</style>'
        elif token.endswith(".js"):
            html += f'<script>{js_content}</script>'
        elif re.match(r'^\w+$', token):
            html += f'Lorem Ipsum'
        else:
            html += token

    return html

def main():
    all_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            all_files.append(os.path.join(root, file))

    # Build vocabularies for each file type
    js_vocabulary = build_vocabulary(all_files, ".js")
    css_vocabulary = build_vocabulary(all_files, ".css")
    html_vocabulary = build_vocabulary(all_files, ".html")

    # Generate new JS, CSS, and HTML files
    placeholder_image_path = "C:\\Users\\goutt\\Documents\\Python Scripts\\placeholder.jpg"
    new_js = generate_file(js_vocabulary, ".js", placeholder_image_path)
    new_css = generate_file(css_vocabulary, ".css", placeholder_image_path)
    new_html = generate_html(html_vocabulary, 1000, new_css, new_js, placeholder_image_path)

    # Save the new files
    with open("new_script.js", "w", encoding="utf-8") as f:
        f.write(new_js)

    with open("new_style.css", "w", encoding="utf-8") as f:
        f.write(new_css)

    with open("new_website.html", "w", encoding="utf-8") as f:
        f.write(new_html)

if __name__ == "__main__":
    main()
