import os
import csv
from bs4 import BeautifulSoup

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
mistune_html_dir = os.path.join(script_dir, '../data-collection/model-readme-collection/html_mistune')
pandoc_html_dir = os.path.join(script_dir, '../data-collection/model-readme-collection/html_pandoc')
csv_file_path = os.path.join(script_dir, 'readme_features.csv')


# ============================================================
# Feature extraction functions (from mistune HTML)
# ============================================================

def count_num_lists(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return len(soup.find_all(['ol', 'ul']))


def count_images(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    images = soup.find_all('img')
    num_static_img = 0
    num_animated_img = 0
    static_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.svg']
    for img in images:
        src = img.get('src', '').lower()
        if any(src.endswith(ext) for ext in static_extensions):
            num_static_img += 1
        elif src.endswith('.gif'):
            num_animated_img += 1
    return num_static_img, num_animated_img


def check_links(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    a_tags = soup.find_all('a')
    link_tags = soup.find_all('link', href=True)
    iframe_tags = soup.find_all('iframe')

    total_num_of_links = len(a_tags) + len(link_tags) + len(iframe_tags)
    num_github_links = sum('github.com' in tag.get('href', '').lower() for tag in a_tags + link_tags)
    num_huggingface_links = sum('huggingface.co' in tag.get('href', '').lower() for tag in a_tags + link_tags)

    has_video = 0
    video_keywords = ['video', 'youtube', 'vimeo']
    video_extensions = ['.mp4', '.webm', '.ogg']
    for tag in a_tags + iframe_tags:
        url = tag.get('href', '') or tag.get('src', '')
        url = url.lower()
        if any(keyword in url for keyword in video_keywords) or any(url.endswith(ext) for ext in video_extensions):
            has_video = 1
            break

    has_arxiv = 1 if any('arxiv.org' in tag.get('href', '').lower() for tag in a_tags + link_tags) else 0
    has_project_page = 1 if any('project' in tag.get('href', '').lower() for tag in a_tags + link_tags) else 0

    return {
        'total_num_of_links': total_num_of_links,
        'num_github_links': num_github_links,
        'num_huggingface_links': num_huggingface_links,
        'has_video': has_video,
        'has_arxiv': has_arxiv,
        'has_project_page': has_project_page,
    }


def count_code(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    code_tags = soup.find_all('code')
    num_code_blk = 0
    num_inline_code = 0
    for code in code_tags:
        if code.parent.name == 'pre':
            num_code_blk += 1
        else:
            num_inline_code += 1
    return num_code_blk, num_inline_code


def has_bibtex(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    code_tags = soup.find_all('code')
    for code in code_tags:
        if 'language-bibtex' in code.get('class', []):
            return 1
    for code in code_tags:
        if code.parent.name == 'pre':
            content = code.get_text(strip=True).lower()
            if all(keyword in content for keyword in ['author=', 'title=', 'year=']):
                return 1
    return 0


# ============================================================
# YAML metadata features (from mistune HTML)
# ============================================================

def extract_yaml_features(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    yaml_lines = 0
    word_count_yaml = 0

    yaml_metadata_div = soup.find('div', class_='yaml-metadata')
    has_metadata = 1 if yaml_metadata_div else 0

    if yaml_metadata_div:
        yaml_str = str(yaml_metadata_div)
        yaml_lines = len(yaml_str.splitlines())
        yaml_text = yaml_metadata_div.get_text()
        word_count_yaml = len(yaml_text.split())
        yaml_metadata_div.extract()

    content_text = soup.get_text()
    content_lines = len(content_text.splitlines())
    word_count_content = len(content_text.split())

    total_lines = yaml_lines + content_lines
    word_count_all = word_count_yaml + word_count_content

    has_license = 1 if 'license:' in html_content else 0
    has_tags = 1 if 'tags:' in html_content else 0
    has_datasets = 1 if 'datasets:' in html_content or 'dataset:' in html_content else 0

    return {
        'has_metadata': has_metadata,
        'has_license_in_readme': has_license,
        'has_tags_in_readme': has_tags,
        'has_datasets_in_readme': has_datasets,
        'number_of_lines_yaml': yaml_lines,
        'number_of_lines_content': content_lines,
        'number_of_total_lines': total_lines,
        'word_count_yaml': word_count_yaml,
        'word_count_content': word_count_content,
        'word_count_all': word_count_all,
    }


# ============================================================
# Table count (from pandoc HTML)
# ============================================================

def count_tables_pandoc(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # pandoc 生成的 HTML 可能没有 body 标签
    if soup.body:
        return len(soup.body.find_all('table'))
    else:
        return len(soup.find_all('table'))


# ============================================================
# Main
# ============================================================

def main():
    # 获取所有 mistune HTML 文件
    mistune_files = {f for f in os.listdir(mistune_html_dir) if f.endswith('.html')}
    pandoc_files = {f for f in os.listdir(pandoc_html_dir) if f.endswith('.html')}

    # 以 mistune 文件为基准（因为 content 和 yaml 特征都从 mistune 提取）
    all_files = sorted(mistune_files)
    total = len(all_files)
    print(f"共发现 {total} 个 HTML 文件，开始提取特征...")

    fieldnames = [
        'file_name', 'has_readme',
        # content features
        'num_lists', 'num_static_img', 'num_animated_img',
        'total_num_of_links', 'num_github_links', 'num_huggingface_links',
        'has_video', 'has_arxiv', 'has_project_page',
        'num_code_blk', 'num_inline_code', 'has_bibtex',
        # yaml features
        'has_metadata', 'has_license_in_readme', 'has_tags_in_readme', 'has_datasets_in_readme',
        'number_of_lines_yaml', 'number_of_lines_content', 'number_of_total_lines',
        'word_count_yaml', 'word_count_content', 'word_count_all',
        # table feature (from pandoc)
        'num_table',
    ]

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        processed = 0
        errors = 0

        for filename in all_files:
            processed += 1
            name_without_ext = os.path.splitext(filename)[0]

            try:
                # --- 从 mistune HTML 提取 content + yaml 特征 ---
                mistune_path = os.path.join(mistune_html_dir, filename)
                with open(mistune_path, 'r', encoding='utf-8') as f:
                    mistune_html = f.read()

                # Content features
                links_info = check_links(mistune_html)
                num_lists = count_num_lists(mistune_html)
                num_static_img, num_animated_img = count_images(mistune_html)
                num_code_blk, num_inline_code = count_code(mistune_html)
                bibtex_exists = has_bibtex(mistune_html)

                # YAML features
                yaml_features = extract_yaml_features(mistune_html)

                # --- 从 pandoc HTML 提取 table 特征 ---
                num_table = 0
                if filename in pandoc_files:
                    pandoc_path = os.path.join(pandoc_html_dir, filename)
                    with open(pandoc_path, 'r', encoding='utf-8') as f:
                        pandoc_html = f.read()
                    num_table = count_tables_pandoc(pandoc_html)

                # --- 组装一行 ---
                row = {
                    'file_name': name_without_ext,
                    'has_readme': 1,
                    'num_lists': num_lists,
                    'num_static_img': num_static_img,
                    'num_animated_img': num_animated_img,
                    'num_code_blk': num_code_blk,
                    'num_inline_code': num_inline_code,
                    'has_bibtex': bibtex_exists,
                    'num_table': num_table,
                }
                row.update(links_info)
                row.update(yaml_features)

                writer.writerow(row)

            except Exception as e:
                errors += 1
                print(f"❌ 处理失败: {filename} - {e}")

            if processed % 50000 == 0:
                print(f"进度: {processed}/{total}，失败: {errors}")

    print(f"✅ 提取完成！总计: {processed}，失败: {errors}")
    print(f"✅ CSV 已保存: {csv_file_path}")


if __name__ == "__main__":
    main()