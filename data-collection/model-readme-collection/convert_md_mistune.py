import os
import mistune
import yaml
import re

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
source_directory = os.path.join(script_dir, 'readmes')
target_directory = os.path.join(script_dir, 'html_mistune')

# 创建目标文件夹（如果不存在）
os.makedirs(target_directory, exist_ok=True)

# 准备 Markdown 解析器
markdown_parser = mistune.create_markdown()

# 统计
total = 0
success = 0
errors = 0

files = [f for f in os.listdir(source_directory) if f.endswith('.md')]
print(f"共发现 {len(files)} 个 README 文件，开始转换...")

for filename in files:
    total += 1
    md_file_path = os.path.join(source_directory, filename)
    html_file_path = os.path.join(target_directory, filename.replace('.md', '.html'))

    # 跳过已转换的文件（断点续传）
    if os.path.exists(html_file_path):
        success += 1
        continue

    try:
        # 读取 Markdown 文件
        with open(md_file_path, 'r', encoding='utf-8') as md_file:
            md_content = md_file.read()

        # 尝试提取 YAML 块
        match = re.match(r'^---\n(.+?)\n---', md_content, re.DOTALL)
        yaml_content = ""
        if match:
            yaml_block = match.group(1)
            try:
                yaml_data = yaml.safe_load(yaml_block)
                yaml_content = f"<div class='yaml-metadata'>{yaml.dump(yaml_data, default_flow_style=False)}</div>"
            except yaml.YAMLError:
                pass  # YAML 解析失败就跳过

        # 将 Markdown 转换为 HTML（移除 YAML 块）
        md_body = re.sub(r'^---\n.+?\n---', '', md_content, flags=re.DOTALL)
        html_content = markdown_parser(md_body)

        # 合并 YAML HTML 和 Markdown HTML
        full_html_content = yaml_content + html_content

        # 写入 HTML 文件
        with open(html_file_path, 'w', encoding='utf-8') as html_file:
            html_file.write(full_html_content)

        success += 1
    except Exception as e:
        errors += 1
        print(f"❌ 转换失败: {filename} - {e}")

    if total % 50000 == 0:
        print(f"进度: {total}/{len(files)}，成功: {success}，失败: {errors}")

print(f"✅ 转换完成！总计: {total}，成功: {success}，失败: {errors}")