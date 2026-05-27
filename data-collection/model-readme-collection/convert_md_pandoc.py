import os
import subprocess

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
source_directory = os.path.join(script_dir, 'readmes')
target_directory = os.path.join(script_dir, 'html_pandoc')

# 创建目标文件夹（如果不存在）
os.makedirs(target_directory, exist_ok=True)

# 检查 pandoc 是否可用
try:
    result = subprocess.run(['pandoc', '--version'], capture_output=True, text=True)
    print(f"Pandoc 版本: {result.stdout.splitlines()[0]}")
except FileNotFoundError:
    print("❌ pandoc 未安装！请先安装:")
    print("   Ubuntu/Debian: sudo apt-get install pandoc")
    print("   macOS: brew install pandoc")
    exit(1)

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
        # 使用 pandoc 将 md 转为 html（pandoc 对表格的解析比 mistune 好）
        result = subprocess.run(
            ['pandoc', md_file_path, '-f', 'markdown', '-t', 'html', '--wrap=none'],
            capture_output=True, text=True, encoding='utf-8', timeout=30
        )

        if result.returncode == 0:
            with open(html_file_path, 'w', encoding='utf-8') as html_file:
                html_file.write(result.stdout)
            success += 1
        else:
            errors += 1
            print(f"❌ pandoc 错误: {filename} - {result.stderr[:200]}")

    except subprocess.TimeoutExpired:
        errors += 1
        print(f"⏰ 超时: {filename}")
    except Exception as e:
        errors += 1
        print(f"❌ 转换失败: {filename} - {e}")

    if total % 50000 == 0:
        print(f"进度: {total}/{len(files)}，成功: {success}，失败: {errors}")

print(f"✅ 转换完成！总计: {total}，成功: {success}，失败: {errors}")