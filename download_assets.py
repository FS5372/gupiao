import requests
import os

def download_file(url, filename):
    """下载文件到指定路径"""
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {url}")

# 需要下载的文件列表
files_to_download = {
    # CSS 文件
    'https://cdnjs.cloudflare.com/ajax/libs/select2/3.5.4/select2-bootstrap.min.css': 'static/css/select2-bootstrap.min.css',
    'https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css': 'static/css/tailwind.min.css',
    
    # JavaScript 文件
    'https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js': 'static/js/echarts.min.js',
}

# 下载所有文件
for url, filename in files_to_download.items():
    download_file(url, filename) 