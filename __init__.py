from flask import Flask
from .api import init_app as init_api

def create_app():
    app = Flask(__name__)
    init_api(app)  # 初始化自定义JSON编码器
    # ... 其他初始化代码 ...
    return app 