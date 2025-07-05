from flask import Flask, render_template, url_for, jsonify, request, send_from_directory
from api import api_bp, init_app
from flask_cors import CORS
from utils import logger

app = Flask(__name__, 
    static_folder='static',
    static_url_path='/static'
)
CORS(app)

# 初始化JSON编码器
init_app(app)

# 注册蓝图
app.register_blueprint(api_bp, url_prefix='/api')

@app.route('/')
def index():
    logger.info("访问首页")
    return render_template('index.html')

# 添加请求日志中间件
@app.before_request
def before_request():
    logger.info(f"收到请求: {request.method} {request.path} {request.json if request.is_json else request.args}")

@app.after_request
def after_request(response):
    logger.info(f"请求完成: {response.status}")
    return response

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"发生错误: {str(error)}")
    return jsonify({'error': str(error)}), 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.static_folder, 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True) 