from flask import Blueprint, request, jsonify
from utils import logger
from stockmodel import StockModel
from stock_utils import format_stock_code
import json
import numpy as np
import traceback

api_bp = Blueprint('api', __name__)

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def init_app(app):
    app.json_encoder = NumpyJSONEncoder

@api_bp.route('/analyze', methods=['POST'])
def analyze():
    """分析股票并生成投资建议"""
    stock_model = None
    try:
        data = request.get_json()
        stocks = data.get('stocks', [])
        investment_amount = data.get('investment_amount', 0)
        
        if not stocks or not investment_amount:
            return jsonify({
                "success": False,
                "message": "缺少必要参数"
            })
        
        stock_model = StockModel()
        result = stock_model.analyze(stocks, investment_amount)
        
        # 处理数据中的NaN值
        def process_nan(obj):
            if isinstance(obj, dict):
                return {k: process_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [process_nan(x) for x in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            return obj
        
        # 处理结果数据
        processed_result = process_nan(result)
        
        return jsonify({
            "success": True,
            "message": "分析完成",
            "data": processed_result.get("data", {}),
            "portfolio": processed_result.get("portfolio", {})
        })
        
    except Exception as e:
        logger.error(f"分析请求处理失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "message": f"分析失败: {str(e)}"
        }), 400
    finally:
        if stock_model is not None:
            stock_model.cleanup()

@api_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "healthy"}), 200 