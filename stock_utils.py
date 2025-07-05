import baostock as bs
import pandas as pd
from utils import logger
from datetime import datetime, timedelta

def safe_baostock_login(retry_count=3):
    """安全地登录 Baostock
    
    Args:
        retry_count: 最大重试次数
        
    Returns:
        bool: 是否登录成功
    """
    for i in range(retry_count):
        try:
            lg = bs.login()
            if lg.error_code == '0':
                logger.info(f"Baostock 登录成功 (尝试 {i+1})")
                return True
            else:
                logger.warning(f"Baostock 登录失败 (尝试 {i+1}): {lg.error_msg}")
        except Exception as e:
            logger.error(f"Baostock 登录异常 (尝试 {i+1}): {str(e)}")
    return False

def safe_baostock_logout():
    """安全地登出 Baostock"""
    try:
        bs.logout()
        logger.info("Baostock 成功注销")
    except Exception as e:
        logger.error(f"Baostock 注销失败: {str(e)}")

def format_stock_code(stock_code):
    """格式化股票代码为 baostock 所需的格式"""
    # 如果已经是正确格式就直接返回
    if stock_code.startswith(('sh.', 'sz.')):
        return stock_code
        
    # 根据股票代码判断市场并格式化
    if stock_code.startswith('6'):
        return f'sh.{stock_code}'
    elif stock_code.startswith(('0', '3')):
        return f'sz.{stock_code}'
    else:
        logger.error(f'无效的股票代码: {stock_code}')
        return None

def get_stock_history(stock_code, days=180):
    """获取股票历史数据"""
    try:
        # 格式化股票代码
        formatted_code = format_stock_code(stock_code)
        if not formatted_code:
            return None
            
        # 确保已登录
        safe_baostock_login()
        
        # 获取当前日期
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # 查询历史数据
        rs = bs.query_history_k_data_plus(
            formatted_code,  # 使用格式化后的代码
            "date,open,high,low,close,volume,amount,turn",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"
        )
        
        # 检查查询是否成功
        if rs.error_code != '0':
            logger.error(f'查询历史数据失败: {rs.error_msg}')
            return None
            
        # 转换为DataFrame
        df = pd.DataFrame(rs.data, columns=rs.fields)
        if df.empty:
            logger.error(f'未获取到股票 {formatted_code} 的历史数据')
            return None
            
        # 转换数据类型
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # 计算移动平均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        
        # 设置日期索引
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"获取历史数据失败: {str(e)}")
        return None
    finally:
        safe_baostock_logout()

def calculate_technical_indicators(df):
    """计算技术指标"""
    try:
        # 计算移动平均线
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 计算布林带
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (std * 2)
        df['BB_lower'] = df['BB_middle'] - (std * 2)
        
        return df
        
    except Exception as e:
        logger.error(f'计算技术指标时出错: {str(e)}')
        return df 