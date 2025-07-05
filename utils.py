import sys
import logging
import os
from datetime import datetime
import traceback
from pathlib import Path
import tempfile
import time
import pandas as pd
import bs4 as bs

class Logger:
    instance = None
    
    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
            cls.instance.__initialize()
        return cls.instance
    
    def __initialize(self):
        # 在当前目录创建logs文件夹
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 使用时间戳创建日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'stock_analysis_{timestamp}.log')
        
        print(f"日志文件路径: {self.log_file}")
        
        # 配置更详细的日志格式
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s:\n%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # 配置logger
        self.logger = logging.getLogger('StockAnalysis')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 记录启动信息
        self.logger.info("=== 股票分析系统启动 ===")
        self.logger.info(f"日志文件: {self.log_file}")
        self.logger.info(f"Python版本: {sys.version}")
        self.logger.info(f"运行目录: {os.getcwd()}")
        
    def debug(self, msg):
        """调试级别日志"""
        self.logger.debug(str(msg))
        
    def info(self, msg):
        """信息级别日志"""
        self.logger.info(str(msg))
        
    def warning(self, msg):
        """警告级别日志"""
        self.logger.warning(str(msg))
        
    def error(self, msg):
        """错误级别日志"""
        self.logger.error(f"[ERROR] {str(msg)}")
        
    def critical(self, msg):
        """严重错误级别日志"""
        self.logger.critical(f"[CRITICAL] {str(msg)}")
        
    def exception(self, msg):
        """异常日志，包含堆栈信息"""
        self.logger.exception(f"[EXCEPTION] {str(msg)}")
        
    def data(self, title, data):
        """记录数据相关的日志"""
        self.logger.debug(f"\n=== {title} ===\n{str(data)}\n")
        
    def step(self, step_name, details=None):
        """记录步骤日志"""
        msg = f"\n{'='*20} {step_name} {'='*20}"
        if details:
            msg += f"\n{str(details)}"
        self.logger.info(msg)
        
    def metric(self, name, value):
        """记录指标日志"""
        self.logger.info(f"指标 - {name}: {value}")
        
    def section(self, name):
        """记录分节日志"""
        self.logger.info(f"\n{'#'*50}\n{name}\n{'#'*50}")

# 全局logger实例
logger = Logger()

def log_gpu_info():
    """记录GPU信息"""
    try:
        import torch
        import subprocess
        
        logger.section("GPU 信息")
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"cuDNN版本: {torch.backends.cudnn.version()}")
            
            device_count = torch.cuda.device_count()
            logger.info(f"GPU数量: {device_count}")
            
            for i in range(device_count):
                device = torch.device(f'cuda:{i}')
                logger.info(f"\nGPU {i} 详细信息:")
                logger.info(f"  名称: {torch.cuda.get_device_name(i)}")
                logger.info(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**2:.2f} MB")
                logger.info(f"  计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
            
            try:
                nvidia_smi = subprocess.check_output(['nvidia-smi']).decode()
                logger.info(f"\nnvidia-smi 输出:\n{nvidia_smi}")
            except:
                logger.warning("无法获取nvidia-smi信息")
        else:
            logger.warning("未检测到CUDA GPU")
            
    except Exception as e:
        logger.error(f"获取GPU信息失败: {str(e)}")
        logger.exception(e)

def fetch_stock_data(stock_code, start_date, end_date):
    """获取股票历史数据"""
    logger.step(f"获取股票数据 - {stock_code}", 
               f"时间范围: {start_date} 到 {end_date}")
    
    try:
        # ... 其余代码保持不变 ...
        pass
    except Exception as e:
        logger.error(f"获取股票数据失败: {str(e)}")
        logger.exception(e)
        return pd.DataFrame() 