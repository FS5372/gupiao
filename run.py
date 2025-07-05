import os
import sys
import traceback
import psutil
import torch
from app import app
from utils import logger, log_gpu_info

def check_gpu_details():
    """详细检查GPU配置"""
    try:
        import torch
        
        logger.info("=== GPU 详细诊断 ===")
        logger.info(f"PyTorch 版本: {torch.__version__}")
        logger.info(f"CUDA 可用性: {torch.cuda.is_available()}")
        logger.info(f"CUDA 版本: {torch.version.cuda}")
        logger.info(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"可用 GPU 数量: {gpu_count}")
        
        for i in range(gpu_count):
            logger.info(f"\nGPU {i} 详情:")
            logger.info(f"  名称: {torch.cuda.get_device_name(i)}")
            logger.info(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**2:.2f} MB")
            
    except Exception as e:
        logger.error(f"GPU 诊断失败: {str(e)}")
        logger.error(traceback.format_exc())

def check_system_resources():
    """检查系统资源"""
    try:
        check_gpu_details()  # 添加 GPU 诊断
        
        process = psutil.Process(os.getpid())
        
        logger.info("=== 系统资源检查 ===")
        logger.info(f"Python版本: {sys.version}")
        logger.info(f"进程ID: {process.pid}")
        logger.info(f"CPU核心数: {psutil.cpu_count()}")
        logger.info(f"总内存: {psutil.virtual_memory().total / (1024*1024*1024):.2f} GB")
        logger.info(f"可用内存: {psutil.virtual_memory().available / (1024*1024*1024):.2f} GB")
        
    except Exception as e:
        logger.error(f"系统资源检查失败: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        # 检查系统资源
        check_system_resources()
        
        # 启动服务
        logger.info("=== 启动股票分析服务 ===")
        app.run(
            debug=True, 
            host='0.0.0.0', 
            port=5000,
            use_reloader=False  # 禁用热重载
        )
    
    except Exception as e:
        logger.error("服务启动失败")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        sys.exit(1) 