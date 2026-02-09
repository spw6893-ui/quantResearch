"""
日志工具模块
"""
import logging
import os
import sys
from config.settings import LOG_DIR, LOG_LEVEL, LOG_FORMAT


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """获取配置好的logger"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    if not logger.handlers:
        # 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(console_handler)

        # 文件输出
        if log_file:
            os.makedirs(LOG_DIR, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(LOG_DIR, log_file), encoding="utf-8"
            )
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            logger.addHandler(file_handler)

    return logger
