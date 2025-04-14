import logging

file_handler = None

def get_logger(name, log_file=None):
    global file_handler

    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        if log_file and file_handler is None:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)

        if file_handler:
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        logger.setLevel(logging.INFO)
    
    return logger

# Example usage
if __name__ == "__main__":
    logger = get_logger(__name__, log_file='app.log')
    logger.info("This will log to both the console and the same file across modules.")
