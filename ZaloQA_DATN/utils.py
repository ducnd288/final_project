import logging

def init_logger():
    logging.basicConfig(filename='log.txt', filemode='a', format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
                    
