import logging
# debug, info, warning, error, critical
# logging.basicConfig(filename='firing.log', level=logging.INFO)

def create_log(log_name= 'firing'):
	my_logger = logging.Logger(log_name)
	my_logger.setLevel(logging.DEBUG)

	my_fh = logging.FileHandler(log_name+'.log')
	my_fh.setLevel(logging.DEBUG)

	my_ch = logging.StreamHandler()
	my_ch.setLevel(logging.WARNING)

	# my_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# my_fh.setFormatter(my_formatter)
	# my_ch.setFormatter(my_formatter)

	my_logger.addHandler(my_fh)
	my_logger.addHandler(my_ch)
	return my_logger,my_fh