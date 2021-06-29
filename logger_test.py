# import absl
# from absl import logging

# absl.flags.FLAGS.mark_as_parsed()

# logging.get_absl_handler().use_absl_log_file(
#     'test_logging.log', './')
# logging.set_verbosity(logging.INFO)
# logging.info('hi!')
# logging.warning('debug!')

# import os
# import absl
# from absl import logging
# if not os.path.exists('./'):
#     os.makedirs('./')
# logging.get_absl_handler().use_absl_log_file('absl_logging', './')
# absl.flags.FLAGS.mark_as_parsed()
# logging.set_verbosity(logging.INFO)

# logging.info('test')

import logging
logging.basicConfig(filename='example.log', level=logging.INFO,
                    format='%(asctime)s %(message)s', 
                    datefmt='%m-%d %H:%M:%S', filemode='w')
logging.getLogger().addHandler(logging.StreamHandler())

logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
