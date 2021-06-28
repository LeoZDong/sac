LOG_INT=1000

python -u train.py --log_interval $LOG_INT --purge |& tee print_logs.txt