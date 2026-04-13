import datetime


def current_timestamp():
    now = datetime.datetime.now()
    timestamp = str(now.year) + '-' + str(now.month).zfill(2) + str(now.day).zfill(2) + '-' + \
        str(now.hour).zfill(2) + str(now.minute).zfill(2) + '-' + str(now.second).zfill(2)
    print(f'current timestamp is {timestamp}')
    return timestamp
