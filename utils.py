from datetime import datetime


def get_current_time():
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    return dt_string