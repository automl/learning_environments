import time
from datetime import time as datetime_time, datetime, timedelta


def time_diff(start, end):
    if isinstance(start, datetime_time):  # convert to datetime
        assert isinstance(end, datetime_time)
        start, end = [datetime.combine(datetime.min, t) for t in [start, end]]
    if start <= end:  # e.g., 10:33:26-11:15:49
        return end - start
    else:  # end < start e.g., 23:55:00-00:25:00
        end += timedelta(1)  # +day
        assert end > start
        return end - start


def x_minutes_passed(start, end, minutes_passed=30):
    diff = time_diff(start=start, end=end)
    diff_total_seconds = diff.total_seconds()
    seconds_tht_should_have_passed = minutes_passed * 60
    return diff_total_seconds >= seconds_tht_should_have_passed


if __name__ == "__main__":
    t1 = datetime.now()
    time.sleep(90)
    t2 = datetime.now()
    diff_ts = time_diff(start=t1, end=t2)
    print(diff_ts)
    print(dir(diff_ts))

    print(x_minutes_passed(start=t1, end=t2))
    print(x_minutes_passed(start=t1, end=t2, minutes_passed=1))