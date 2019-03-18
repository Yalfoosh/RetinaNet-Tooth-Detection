import time
from train import main
from datetime import datetime, timedelta
from threading import Thread


def run():
    print("Sleeping for 2 minutes now.")
    time.sleep(120)

    main()



def run_on_different_thread(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


while 1:
    run_on_different_thread(fn=main)

    dt = datetime.now() + timedelta(seconds=10)

    while datetime.now() < dt:
        time.sleep(1)



