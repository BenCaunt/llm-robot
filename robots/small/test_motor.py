from small import TurboPi
import time

if __name__ == "__main__":
    turbo = TurboPi()
    turbo.drive(100, 100)
    time.sleep(1)
    turbo.drive(0, 0)
