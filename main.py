
from Buoy_Detection import *
from motion_detection import *




motion_detection()
#display_video()
"""
thread1 = threading.Thread(target= Multi_Threading.display_video())
thread2 = threading.Thread(target= Multi_Threading.motion_detection())
thread2.start()
thread1.start()
"""

"""
pool = ThreadPool(processes=2)
async_result = pool.apply_async(line_detection, (cnts, points))
intersect_points = async_result.get()"""