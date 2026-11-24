1,2,3:
    The slow threshold factor is 1, which is 8192, any token hands with wavelength over that will get a full slowdown of dividing by 32.
    The fast threshold factor is 4, which is 8192/4 = 2048, any token hands with wavelength less than that will be kept the same
4: The get partly slowed down, depending on how close they are to both ends, closer to 2048 spectrum with be slowed barely and those closer to the 8192 spectrum will be slowed nearly the full 32x. Although I dont know the actual math but yes