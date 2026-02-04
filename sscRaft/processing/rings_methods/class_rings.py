import numpy as np

class TomoAttributes_:
    def __init__(self,  n_length, n_angle, n_slice,
                        size_median, dim, large_stripes_size,
                        background_ratio, drop_ratio):
        self.n_length               = np.int32(n_length)
        self.n_angle                = np.int32(n_angle)
        self.n_slice                = np.int32(n_slice)
        self.size_median            = np.int32(size_median)
        self.dim                    = np.int32(dim)
        self.large_stripes_size     = np.int32(large_stripes_size)
        self.background_ratio       = np.float32(background_ratio)
        self.drop_ratio             = np.float32(drop_ratio)

    def display_info(self):
        # Display a formatted string containing all the details of the vehicle
        print(f"TomoAttributes: {self.n_angle} {self.n_slice} {self.n_length}")
