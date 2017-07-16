import numpy as np

class MathUtils:
    @staticmethod
    def __create_time(dt, number):

        t = np.zeros((number))
        for i in range(number):
            t[i] = i * dt
        return t

    @staticmethod
    def get_polynomial_coefficients(fps, data, degree):
        N = len(data)
        dt = 1.0/fps
        t = MathUtils.__create_time(dt, N)
        return np.polyfit(t, data, degree)

if __name__ == '__main__':
    pass
