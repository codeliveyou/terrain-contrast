
from Classes import point
import math

earth_R = 6730000

class aircraft:
    def __init__(self):
        self.N = 0
        self.E = 0
        self.velocity = 0
        self.direction = point()
        self.position = point()

    def get_direction(self, inclination_angle, azimuth):
        return point(math.cos(azimuth) * math.cos(inclination_angle), math.sin(azimuth) * math.cos(inclination_angle), math.sin(inclination_angle))
    
    def get_len(self, x, y):
        return math.sqrt(x * x + y * y)

    def get_NE(self):
        E_val = math.atan2(self.direction.y, self.direction.x)
        N_val = math.atan2(self.direction.z, self.get_len(self.direction.x, self.direction.y))
        return [E_val, N_val]

    def set_direction(self, inclination_angle, azimuth):
        self.direction = self.get_direction(inclination_angle, azimuth)
    
    def fly(self, v, d):
        self.position += d * v
    
    def update_NE(self):
        _x = point(-math.sin(self.E), math.cos(self.E), 0)
        _z = self.get_direction(self.N, self.E)
        _y = (_z ^ _z).e()
        earth_position = _x * self.position.x + _y * self.position.y + _z * self.position.z
        earth_position += self.get_direction(self.N, self.E) * earth_R
        current_N, current_E = self.get_NE(earth_position.E())
        self.N = current_N
        self.E = current_E







