import math

pi = math.pi

class point:
    def __init__(self, x = 0, y = 0, z = 0):
        self.x = x
        self.y = y
        self.z = z
    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    def __add__(self, a):
        return point(self.x + a.x, self.y + a.y, self.z + a.z)
    
    def __sub__(self, a):
        return point(self.x - a.x, self.y - a.y, self.z - a.z)
    def __mul__(self, a):
        if isinstance(a, point):
            return self.x * a.x + self.y * a.y + self.z * a.z
        else:
            return point(self.x * a, self.y * a, self.z * a)
    def __xor__(self, a):
        return point(self.y * a.z - self.z * a.y,
                     -(self.x * a.z - self.z * a.x),
                     self.x * a.y - self.y * a.x)

    def __truediv__(self, a):
        return point(self.x / a, self.y / a, self.z / a)
    
    def len(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def e(self):
        L = self.len()
        return point(self.x / L, self.y / L, self.z / L)
    

class point2:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
    def __repr__(self):
        return f"({self.x}, {self.y})"
    
    def __add__(self, a):
        return point2(self.x + a.x, self.y + a.y)
    def __sub__(self, a):
        return point2(self.x - a.x, self.y - a.y)
    def __mul__(self, a):
        if isinstance(a, point2):
            return self.x * a.x + self.y * a.y
        else:
            return point2(self.x * a, self.y * a)
    def __xor__(self, a):
        return self.x * a.y - self.y * a.x

    def __truediv__(self, a):
        return point2(self.x / a, self.y / a)
    
    def len(self):
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def E_from_angle(self, angle):
        return point2(math.cos(angle), math.sin(angle))
    
    def conj(self):
        return point2(-self.y, self.x)

    def e(self):
        L = self.len()
        return point2(self.x / L, self.y / L)