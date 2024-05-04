import math
# __mul__ for multiplication (*)
# __truediv__ for division (/)
# __and__ for bitwise AND (&)
# __or__ for bitwise OR (|)


class point:
    def __init__(self, x, y, z):
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
        elif isinstance(a, float):
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
    


