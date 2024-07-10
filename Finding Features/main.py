import math

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
        return math.sqrt(self * self)
    
    def E_from_angle(self, angle):
        return point2(math.cos(angle), math.sin(angle))
    
    def conj(self):
        return point2(-self.y, self.x)

    def e(self):
        L = self.len()
        return point2(self.x / L, self.y / L)
    
    def angle(self):
        return math.atan2(self.y, self.x)


def find_patterns(A : point2, B : point2, C : point2, eps : float, similarities : list[list[list[float]]], delta : float) -> list[list[point2]]:
    # Input Values : 
        # A, B, C must be clockwise order
        # eps is the acceptable error when pattarn and terrain are matched
        # similarities is similarity matrix for each feature of pattern formed 3 * row_num * col_num
        # delta is the length of small square's edge
    # Output Values : 
        # the pairs of terrain points that matched with pattern formed 3 * n
    
    row_num = len(similarities)
    col_num = len(similarities[0])
    
    def get_candidates(similarity : list[list[float]], threshold : float, max_count : int) -> list[point2]:
        # Input Values:
            # similarity is the similarity matrix
            # threshold is the limit value of acceptable similarity
            # only the top max_count features will be considered
        # Output Values:
            # candidate points list
    
        ids = []
        for i in range(row_num):
            for j in range(col_num):
                if similarity[i][j] > threshold:
                    ids.append([i, j])
        
        ids = sorted(ids, key = lambda x : -similarity[x[0]][x[1]])

        return [point2(x[0] * delta, x[1] * delta) for x in ids[ : min(len(ids), max_count)]]

    candidates = [get_candidates(similarities[i], 0.8, 100) for i in range(3)]

    AB = (A - B).len()
    BC = (B - C).len()
    CA = (C - A).len()

    result = []

    # option 1 : consider rotating
    for a in candidates[0]:
        for b in candidates[1]:
            ab = (b - a).len()
            if ab < AB - eps or ab > AB + eps:
                continue
            if ((b - a) - (B - A)).len() > eps:
                continue
            for c in candidates[2]:
                if ((b - a) ^ (c - a)) < 0:
                    continue
                if ((c - a) - (C - A)).len() > eps:
                    continue
                result.append([a, b, c])
                break
    
    # option 2 : ignore rotating
    for a in candidates[0]:
        for b in candidates[1]:
            if abs((a - b).len() - AB) > eps:
                continue
            for c in candidates[2]:
                if abs((b - c).len() - BC) > eps:
                    continue
                if abs((c - a).len() - CA) > eps:
                    continue
                if ((b - a) ^ (c - a)) < 0:
                    continue
                result.append([a, b, c])
                break
    
    return result


