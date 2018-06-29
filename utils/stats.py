class AVG:
    def __init__(self):
        self.count = 0
        self.sum = 0

    def add(self, n):
        self.count += 1
        self.sum += n

    def __str__(self):
        return str(self.sum / self.count)
