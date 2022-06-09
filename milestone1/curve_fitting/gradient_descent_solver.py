__author__ = 'Yu Wang <ywang27@uchicago.edu>'

import math
import matplotlib.pyplot as plt


class GradientDesentSolverSecondOrder:
    def __init__(self, lr: float, epsilon: float, epoch: int) -> None:
        self.h = 1.0
        self.m = 1.0
        self.b = 1.0
        self.h_prd = 0.0
        self.m_prd = 0.0
        self.b_prd = 0.0
        # Learning rate
        self.lr = lr
        # Convergence criteria
        self.epsilon = epsilon
        # Epoch num
        self.epoch = epoch
        self.train_data = []
        self.grads = []
    
    def __len__(self) -> int:
        return len(self.train_data)

    def set_train_data(self, path: str):
        with open(path, 'r') as f:
            points = f.readlines()
            for point in points[1:]:
                [x, y] = point.strip('\n').split('\t')
                x, y = float(x), float(y)
                self.train_data.append((x, y))
    
    def plot_gld(self, path):
        # points = sorted(self.train_data, key=lambda x:x[0], reverse=True)
        x = [point[0] for point in self.train_data]
        y = [point[1] for point in self.train_data]
        plt.scatter(x, y)
        plt.savefig(path)
    
    def train(self):
        for i in range(self.epoch):
            nabla_h = 0.0
            nabla_m = 0.0
            nabla_b = 0.0

            # Iterate over all training data 
            for (x, y) in self.train_data:
                nabla_h += 2 * self.h * x**4 + 2 * self.m * x**3 + 2 * self.b * x**2 - 2 * y * x**2
                nabla_m += 2 * self.h * x**3 + 2 * self.m * x**2 + 2 * self.b * x - 2 * x * y
                nabla_b += 2 * self.h * x**2 + 2 * self.m * x + 2 * self.b - 2 * y
            
            nabla_h = nabla_h / len(self)
            nabla_m = nabla_m / len(self)
            nabla_b = nabla_b / len(self)

            self.h = self.h - self.lr * nabla_h
            self.m = self.m - self.lr * nabla_m
            self.b = self.b - self.lr * nabla_b

            try:
                size = nabla_h**2 + nabla_m**2 + nabla_b**2
            except:
                print("Divergence detected. Try a smaller learning rate.")
                return
            norm = math.sqrt(size)
            if i % 10000 == 0:
                print(f"Epoch {i + 1}\tError function's gradient: {norm}")
                self.grads.append(norm)

            if norm < self.epsilon:
                self.h_prd = self.h
                self.m_prd = self.m
                self.b_prd = self.b
                print(f"Convergence detected on epoch {i + 1}. ")
                print(f"Found optimal values: h - {self.h_prd}, m - {self.m_prd}, b - {self.b_prd}")
                print(self.grads)
                return
        
        self.h_prd = self.h
        self.m_prd = self.m
        self.b_prd = self.b
        print(f"Cannot converge with the given epsilon {self.epsilon} in {self.epoch} epochs.")
        print(f"Values for the last training step are: h - {self.h_prd}, m - {self.m_prd}, b - {self.b_prd}")
        print(self.grads)
            
    def plot_prd(self, path):
        x_prd = []
        y_prd = []
        points = sorted(self.train_data, key=lambda x:x[0], reverse=True)
        y = [p[1] for p in points]
        for p in points:
            x = p[0]
            x_prd.append(x)
            y_prd.append(self.h_prd * x**2 + self.m_prd * x + self.b_prd)
        plt.plot(x_prd, y_prd, c='b')
        plt.scatter(x_prd, y, c='r')
        plt.savefig(path)


if __name__ == "__main__":
    solver = GradientDesentSolverSecondOrder(0.0001, 0.00001, 200000)
    solver.set_train_data("./points.txt")
    solver.train()
    solver.plot_prd("./prd.png")
    # solver.paint_gld("./gold.png")