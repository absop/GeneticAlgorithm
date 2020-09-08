import math
import random
import numpy as np

omega = 10
sigma = 2

def rand(a, b):
    number =  random.uniform(a, b)
    return math.floor(number * 100) / 100


def fitness(x, y):
    r1 = np.sin(omega * x)**2
    r2 = np.sin(omega * y)**2
    r3 = np.e**((x + y) / sigma)
    return r1 * r2 * r3


code_string = (
    lambda code: "\n".join(l.lstrip() for l in code.split("\n")))("""
    bit_mask_lo = (1 << (self.half_length - cross_pos)) - 1
    bit_mask_hi = ~bit_mask_lo
    s_hi = self.{0} & bit_mask_hi
    s_lo = self.{0} & bit_mask_lo
    g_hi = gene.{0} & bit_mask_hi
    g_lo = gene.{0} & bit_mask_lo
    self.{0} = s_hi | g_lo
    gene.{0} = g_hi | s_lo
""")


class Gene():
    full_length = 46
    half_length = full_length // 2
    scale = (1 << half_length) / 10

    def __init__(self, decimal_x, decimal_y):
        self.x = self.__encode(decimal_x)
        self.y = self.__encode(decimal_y)

    def __encode(self, decimal):
        integer = int((decimal + 5.0) * self.scale)
        return integer

    def __decode(self, integer):
        decimal = (integer / self.scale) - 5.0
        return decimal

    def decode(self):
        x = self.__decode(self.x)
        y = self.__decode(self.y)
        return x, y

    def mutation(self):
        r = random.randint(0, self.full_length - 1)
        j = r // self.half_length
        k = r %  self.half_length
        exec("self.{} ^= (1 << (self.half_length - {} - 1))".format(
            ["x", "y"][j], k))

    def crossover(self, gene):
        cross_pos = random.randint(1, self.full_length - 2)
        if cross_pos < self.half_length:
            self.y, gene.y = gene.y, self.y
            chromosome = "x"
        else:
            cross_pos -= self.half_length
            chromosome = "y"

        exec(code_string.format(chromosome))


class Population():
    pc = 0.6
    pm = 0.01

    def __init__(self, size=500):
        self.size = size
        self.genes = []
        self.fitnesses = []
        self.results = []

        for i in range(size):
            x = rand(-5.0, 5.0)
            y = rand(-5.0, 5.0)
            g = Gene(x, y)
            f = fitness(x, y)
            self.genes.append(g)
            self.fitnesses.append(f)

        self.getbest()

    def eliminate(self):
        for i in range(self.size):
            if self.fitnesses[i] < 0:
                self.fitnesses[i] = 0.0

    def getbest(self):
        index = self.fitnesses.index(max(self.fitnesses))
        x, y = self.genes[index].decode()
        z = self.fitnesses[index]
        self.results.append((x, y, z))

    def select(self):
        total = sum(self.fitnesses)
        p = 0.0
        plist = []
        for i in range(self.size):
            p += self.fitnesses[i] / total
            plist.append(p)

        new_genes = []
        for i in range(self.size):
            num = random.random()
            j = 0
            for j in range(self.size):
                if plist[j] < num:
                    continue
                else:
                    break

            index = j if j != 0 else (self.size - 1)
            new_genes.append(self.genes[index])

        self.genes = new_genes

    def crossover(self):
        for i in range(0, self.size - 1, 2):
            if random.random() < self.pc:
                self.genes[i].crossover(self.genes[i + 1])

    def mutation(self):
        for i in range(self.size):
            if random.random() < self.pm:
                self.genes[i].mutation()

    def update_fitness(self):
        for i in range(self.size):
            x, y = self.genes[i].decode()
            self.fitnesses[i] = fitness(x, y)

    def evolute(self, ngeneration):
        for i in range(ngeneration):
            self.eliminate()
            self.select()
            self.crossover()
            self.mutation()
            self.update_fitness()
            self.getbest()

    def returnbest(self):
        self.results.sort(key=lambda i:i[2])
        return self.results[len(self.results)-1]


if __name__ == '__main__':
    demo = Population(size=500)
    demo.evolute(1000)
    x, y, z = demo.returnbest()
    print(x, y, z)
