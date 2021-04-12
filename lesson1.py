import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy.stats import multivariate_normal


x_max = 10
y_max = 10
t_max = 6000
i_num = 100
j_num = 100
alpha = 1
a = 0.0005
b = 0.001
x_delta = x_max / i_num
y_delta = y_max / j_num
t_delta = 0.004

center = [3, 3]
std = [[0.1, 0],
       [0, 0.1]]
x_array = np.linspace(0, x_max, i_num)
y_array = np.linspace(0, y_max, j_num)
x_grid, y_grid = np.meshgrid(x_array, y_array)
grid = np.dstack([x_grid, y_grid])
population = multivariate_normal.pdf(grid, center, std)

plt.pcolormesh(x_grid, y_grid, population)
plt.show()

figure = plt.figure(figsize=(5, 5))
image_list = []
for t in range(t_max):
    for i in range(1, i_num - 1):
        for j in range(1, j_num-1):
            num_update = alpha * t_delta * (
                    (population[i + 1, j] - 2 * population[i, j] + population[i - 1, j]) / (x_delta ** 2)
                    + (population[i, j + 1] - 2 * population[i, j] + population[i, j - 1]) / (y_delta ** 2))
            population[i, j] = population[i, j] + num_update
    for i in range(i_num - 1):
        population[i, 0] = population[i, 1] - a * y_delta
        population[i, j_num-1] = population[i, j_num-2] - b * y_delta
    for j in range(j_num - 1):
        population[0, j] = population[1, j]
        population[i_num-1, j] = population[i_num-2, j]
    if t % 50 == 0:
        image = plt.pcolormesh(x_grid, y_grid, population)
        image_list.append([image])
        if t % 100 == 0:
            plt.savefig(f"images_lesson1/image{t}.png")


animation = ArtistAnimation(figure, image_list, interval=100, blit=True, repeat_delay=10)

animation.save("population.gif", writer="pillow")
plt.show()