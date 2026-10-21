import numpy as np


b_0 = [1/3, 1/3, 1/3]
b_1 = np.zeros(3)
b_prime_1 = np.zeros(3)
like_1 = np.full(3, 0.15)
h = 0.06
bot_move_1 = 0

for i in range(3):
    b_prime_1[i] = b_0[i] * (1 - h) + (1 - b_0[i]) * h / 2

like_1[bot_move_1] = 0.7
for i in range(3):
    b_1[i] = b_prime_1[i] * like_1[i] / np.sum(b_prime_1 * like_1)

print(b_prime_1, like_1, b_1,)
