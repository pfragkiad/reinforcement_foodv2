import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import time
import cv2

# create reinforcement learning environment
#style.use("ggplot")

size = 10
episodes = 25_000
movePenalty = 1
enemyPenalty = 300
foodReward = 25

epsilon = 0.5
epsilonDecay = 0.98

showEvery = 5_000

startQ = None #'qtable-1688339047.pickle' # None  # or filename

learningRate = 0.1
discount = 0.95

actionsCount = 4

from enum import Enum


class ItemColor(Enum):
    PLAYER = 1
    FOOD = 2
    ENEMY = 3


# colors in bgr
d = {
    ItemColor.PLAYER: (255, 175, 0),
    ItemColor.FOOD: (0, 255, 0),
    ItemColor.ENEMY: (0, 0, 255),
}

if startQ is None:
    q = {}
    # observation space (x1,y1),(x2,y2)
    for x1 in range(-size + 1, size):
        for y1 in range(-size + 1, size):
            for x2 in range(-size + 1, size):
                for y2 in range(-size + 1, size):
                    # we have 4 discreet actions, so we need 4 values for each combination
                    q[((x1, y1), (x2, y2))] = [
                        np.random.uniform(-5, 0) for i in range(actionsCount)
                    ]
else:
    with open(startQ, "rb") as f:
        q = pickle.load(f)


from Blob import Blob

episodeRewards = []
for episode in range(episodes):
    player = Blob(size)
    food = Blob(size)
    enemy = Blob(size)

    if (episode+1) % showEvery == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{showEvery} ep mean {np.mean(episodeRewards[-showEvery:])}")
        show = True
    else:
        show = False

    episodeReward = 0
    for i in range(500):

        #s(t)
        currentState = (player - food, player - enemy)

        #a(t)
        if np.random.random() > epsilon:
            action = np.argmax(q[currentState])
        else:
            action = np.random.randint(0, actionsCount)

        player.action(action)

        ##maybe later (train without moving the following)
        # enemy.move()
        # food.move()

        if player.x == enemy.x and player.y == enemy.y:
            reward = -enemyPenalty
        elif player.x == food.x and player.y == food.y:
            reward = foodReward
        else:
            reward = -movePenalty

        nextState = (player - food, player - enemy)

        # max ( Q(s(t+1), a(t)) )
        maxFutureQ = np.max(q[nextState])
        #currentQ = q[nextState][action]

        # Q (s(t), a(t))
        currentQ = q[currentState][action] 

        if reward == foodReward:
            newQ = foodReward
        elif reward == -enemyPenalty:
            newQ = -enemyPenalty
        else:
            newQ = (1 - learningRate) * currentQ + learningRate * (
                reward + discount * maxFutureQ
            )

        #q[nextState][action] = newQ
        #update Q (s(t), a(t) )
        q[currentState][action] = newQ

        #https://docs.python.org/3/library/enum.html
        if show:
            env = np.zeros((size, size, 3), dtype=np.uint8)
            env[food.y][food.x] = d[ItemColor.FOOD]
            env[player.y][player.x] = d[ItemColor.PLAYER]
            env[enemy.y][enemy.x] = d[ItemColor.ENEMY]

            img = Image.fromarray(env,"RGB")
            img = img.resize((300,300))
            cv2.imshow("",np.array(img))

            if reward == foodReward or reward == -enemyPenalty:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break


        episodeReward +=reward
        if reward == foodReward or reward == -enemyPenalty:
            break

    episodeRewards.append(episodeReward)
    epsilon*=epsilonDecay

movingAverage = np.convolve(episodeRewards,np.ones((showEvery,)) / showEvery, mode="valid")   

plt.plot([i for i in range(len(movingAverage))],movingAverage)
plt.ylabel(f'reward {showEvery}')
plt.xlabel("episode #")
plt.show()

# with open(f'qtable-{int(time.time())}.pickle','wb') as f:
#     pickle.dump(q,f)