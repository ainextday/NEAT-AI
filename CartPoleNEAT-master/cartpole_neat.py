import numpy as np

# Monkey Patch: ทำให้ numpy.bool8 = bool
if not hasattr(np, 'bool8'):
    np.bool8 = bool

import gym
import os
import neat
import time  # ✅ ใช้สำหรับ delay

# ✅ กำหนด render_mode ให้แสดงผลเกมได้
env = gym.make('CartPole-v1', render_mode="human")
env.reset(seed=42)  

def eval_genomes(genomes, config):
    for _, genome in genomes:
        observation, _ = env.reset(seed=42)  # Reset environment
        done = False
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0  

        while not done:
            output = net.activate(observation)  
            action = np.argmax(output)  

            step_result = env.step(action)
            if len(step_result) == 4:  
                observation, reward, done, _ = step_result
            else:  
                observation, reward, terminated, truncated, _ = step_result
                terminated = bool(terminated)  
                truncated = bool(truncated)  
                done = terminated or truncated  

            if abs(observation[3]) > 0.5:
                done = True
                reward = -5  
            genome.fitness += reward  
        env.reset(seed=42)  

def run(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50)

    print('\nBest genome:\n{!s}'.format(winner))
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    test_model(winner_net)

def test_model(winner):
    total_score = 0

    for _ in range(10):  # ✅ ให้โมเดลเล่น 10 รอบเพื่อดูพฤติกรรม
        observation, _ = env.reset(seed=42)
        score = 0
        done = False

        while not done:
            env.render()  # ✅ แสดงผลเกม

            output = winner.activate(observation)
            action = np.argmax(output)

            step_result = env.step(action)
            if len(step_result) == 4:
                observation, reward, done, _ = step_result
            else:
                observation, reward, terminated, truncated, _ = step_result
                terminated = bool(terminated)  
                truncated = bool(truncated)  
                done = terminated or truncated  

            if abs(observation[3]) > 0.5:
                done = True
            score += reward

            time.sleep(0.05)  # ✅ ใส่ delay ให้ดูการเคลื่อนไหวชัดขึ้น

        total_score += score

    env.close()  # ✅ ปิดหน้าต่างเกมเมื่อเล่นเสร็จ
    print(f"Score Over 10 tries: {total_score / 10}")

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
