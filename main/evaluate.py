import os
import sys
# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import multiprocessing
import http.server
import socketserver
import time
import glob
from functools import partial

from stable_baselines3 import PPO
from game.env import QWOPEnv

def run_server(port, directory):
    """지정된 디렉토리에서 간단한 HTTP 서버를 실행합니다."""
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    # 주소 재사용 허용
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('', port), handler) as httpd:
        print(f"Serving at port {port} for directory {directory}")
        httpd.serve_forever()

def evaluate_model(model_path, env, num_episodes=30):
    """주어진 모델을 지정된 횟수만큼 평가하고 결과를 반환합니다."""
    model = PPO.load(model_path, env=env)
    total_rewards = []
    total_distances = []

    print(f"\n--- {os.path.basename(model_path)} 모델 평가 시작 ---")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
        
        # 에피소드 종료 후 최종 거리 가져오기
        distance = env.previous_torso_x
        total_rewards.append(total_reward)
        total_distances.append(distance)
        print(f"에피소드 {episode + 1}: 보상 = {total_reward:.2f}, 거리 = {distance:.2f}m")

    avg_reward = sum(total_rewards) / num_episodes
    avg_distance = sum(total_distances) / num_episodes
    print(f"--- 평가 완료 ---")
    print(f"평균 보상: {avg_reward:.2f}")
    print(f"평균 거리: {avg_distance:.2f}m")
    
    return avg_reward, avg_distance

if __name__ == '__main__':
    # 설정
    model_dir = "./models/"
    start_port = 8001  # 학습 포트와 다른 포트 사용
    game_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'game'))
    num_eval_episodes = 30

    # 웹 서버 실행
    server_process = multiprocessing.Process(target=run_server, args=(start_port, game_dir))
    server_process.daemon = True
    server_process.start()
    time.sleep(2)  # 서버가 시작될 시간을 줍니다.

    # 평가 환경 생성 (GUI 렌더링 비활성화)
    env = QWOPEnv(port=start_port, render_mode=None)

    try:
        # 모델 파일 목록 가져오기
        model_files = glob.glob(os.path.join(model_dir, "*.zip"))
        if not model_files:
            print(f"{model_dir}에서 모델 파일을 찾을 수 없습니다.")
        else:
            # 각 모델 평가
            for model_path in sorted(model_files):
                evaluate_model(model_path, env, num_episodes=num_eval_episodes)
                # 모델 간 평가 사이에 잠시 대기
                print("\n다음 모델 평가를 위해 5초간 대기합니다...")
                time.sleep(5)

    finally:
        # 환경 및 서버 종료
        env.close()
        server_process.terminate()
        server_process.join()
        print("\n평가가 종료되었습니다.")
