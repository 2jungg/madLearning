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
import socket
import msvcrt

from stable_baselines3 import PPO
from game.env_v2 import QWOPEnv

def find_available_port(start_port=8000, end_port=9000):
    """지정된 범위 내에서 사용 가능한 포트를 찾습니다."""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    raise IOError("No available ports found in the specified range.")

def run_server(port, directory):
    """지정된 디렉토리에서 간단한 HTTP 서버를 실행합니다."""
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    # 주소 재사용 허용
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('', port), handler) as httpd:
        print(f"Serving at port {port} for directory {directory}")
        httpd.serve_forever()

def evaluate_model(model_path, env, num_episodes=5):
    """주어진 모델을 지정된 횟수만큼 평가하고 결과를 반환합니다."""
    model = PPO.load(model_path, env=env)
    total_rewards = []
    total_distances = []

    print(f"\n--- {os.path.basename(model_path)} 모델 평가 시작 ---")
    print("평가를 중단하고 다음 모델로 넘어가려면 Enter 키를 누르세요.")

    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            skipped = False
            distance = 0
            while not done:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\r':  # Enter key on Windows
                        print("\n사용자 입력으로 현재 모델 평가를 중단합니다.")
                        skipped = True
                        break
                
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                if done:
                    distance = info.get('distance', 0)

            if skipped:
                break

            # 에피소드 종료 후 최종 거리 가져오기
            total_rewards.append(total_reward)
            total_distances.append(distance)
            print(f"episode {episode + 1}: reward = {total_reward:.2f}, distance = {distance:.2f}m")

    finally:
        # Windows에서는 터미널 설정을 복원할 필요가 없습니다.
        pass

    if total_rewards:
        avg_reward = sum(total_rewards) / len(total_rewards)
        avg_distance = sum(total_distances) / len(total_distances)
        print(f"--- eval done ---")
        print(f"avg reward: {avg_reward:.2f}")
        print(f"avg dist: {avg_distance:.2f}m")
        return avg_reward, avg_distance
    else:
        print(f"--- eval terminated ---")
        return 0, 0

if __name__ == '__main__':
    # 설정
    model_dir = "./models_v2/"
    start_port = find_available_port() # 사용 가능한 포트를 동적으로 찾기
    print(f"Found available port: {start_port}")
    game_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'game'))
    num_eval_episodes = 5

    # 웹 서버 실행
    server_process = multiprocessing.Process(target=run_server, args=(start_port, game_dir))
    server_process.daemon = True
    server_process.start()
    time.sleep(2)  # 서버가 시작될 시간을 줍니다.

    # 평가 환경 생성 (GUI 렌더링 비활성화)
    env = QWOPEnv(port=start_port, render_mode='human')

    try:
        # 모델 파일 목록 가져오기
        model_files = glob.glob(os.path.join(model_dir, "*.zip"))
        if not model_files:
            print(f"{model_dir}에서 모델 파일을 찾을 수 없습니다.")
        else:
            # 각 모델 평가
            for model_path in sorted(model_files):
                evaluate_model(model_path, env, num_episodes=num_eval_episodes)

    finally:
        # 환경 및 서버 종료
        env.close()
        server_process.terminate()
        server_process.join()
        print("\neval finished.")
