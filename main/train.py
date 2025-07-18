import os
import torch
import multiprocessing
import http.server
import socketserver
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from game.env import make_env


def run_server(port, directory):
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    with socketserver.TCPServer(('', port), handler) as httpd:
        print(f"Serving at port {port} for directory {directory}")
        httpd.serve_forever()


if __name__ == '__main__':
    # 로그 및 모델 저장 디렉토리 생성
    log_dir = "./logs/"
    model_dir = "./models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 병렬 환경 생성 (4개 워커)
    num_cpu = 4
    start_port = 8000
    game_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'game'))

    servers = []
    for i in range(num_cpu):
        port = start_port + i
        server_process = multiprocessing.Process(target=run_server, args=(port, game_dir))
        server_process.daemon = True
        server_process.start()
        servers.append(server_process)

    vec_env = SubprocVecEnv([make_env(port=start_port + i) for i in range(num_cpu)])

    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # 체크포인트 콜백 설정: 10,000 스텝마다 모델 저장
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=model_dir, name_prefix='ppo_qwop')

    # PPO 모델 정의
    model = PPO('MlpPolicy', vec_env, verbose=1, tensorboard_log=log_dir, device=device)

    try:
        # 모델 학습 시작
        model.learn(total_timesteps=1000000, callback=checkpoint_callback)

        # 최종 모델 저장
        model.save(f"{model_dir}/ppo_qwop_final")

    finally:
        # 환경 및 서버 종료
        vec_env.close()
        for server in servers:
            server.terminate()
            server.join()
