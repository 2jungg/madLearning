import os
import torch
import multiprocessing
import http.server
import socketserver
from functools import partial

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from game.env_v2 import make_env


def run_server(port, directory):
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(('', port), handler) as httpd:
        print(f"Serving at port {port} for directory {directory}")
        httpd.serve_forever()


if __name__ == '__main__':
    # --- 하이퍼파라미터 설정 ---
    N_LSTM_LAYERS = 2          # 사용할 LSTM 계층의 수
    LSTM_HIDDEN_SIZE = 128     # 각 LSTM 계층의 히든 유닛 크기
    MLP_SIZE = 128             # 정책/가치 네트워크의 MLP 크기
    TOTAL_TIMESTEPS = 1000000  # 총 학습 타임스텝
    N_STEPS = 2048             # 각 환경에서 데이터를 수집할 스텝 수 (늘림)
    NUM_CPU = 4                # 사용할 CPU 코어 수 (1로 고정)
    # --------------------------

    # 로그 및 모델 저장 디렉토리 생성
    log_dir = "./logs/"
    model_dir = "./models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 단일 환경 생성
    start_port = 8000
    game_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'game'))

    server_process = multiprocessing.Process(target=run_server, args=(start_port, game_dir))
    server_process.daemon = True
    server_process.start()

    env = make_env(port=start_port, render_mode='headless')()

    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # 커스텀 정책 아키텍처 정의
    policy_kwargs = dict(
        net_arch=dict(
            pi=[MLP_SIZE, MLP_SIZE],
            vf=[MLP_SIZE, MLP_SIZE],
        ),
        n_lstm_layers=N_LSTM_LAYERS,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
    )

    # 체크포인트 콜백 설정: 10,000 스텝마다 모델 저장
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=model_dir, name_prefix='ppo_lstm_qwop')

    # PPO 모델 정의
    model = RecurrentPPO(
        'MlpLstmPolicy',
        env,  # vec_env 대신 env 사용
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
        n_steps=N_STEPS,
        policy_kwargs=policy_kwargs
    )

    try:
        # 모델 학습 시작
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

        # 최종 모델 저장
        model.save(f"{model_dir}/ppo_lstm_qwop_final")

    finally:
        # 환경 및 서버 종료
        env.close()
        server_process.terminate()
        server_process.join()
