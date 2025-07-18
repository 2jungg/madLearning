import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from game.env import make_env

if __name__ == '__main__':
    # 로그 및 모델 저장 디렉토리 생성
    log_dir = "./logs/"
    model_dir = "./models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 병렬 환경 생성 (4개 워커)
    num_cpu = 4
    vec_env = SubprocVecEnv([make_env() for i in range(num_cpu)])

    # 체크포인트 콜백 설정: 10,000 스텝마다 모델 저장
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=model_dir, name_prefix='ppo_qwop')

    # PPO 모델 정의
    model = PPO('MlpPolicy', vec_env, verbose=1, tensorboard_log=log_dir)

    # 모델 학습 시작
    model.learn(total_timesteps=1000000, callback=checkpoint_callback)

    # 최종 모델 저장
    model.save(f"{model_dir}/ppo_qwop_final")

    # 환경 종료
    vec_env.close()
