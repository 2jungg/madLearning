# BatchCollector.py
from multiprocessing import Process, Queue
import time
from queue import Empty
from ReplayBuffer import ReplayBuffer
from env import QWOPEnv

def actor_worker(
    env_class,
    sequence_length,
    result_queue,
    cycles_per_worker=50,
    model_path="model.pth",
    reload_every=10
):
    import torch

    def load_policy(model_path):
        # model = torch.load(model_path, map_location='cpu')
        # model.eval()
        # return model
        pass

    def select_action(model, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_probs = model(obs_tensor)
            action = torch.argmax(action_probs, dim=1).item()
        return 1

    model = load_policy(model_path)
    for episode_idx in range(cycles_per_worker):
        # 모델 주기적 재로딩
        if episode_idx > 0 and (episode_idx % reload_every == 0):
            try:
                model = load_policy(model_path)
            except Exception as e:
                print(f"[actor_worker] 모델 로딩 예외 발생: {e}. 구 모델로 진행")
        # 환경 생성 예외(크롬 브라우저 crash 등) 대응
        try:
            env = env_class()
        except Exception as e:
            print(f"[actor_worker] 환경 초기화 예외 발생: {e}. 3초 대기 후 재시도")
            time.sleep(3)
            continue

        try:
            try:
                obs = env.reset()
            except Exception as e:
                print(f"[actor_worker] 환경 reset 예외 발생: {e}. 환경 재생성")
                try:
                    env.close()
                except Exception:
                    pass
                continue

            trajectory = []
            done = False
            interval = 0.1
            trigger_time = time.time()
            for t in range(sequence_length):
                try:
                    # action = select_action(model, obs)
                    action = 1
                except Exception as e:
                    print(f"[actor_worker] 모델 인퍼런스 예외 발생: {e}. 무작위 행동 사용")
                    # 예외 발생 시 랜덤 행동
                    action = env.action_space.sample()
                try:
                    next_obs, reward, done, info = env.step(action)
                except Exception as e:
                    print(f"[actor_worker] 환경 step 예외 발생: {e}. 에피소드 종료로 간주")
                    break
                
                now = time.time()
                sleep_time = trigger_time + interval - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    print(f"[actor_worker] Frame delayed by {sleep_time:.4f} sec at cycle {episode_idx}, step {t}")
                else:
                    print(f"[actor_worker] Frame delayed by {sleep_time:.4f} sec at cycle {episode_idx}, step {t}")
                trajectory.append((obs, action, reward, next_obs, done))
                obs = next_obs
                trigger_time = time.time()
                if done:
                    break
            # 완주 트래젝터리만 저장
            try:
                if len(trajectory) == sequence_length and not any([step[4] for step in trajectory[:-1]]):
                    result_queue.put(trajectory)
            except Exception as e:
                print(f"[actor_worker] 큐 제출 예외 발생: {e}. 해당 trajectory 무시")
        except Exception as e:
            print(f"[actor_worker] 수집 루프의 예외 발생: {e}")
        finally:
            try:
                env.close()
            except Exception as e:
                print(f"[actor_worker] 환경 종료 예외: {e}")

class BatchCollector:
    def __init__(self, env_class, actor_num=7, sequence_length=50, cycles_per_worker=10, total_cycles=100):
        """
        total_cycles : 전체 수집하고자 하는 trajectory 수 (최소)
        """
        self.env_class = env_class
        self.actor_num = actor_num
        self.sequence_length = sequence_length
        self.cycles_per_worker = cycles_per_worker
        self.total_cycles = total_cycles
        self.replay_buffer = ReplayBuffer()

    def collect(self):
        cycles_collected = 0
        while cycles_collected < self.total_cycles:
            left = min(self.actor_num * self.cycles_per_worker, self.total_cycles - cycles_collected)
            per_worker = (left // self.actor_num) or 1
            processes = []
            result_queue = Queue()
            for _ in range(self.actor_num):
                p = Process(target=actor_worker, args=(self.env_class, self.sequence_length, result_queue, per_worker))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            # 결과 수집
            count = 0
            while True:
                try:
                    trajectory = result_queue.get_nowait()
                    self.replay_buffer.add(trajectory)
                    count += 1
                except Empty:
                    break
            print(f"Batch round finished: collected {count} trajectories.")
            cycles_collected += self.actor_num * per_worker  # 실제 batch마다 프로세스가 한번씩 종료/생성됨

    def get_buffer(self):
        return self.replay_buffer


def main():
    collector = BatchCollector(
        QWOPEnv,
        actor_num=2,              # 동시 워커 수: 학습 1, 수집 1, 워커 6 or 학습+수집 1 워커 6 여유 1
        sequence_length=50,
        cycles_per_worker=20000,     # 워커 1회 실행에서 몇 번까지 trajectory를 연속 수집할지 (메모리/누수 트레이드오프)
        total_cycles=20000          # 전체 수집할 trajectory 수 (원하는 만큼 반복)
    )
    collector.collect()
    print(f"ReplayBuffer에 저장된 trajectory 수: {len(collector.get_buffer())}")

    if len(collector.get_buffer()) > 0:
        sample_trajs = collector.get_buffer().sample(batch_size=min(5, len(collector.get_buffer())))
        print(f"예시 샘플 trajectory 길이: {len(sample_trajs[0])}")

if __name__ == "__main__":
    main()


# # BatchCollector.py (단일 워커 버전, batch/sequence 없음)

# import time
# from ReplayBuffer import ReplayBuffer
# from env import QWOPEnv

# def actor_worker(result_buffer, total_steps=20000, model_path="model.pth", reload_every=1000):
#     import torch

#     def load_policy(model_path):
#         # 실제 모델 로딩 코드 사용. 임시로 무작위 정책:
#         # model = torch.load(model_path, map_location='cpu')
#         # model.eval()
#         # return model
#         return None

#     def select_action(model, obs):
#         # 실제 모델 인퍼런스와 동일하게 작성
#         # obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
#         # with torch.no_grad():
#         #     action_probs = model(obs_tensor)
#         #     action = torch.argmax(action_probs, dim=1).item()
#         # return action
#         return 1  # 예시: 항상 액션 1 (실전에서는 모델의 인퍼런스 결과로 대체)

#     model = load_policy(model_path)

#     env = QWOPEnv()
#     obs = env.reset()
#     steps_collected = 0
#     episode_idx = 0
#     interval = 0.1
#     trigger_time = time.time()
#     t = 0
#     while steps_collected < total_steps:
#         try:
#             # 필요 시 주기적으로 모델을 reload (엄청 자주 할 필요 없음)
#             if reload_every > 0 and steps_collected > 0 and steps_collected % reload_every == 0:
#                 try:
#                     model = load_policy(model_path)
#                 except Exception as e:
#                     print(f"[actor_worker] 모델 로딩 예외 발생: {e}")

#             action = select_action(model, obs)
#             next_obs, reward, done, info = env.step(action)
#             now = time.time()
#             sleep_time = trigger_time + interval - now
#             if sleep_time > 0:
#                 time.sleep(sleep_time)
#             else:
#                 print(f"[actor_worker] Frame delayed by {abs(sleep_time):.4f} sec at cycle {episode_idx}, step {t}")
#             trigger_time += interval
#             t += 1
#             # Transition을 바로 저장
#             result_buffer.add((obs, action, reward, next_obs, done))

#             obs = next_obs
#             steps_collected += 1

#             if done:
#                 print(f"[actor_worker] Episode {episode_idx} 종료. 현재까지 step 수: {steps_collected}")
#                 obs = env.reset()
#                 trigger_time = time.time()
#                 episode_idx += 1

#         except Exception as e:
#             print(f"[actor_worker] 예외 발생: {e}. 환경 리셋 후 다음 에피소드 진행")
#             try:
#                 env.close()
#             except:
#                 pass
#             env = QWOPEnv()
#             obs = env.reset()
#             # (필요시 sleep 추가 가능)
#     env.close()
#     print(f"[actor_worker] 지정한 총 step({total_steps}) 수집 완료")

# class BatchCollector:
#     def __init__(self, env_class, total_steps=20000, model_path="model.pth", reload_every=1000):
#         self.env_class = env_class
#         self.total_steps = total_steps
#         self.model_path = model_path
#         self.reload_every = reload_every
#         self.replay_buffer = ReplayBuffer()

#     def collect(self):
#         actor_worker(self.replay_buffer, self.total_steps, self.model_path, self.reload_every)

#     def get_buffer(self):
#         return self.replay_buffer

# def main():
#     collector = BatchCollector(
#         QWOPEnv,
#         total_steps=20000,
#         model_path="model.pth",
#         reload_every=1000
#     )
#     collector.collect()
#     print(f"ReplayBuffer에 저장된 transition 수: {len(collector.get_buffer())}")

#     if len(collector.get_buffer()) > 0:
#         sample_trans = collector.get_buffer().sample(batch_size=min(5, len(collector.get_buffer())))
#         print(f"예시 샘플 transition: {sample_trans[0]}")

# if __name__ == "__main__":
#     main()
