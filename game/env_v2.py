import time
import uuid

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from stable_baselines3.common.env_checker import check_env

PRESS_DURATION = 0.1
MAX_EPISODE_DURATION_SECS = 120
STATE_SPACE_N = 71
KEYS = ['q', 'w', 'o', 'p']


class QWOPEnv(gym.Env):

    meta_data = {'render.modes': ['human']}

    def __init__(self, port, render_mode=None):

        # Open AI gym specifications
        super(QWOPEnv, self).__init__()
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[STATE_SPACE_N], dtype=np.float32
        )
        self.num_envs = 1

        # QWOP specific stuff
        self.port = port
        self.gameover = False
        self.previous_score = 0
        self.previous_time = 0
        self.previous_torso_x = 0
        self.previous_torso_y = 0
        self.previous_head_y = 0
        self.evoke_actions = True
        self.pressed_keys = set()

        # Open browser and go to QWOP page
        options = webdriver.ChromeOptions()
        
        # 자동화/컨테이너 환경에서 안정적인 실행을 위한 옵션
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        
        if render_mode != 'human':
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu') # GPU 하드웨어 가속 비활성화
            options.add_argument('--disable-extensions') # 확장 프로그램 비활성화
            options.add_argument('--disable-notifications') # 알림 비활성화
            options.add_argument('--disable-popup-blocking') # 팝업 차단 비활성화
            options.add_argument('--disable-backgrounding-occluded-windows') # 백그라운드 탭 비활성화
            options.add_argument('--incognito') # 시크릿 모드 (일부 오버헤드 감소)
            options.add_argument('--log-level=3') # 브라우저 로그 레벨 최소화 (SEVERE만 표시)
            options.add_argument('blink-settings=imagesEnabled=false') # 이미지 로딩 비활성화 (QWOP 게임에 필요 없다면)
            options.add_argument('--mute-audio') # 오디오 음소거 (QWOP 게임에 필요 없다면)
            options.add_argument('--window-size=800,600') # 창 크기 지정 (더 작은 렌더링 영역)
        
        # 충돌을 피하기 위해 고유한 사용자 데이터 디렉토리 지정
        user_data_dir = f"/tmp/chrome-user-data-{uuid.uuid4()}"
        options.add_argument(f"--user-data-dir={user_data_dir}")

        self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        self.driver.get(f'http://localhost:{self.port}/Athletics.html')

        # Wait until the game is running
        print("Waiting for game to load...")
        start_time = time.time()
        while True:
            try:
                # Check if the globalgamestate object exists and is not null
                game_state = self._get_variable_('globalgamestate')
                if game_state and game_state.get('gameEnded') is not None: # 더 구체적인 조건 확인
                    print("Game loaded.")
                    break
            except Exception as e:
                # Game might not be initialized yet, ignore and retry
                pass
            if time.time() - start_time > 30: # 30초 타임아웃
                raise RuntimeError("Timeout waiting for game to load.")
            time.sleep(0.5)

        # Wait a bit and then start game by clicking
        print("Focusing and clicking the game window to start.")
        time.sleep(2) # 사용자가 제안한 대기 시간
        try:
            self.driver.execute_script("window.focus();")
            self.body = self.driver.find_element(By.XPATH, "//body")
            ActionChains(self.driver).click(self.body).perform()
            print("Game window clicked.")
        except Exception as e:
            print(f"Warning: Could not click the game window: {e}")


        self.last_press_time = time.time()

    def _get_variable_(self, var_name):
        # Headless 모드에서 JavaScript 변수가 로드될 시간을 주기 위해 짧게 대기
        time.sleep(0.01) 
        result = self.driver.execute_script(f'return typeof {var_name} === "undefined" ? null : {var_name};')
        
        # None 또는 dict가 아닌 경우, 게임이 아직 준비되지 않았을 수 있음
        if result is None:
            # print(f"DEBUG: _get_variable_ - {var_name} is undefined.")
            return None
        if not isinstance(result, dict):
            # print(f"DEBUG: _get_variable_ - {var_name} returned non-dict: {type(result)}. Returning None.")
            return None
            
        # print(f"DEBUG: _get_variable_ - {var_name} returned: {result}")
        return result

    def _get_state_(self):
        game_state = self._get_variable_('globalgamestate')
        body_state = self._get_variable_('globalbodystate')

        # Ensure game_state and body_state are valid dictionaries and body_state is not empty
        if game_state is None or body_state is None or not body_state:
            # If states are not valid or body_state is empty, return a zero-filled array and mark as done
            return np.zeros(STATE_SPACE_N, dtype=np.float32), 0, True, {}

        # Get done
        if (
            (game_state.get('gameEnded', 0) > 0)
            or (game_state.get('gameOver', 0) > 0)
            or (game_state.get('scoreTime', 0) > MAX_EPISODE_DURATION_SECS)
        ):
            self.gameover = done = True
        else:
            self.gameover = done = False

        # Get reward
        torso_x = body_state.get('torso', {}).get('position_x', self.previous_torso_x)
        torso_y = body_state.get('torso', {}).get('position_y', self.previous_torso_y)

        head_y = body_state.get('head', {}).get('position_y', self.previous_head_y)

        # Reward for moving forward
        reward1 = max(torso_x - self.previous_torso_x, 0)
        reward2 = (head_y + 4) * (-0.2)

        # Combine rewards
        reward = reward1 + reward2

        # Update previous scores
        self.previous_torso_x = torso_x
        self.previous_torso_y = torso_y
        self.previous_head_y = head_y
        self.previous_score = game_state.get('score', 0)
        self.previous_time = game_state.get('scoreTime', 0)
        # Normalize torso_x
        for part_name, values in body_state.items():
            if isinstance(values, dict) and 'position_x' in values:
                values['position_x'] -= torso_x

        # Initialize state as a zero-filled numpy array
        state = np.zeros(STATE_SPACE_N, dtype=np.float32)
        idx = 0

        # Process main body parts
        for part_name in ['torso', 'head', 'left_upper_leg', 'left_lower_leg', 'right_upper_leg', 'right_lower_leg']:
            part = body_state.get(part_name, {})
            if idx + 6 <= STATE_SPACE_N:
                state[idx] = part.get('position_x', 0)
                state[idx+1] = part.get('position_y', 0)
                state[idx+2] = part.get('velocity_x', 0)
                state[idx+3] = part.get('velocity_y', 0)
                state[idx+4] = part.get('angle', 0)
                state[idx+5] = part.get('angular_velocity', 0)
                idx += 6
            else:
                break

        # Add joint angles and velocities if available
        for i in range(5): # Adjust range based on actual number of joints
            if idx + 2 <= STATE_SPACE_N:
                joint_angle = body_state.get(f'joint{i}_angle', 0)
                joint_angular_velocity = body_state.get(f'joint{i}_angular_velocity', 0)
                state[idx] = joint_angle
                state[idx+1] = joint_angular_velocity
                idx += 2
            else:
                break

        return state, reward, done, {}

    def send_keys(self, keys):
        keys_to_press = set(keys)

        keys_to_release = self.pressed_keys - keys_to_press
        new_keys_to_press = keys_to_press - self.pressed_keys

        action = ActionChains(self.driver)
        for key in keys_to_release:
            action.key_up(key)
        for key in new_keys_to_press:
            action.key_down(key)
        action.perform()

        self.pressed_keys = keys_to_press
        time.sleep(PRESS_DURATION)

    def reset(self, seed=None, options=None):
        # Release any currently pressed keys
        if self.pressed_keys:
            action = ActionChains(self.driver)
            for key in self.pressed_keys:
                action.key_up(key)
            action.perform()
            self.pressed_keys.clear()

        # Send 'R' and SPACE key press to restart game
        action = ActionChains(self.driver)
        action.key_down('r').key_down(Keys.SPACE).pause(PRESS_DURATION).key_up('r').key_up(Keys.SPACE).perform()

        # Wait until the game is running again and body state is available
        start_time = time.time()
        while True:
            try:
                game_state = self._get_variable_('globalgamestate')
                body_state = self._get_variable_('globalbodystate') # Get body state here
                if (
                    game_state is not None
                    and not game_state.get('gameOver')
                    and body_state is not None
                    and 'torso' in body_state
                    and 'head' in body_state
                ):
                    break
            except Exception as e:
                pass
            if time.time() - start_time > 10: # 10초 타임아웃
                print("Warning: Timeout waiting for game to restart and body state to be ready.")
                break
            time.sleep(0.1)

        self.gameover = False
        self.previous_score = 0
        self.previous_time = 0
        self.previous_torso_x = 0
        self.previous_torso_y = 0
        
        # Click to ensure focus, especially in headless mode
        try:
            self.driver.execute_script("window.focus();")
            ActionChains(self.driver).click(self.body).perform()
        except Exception as e:
            print(f"Warning: Could not click the game window during reset: {e}")


        # Add a small delay to ensure the game state is fully updated
        time.sleep(0.5)

        state, _, _, _ = self._get_state_()
        print(f"DEBUG: reset - State shape before return: {state.shape}")
        return state, {}

    def step(self, action):

        # send action
        keys_to_press = []
        binary_action = f'{action:04b}'
        for i in range(4):
            if binary_action[i] == '1':
                keys_to_press.append(KEYS[i])

        if self.evoke_actions:
            self.send_keys(keys_to_press)
        # else:
        #     time.sleep(PRESS_DURATION)

        state, reward, done, _ = self._get_state_()
        return state, reward, done, False, {}

    def render(self, mode='headless'):
        pass

    def close(self):
        self.driver.quit()


def make_env(port, render_mode=None):
    def _init():
        env = QWOPEnv(port=port, render_mode=render_mode)
        return env
    return _init
