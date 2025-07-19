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
ACTIONS = {
    0: 'qw',
    1: 'qo',
    2: 'qp',
    3: 'q',
    4: 'wo',
    5: 'wp',
    6: 'w',
    7: 'op',
    8: 'o',
    9: 'p',
    10: '',
}


class QWOPEnv(gym.Env):

    meta_data = {'render.modes': ['human']}

    def __init__(self, port, render_mode=None):

        # Open AI gym specifications
        super(QWOPEnv, self).__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
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
        
        # 충돌을 피하기 위해 고유한 사용자 데이터 디렉토리 지정
        user_data_dir = f"/tmp/chrome-user-data-{uuid.uuid4()}"
        options.add_argument(f"--user-data-dir={user_data_dir}")

        self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        self.driver.get(f'http://localhost:{self.port}/Athletics.html')

        # Wait a bit and then start game
        time.sleep(2)
        self.body = self.driver.find_element(By.XPATH, "//body")
        self.body.click()

        self.last_press_time = time.time()

    def _get_variable_(self, var_name):
        return self.driver.execute_script(f'return {var_name};')

    def _get_state_(self):

        game_state = self._get_variable_('globalgamestate')
        body_state = self._get_variable_('globalbodystate')

        # Get done
        if (
            (game_state['gameEnded'] > 0)
            or (game_state['gameOver'] > 0)
            or (game_state['scoreTime'] > MAX_EPISODE_DURATION_SECS)
        ):
            self.gameover = done = True
        else:
            self.gameover = done = False

        # Get reward
        torso_x = body_state['torso']['position_x']
        torso_y = body_state['torso']['position_y']

        head_y = body_state['head']['position_y']

        # Reward for moving forward
        reward1 = max(torso_x - self.previous_torso_x, 0)
        reward2 = min(head_y - self.previous_head_y, 0)

        # Combine rewards
        reward = reward1 + reward2

        # Update previous scores
        self.previous_torso_x = torso_x
        self.previous_torso_y = torso_y
        self.previous_head_y = head_y
        self.previous_score = game_state['score']
        self.previous_time = game_state['scoreTime']

        # Normalize torso_x
        for part, values in body_state.items():
            if 'position_x' in values:
                values['position_x'] -= torso_x

        # Convert body state
        state = []
        for part in body_state.values():
            state = state + list(part.values())
        state = np.array(state)
        # print(f"state: {state}")
        # print(f"reward: {reward}")
        # print(f"done: {done}")
        # if done:
        #     print(f"distance: {torso_x}")
        #     time.sleep(PRESS_DURATION*10)

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

        self.gameover = False
        self.previous_score = 0
        self.previous_time = 0
        self.previous_torso_x = 0
        self.previous_torso_y = 0
        self.body.click()

        state, _, _, _ = self._get_state_()
        return state, {}

    def step(self, action_id):

        # send action
        if isinstance(action_id, np.ndarray):
            action_id = action_id.item()  # NumPy 배열에서 스칼라 값 추출
        keys = ACTIONS[action_id]

        if self.evoke_actions:
            self.send_keys(list(keys))
        # else:
        #     time.sleep(PRESS_DURATION)

        state, reward, done, _ = self._get_state_()
        return state, reward, done, False, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.driver.quit()


def make_env(port, render_mode=None):
    def _init():
        env = QWOPEnv(port=port, render_mode=render_mode)
        return env
    return _init
