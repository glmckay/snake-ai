import atexit
import curses
import numpy
import os
import time
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from typing import Optional, Tuple
from Agent_Snake import choose_action
from snake import SnakeGame
from game_options import game_options

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402

# Game adapted from https://gist.github.com/sanchitgangwar/2158089


def play_game(game: SnakeGame, model: Optional["tf.keras.Model"] = None):
    def cleanup():
        curses.endwin()

    atexit.register(cleanup)

    scr_win = curses.initscr()
    curses.noecho()  # don't echo input characters
    curses.curs_set(0)  # invisible cursor

    try:
        if model:
            snake_terminal = SnakeTerminalWithModel(game, scr_win, model)
            snake_terminal.run_state_machine(initial_state="PAUSED")
        else:
            snake_terminal = SnakeTerminal(game, scr_win)
            snake_terminal.run_state_machine(initial_state="PLAY")

    finally:
        cleanup()
        atexit.unregister(cleanup)


def clamp_value(m, v, M):
    return max(m, min(v, M))


class SnakeTerminal:

    CHAR_MAP = {
        SnakeGame.BoardElement.BLANK: " ",
        SnakeGame.BoardElement.FRUIT: "#",
        SnakeGame.BoardElement.HEAD_UP: "∧",
        SnakeGame.BoardElement.HEAD_LEFT: "<",
        SnakeGame.BoardElement.HEAD_RIGHT: ">",
        SnakeGame.BoardElement.HEAD_DOWN: "∨",
        SnakeGame.BoardElement.BODY_LEFT_RIGHT: "═",
        SnakeGame.BoardElement.BODY_DOWN_UP: "║",
        SnakeGame.BoardElement.BODY_LEFT_DOWN: "╗",
        SnakeGame.BoardElement.BODY_LEFT_UP: "╝",
        SnakeGame.BoardElement.BODY_RIGHT_DOWN: "╔",
        SnakeGame.BoardElement.BODY_RIGHT_UP: "╚",
    }

    KEY_QUIT = ord("q")
    KEY_PAUSE = ord(" ")  # space
    KEY_SLOWER = ord("-")
    KEY_FASTER = ord("+")
    MIN_SECONDS_PER_MOVE = 0.1
    MAX_SECONDS_PER_MOVE = 10

    KEY_MAP = {
        ord("a"): SnakeGame.Move.LEFT,
        ord("d"): SnakeGame.Move.RIGHT,
        ord("w"): SnakeGame.Move.UP,
        ord("s"): SnakeGame.Move.DOWN,
    }

    def __init__(
        self,
        game: SnakeGame,
        scr_win: "curses._CursesWindow",
    ):
        self.game = game
        self.scr_win = scr_win
        self.quit = False
        self.seconds_per_move = 1

        # register event handlers
        self.game.on_new_fruit = lambda *args: self.on_new_fruit(*args)
        self.game.on_snake_move = lambda *args: self.on_snake_move(*args)

        self.initialize_game_window()
        self.initialize_info_window()

    def draw_element(self, pos: Tuple[int, int], c: "SnakeGame.BoardElement"):
        # shift indices since border takes up the first row and column
        self.game_win.addch(pos[0] + 1, pos[1] + 1, self.CHAR_MAP[c])

    def draw_part(self, part: "SnakeGame.SnakePart"):
        self.draw_element(part.pos, part.part)

    def on_new_fruit(self, fruit: Tuple[int, int]):
        self.draw_element(fruit, SnakeGame.BoardElement.FRUIT)

    def on_snake_move(
        self,
        new_head: "SnakeGame.SnakePart",
        old_head: "SnakeGame.SnakePart",
        old_tail: Optional["SnakeGame.SnakePart"],
    ):
        self.draw_part(new_head)
        self.draw_part(old_head)
        if old_tail is not None:
            self.draw_element(old_tail.pos, SnakeGame.BoardElement.BLANK)

    def initialize_game_window(self):

        # +2 for the borders
        self.game_win = curses.newwin(self.game.height + 2, self.game.width + 2, 0, 0)
        self.game_win.keypad(
            True
        )  # interpret escape sequences (in particular arrow keys)
        self.game_win.border(0)
        self.game_win.nodelay(True)  # make getch non-blocking

        # Draw the current board
        snake_iter = iter(self.game.snake)
        # head
        self.draw_part(next(snake_iter))
        # rest of snake
        for body in snake_iter:
            self.draw_part(body)
        # fruits
        for fruit in self.game.fruits:
            self.draw_element(fruit, SnakeGame.BoardElement.FRUIT)

    def initialize_info_window(self):
        height = self.game_win.getmaxyx()[0]
        offset_x = self.game_win.getmaxyx()[1] + 2
        width = self.scr_win.getmaxyx()[1] - offset_x
        self.info_win = curses.newwin(height, width, 0, offset_x)

        pause_key_name = "space" if self.KEY_PAUSE == ord(" ") else chr(self.KEY_PAUSE)

        self.info_win.addstr(5, 0, "Controls:")
        self.info_win.addstr(6, 0, f" {chr(self.KEY_QUIT):>5} quit game")
        self.info_win.addstr(7, 0, f" {pause_key_name:>5} pause game")

    def update_info_window(self):
        self.info_win.addstr(1, 0, f"Score: {self.game.score}")
        if self.game.game_over:
            self.info_win.addstr(2, 0, "GAME OVER")
        elif self.state == "PAUSED":
            self.info_win.addstr(2, 0, "PAUSED")

        self.info_win.refresh()

    def get_next_key(self, timeout: Optional[float] = None):
        self.game_win.nodelay(timeout is not None)
        return self.game_win.getch()

    def advance_game(self, action: Optional["SnakeGame.Move"]):
        self.game.tick(action=action)
        self.game_win.refresh()

        self.update_info_window()

    def update_play_speed(self, key: int):        
        old_delay = self.seconds_per_move
        self.seconds_per_move = clamp_value(
            self.MIN_SECONDS_PER_MOVE,
            self.seconds_per_move * (0.5 if key == self.KEY_FASTER else 2),
            self.MAX_SECONDS_PER_MOVE,
        )
        self.update_info_window()
        return old_delay - self.seconds_per_move

    def run_state_machine(self, initial_state):
        self.state = initial_state
        while self.state != "QUIT":
            self.update_info_window()
            if self.state == "PLAY":
                self.state = self.play()
            elif self.state == "PAUSED":
                self.state = self.paused()
            else:
                raise ValueError(f"Unknown state '{self.state}'")

    def paused(self):
        while True:
            key = self.get_next_key()
            if key == self.KEY_QUIT:
                return "QUIT"            
            elif key in [self.KEY_FASTER, self.KEY_SLOWER]:
                self.seconds_per_move = self.update_play_speed(key)
            elif key == self.KEY_PAUSE and not self.game.game_over:
                return "PLAY"

    def play(self):
        while True:
            key = self.get_next_key(self.seconds_per_move)

            if key == self.KEY_QUIT:
                return "QUIT"
            elif key == self.KEY_PAUSE:
                return "PAUSED"            
            elif key in [self.KEY_FASTER, self.KEY_SLOWER]:
                self.seconds_per_move = self.update_play_speed(key)
            else:
                self.advance_game(self.KEY_MAP.get(key))

            if self.game.game_over:
                return "PAUSED"


class SnakeTerminalWithModel(SnakeTerminal):

    GAME_ACTIONS = [
        (SnakeGame.Move.UP, "UP"),
        (SnakeGame.Move.DOWN, "DOWN"),
        (SnakeGame.Move.LEFT, "LEFT"),
        (SnakeGame.Move.RIGHT, "RIGHT"),
    ]

    KEY_SLOWER = ord("o")
    KEY_FASTER = ord("p")
    MIN_SECONDS_PER_MOVE = 0.1
    MAX_SECONDS_PER_MOVE = 10

    def __init__(
        self,
        game: SnakeGame,
        scr_win: "curses._CursesWindow",
        model: "tf.keras.Model",
    ):
        super().__init__(game, scr_win)
        self.model = model
        self.next_action = None
        self.last_key = None
        self.seconds_per_move = 1

        self.initialize_model_window()

    def initialize_model_window(self):

        self.model_win = curses.newwin(
            6, self.scr_win.getmaxyx()[1], self.game_win.getmaxyx()[0] + 1, 0
        )

        self.update_next_action()

    def initialize_info_window(self):
        super().initialize_info_window()

        self.info_win.addstr(8, 0, f" {chr(self.KEY_FASTER):>5} increase speed")
        self.info_win.addstr(9, 0, f" {chr(self.KEY_SLOWER):>5} decrease speed")

    def update_info_window(self):
        self.info_win.addstr(3, 0, f" move delay: {self.seconds_per_move}s")
        super().update_info_window()

    def update_model_window(self):

        self.model_win.erase()
        # line to separate from game area
        self.model_win.addstr(0, 0, "─" * self.model_win.getmaxyx()[1])

        board = self.game.get_board()
        logits = self.model(numpy.expand_dims(board, axis=0))[0]

        weights = numpy.exp(logits)  # logits === log probabilities?
        total_weights = sum(weights)

        for i, w, l in zip(range(4), weights, logits):
            action_i = self.GAME_ACTIONS[i]
            if action_i == self.next_action:
                move_str = f">{action_i[1]}<"
            else:
                move_str = f"{action_i[1]} "  # ending space to match potential '<'

            line = f"{move_str:>7} [{'#' * int(20 * w / total_weights):<20}] {l}"
            self.model_win.addstr(2 + i, 0, line)

        self.model_win.refresh()

    def update_next_action(self):
        self.next_action = self.GAME_ACTIONS[
            choose_action(self.model, self.game.get_board())
        ]
        self.update_model_window()

    def advance_game(self, action: Optional["SnakeGame.Move"]):
        super().advance_game(action)
        if not self.game.game_over:
            self.update_next_action()

    def update_play_speed(self, key: int):
        old_delay = self.seconds_per_move
        self.seconds_per_move = clamp_value(
            self.MIN_SECONDS_PER_MOVE,
            self.seconds_per_move * (0.5 if key == self.KEY_FASTER else 2),
            self.MAX_SECONDS_PER_MOVE,
        )
        self.update_info_window()
        return old_delay - self.seconds_per_move

    def play(self):
        next_move_time = -1
        while True:
            now = time.thread_time()
            if next_move_time <= now:
                self.advance_game(self.next_action[0])
                if self.game.game_over:
                    return "PAUSED"
                self.update_next_action()
                next_move_time = now + self.seconds_per_move

            key = self.get_next_key(int(next_move_time - now * 1000))

            if key == self.KEY_QUIT:
                return "QUIT"
            elif key == self.KEY_PAUSE:
                return "PAUSED"
            elif key in [self.KEY_FASTER, self.KEY_SLOWER]:
                next_move_time += self.update_play_speed(key)

    def paused(self):
        while True:
            key = self.get_next_key()
            if key == self.KEY_QUIT:
                return "QUIT"
            elif key == self.KEY_PAUSE and not self.game.game_over:
                return "PLAY"
            elif key in [self.KEY_FASTER, self.KEY_SLOWER]:
                self.update_play_speed(key)
            elif key in self.KEY_MAP and not self.game.game_over:
                self.advance_game(self.KEY_MAP[key])

if __name__ == "__main__":
    the_game = SnakeGame(game_options["width"], game_options["height"], game_options["num_fruits"])
    play_game(the_game)