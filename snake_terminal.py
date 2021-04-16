import atexit
import curses
import numpy
import os
import time
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from typing import Optional, Tuple
from Agent_Snake import choose_action
from snake import SnakeGame

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
        else:
            snake_terminal = SnakeTerminal(game, scr_win)

        snake_terminal.run_game()
    finally:
        cleanup()
        atexit.unregister(cleanup)


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

    KEY_MAP = {
        KEY_LEFT: SnakeGame.Move.LEFT,
        KEY_RIGHT: SnakeGame.Move.RIGHT,
        KEY_UP: SnakeGame.Move.UP,
        KEY_DOWN: SnakeGame.Move.DOWN,
    }

    def __init__(
        self,
        game: SnakeGame,
        scr_win: "curses._CursesWindow",
    ):
        self.game = game
        self.scr_win = scr_win
        self.paused = False
        self.quit = False
        self.seconds_per_move = 0.15

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

        self.info_win.addstr(4, 0, "Controls:")
        self.info_win.addstr(5, 0, f" {chr(self.KEY_QUIT):>5} quit game")
        self.info_win.addstr(6, 0, f" {pause_key_name:>5} pause game")

    def update_info_window(self):
        self.info_win.addstr(1, 0, f"Score: {self.game.score}")
        if self.game.game_over:
            self.info_win.addstr(2, 0, "GAME OVER")
        elif self.paused:
            self.info_win.addstr(2, 0, "PAUSED")

        self.info_win.refresh()

    def pause(self):
        self.game_win.nodelay(False)
        self.paused = True
        self.update_info_window()

        while True:
            key = self.game_win.getch()
            if key == -1:
                pass
            elif key == self.KEY_QUIT:
                self.quit = True
                break
            elif key == self.KEY_PAUSE and not self.game.game_over:
                break

        self.paused = False
        self.update_info_window()
        self.game_win.nodelay(True)

    def get_latest_key(self):
        latest = -1
        while True:
            key = self.game_win.getch()
            if key == -1:
                break
            latest = key
        return latest

    def advance_game(self, key: int):
        self.game.tick(new_direction=self.KEY_MAP.get(key))
        self.game_win.refresh()

        self.update_info_window()

        # update score text

    def run_game(self):
        while not self.quit:
            self.game_win.timeout(int(self.seconds_per_move * 1000))
            key = self.game_win.getch()

            if key == self.KEY_PAUSE:
                self.pause()
            elif key == self.KEY_QUIT:
                self.quit = True
            else:
                self.advance_game(key)

            if self.game.game_over:
                self.pause()


class SnakeTerminalWithModel(SnakeTerminal):

    GAME_ACTIONS = [
        (SnakeGame.Move.UP, "UP"),
        (SnakeGame.Move.DOWN, "DOWN"),
        (SnakeGame.Move.LEFT, "LEFT"),
        (SnakeGame.Move.RIGHT, "RIGHT"),
    ]

    KEY_SLOWER = ord("+")
    KEY_FASTER = ord("-")
    MIN_SECONDS_PER_MOVE = 0.1
    MAX_SECONDS_PER_MOVE = 1.6

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

        self.initialize_model_window()

    def initialize_model_window(self):

        self.model_win = curses.newwin(
            6, self.scr_win.getmaxyx()[1], self.game_win.getmaxyx()[0] + 1, 0
        )

        self.update_next_action()

    def initialize_info_window(self):
        super().initialize_info_window()

        self.info_win.addstr(7, 0, f" {chr(self.KEY_FASTER):>5} increase speed")
        self.info_win.addstr(8, 0, f" {chr(self.KEY_SLOWER):>5} decrease speed")

    def update_next_action(self):
        self.next_action = self.GAME_ACTIONS[
            choose_action(self.model, self.game.get_board())
        ]

        self.update_model_window()

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

    def advance_game(self, key: int):

        if key == self.KEY_SLOWER:
            self.seconds_per_move = max(
                self.MIN_SECONDS_PER_MOVE, self.seconds_per_move / 2
            )
            time.sleep(0.1)  # short delay to prevent zipping through all the speeds
        elif key == self.KEY_FASTER:
            self.seconds_per_move = min(
                self.MAX_SECONDS_PER_MOVE, self.seconds_per_move * 2
            )
            time.sleep(0.1)
        else:
            self.game.tick(new_direction=self.next_action[0])
            self.game_win.refresh()

            self.update_next_action()
            self.update_info_window()

        self.last_key = key


if __name__ == "__main__":
    the_game = SnakeGame(10, 10, 3)
    play_game(the_game)