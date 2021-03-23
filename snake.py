import atexit
import collections
import curses
import enum
import itertools
import numpy
import random
import time
from enum import Enum
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from typing import NamedTuple, Tuple

from Agent_Snake import choose_action

# Game adapted from https://gist.github.com/sanchitgangwar/2158089


class SnakeGame:

    BLANK = 0
    HEAD = 0.5
    BODY = 0.25
    FRUIT = 1

    class SnakePartType(Enum):
        HEAD_UP = enum.auto()
        HEAD_RIGHT = enum.auto()
        HEAD_LEFT = enum.auto()
        HEAD_DOWN = enum.auto()
        BODY_LEFT_RIGHT = enum.auto()
        BODY_DOWN_UP = enum.auto()
        BODY_LEFT_DOWN = enum.auto()
        BODY_LEFT_UP = enum.auto()
        BODY_RIGHT_DOWN = enum.auto()
        BODY_RIGHT_UP = enum.auto()

    class SnakePart(NamedTuple):
        pos: Tuple[int, int]
        part: "SnakeGame.SnakePartType"

    class Move(Enum):
        UP = (-1, 0)
        RIGHT = (0, 1)
        DOWN = (1, 0)
        LEFT = (0, -1)

        def __add__(self, other):
            return (self.value[0] + other.value[0], self.value[1] + other.value[1])

        def __sub__(self, other):
            return (self.value[0] + other.value[0], self.value[1] + other.value[1])

    def __init__(self, width, height, num_fruit=1):

        assert width >= 3 and height >= 3
        assert num_fruit > 0

        self.width = width
        self.height = height
        self.board = numpy.zeros((self.height, self.width))
        self.snake = collections.deque()  # left end is head, right end is tail
        self.snake_direction = SnakeGame.Move.RIGHT
        self.fruits = []
        self.score = 0
        self.just_ate_fruit = False
        self.game_over = False
        self.on_new_fruit = None
        self.on_snake_move = None

        # snake initial position
        center_row = self.height // 2
        center_col = self.width // 2
        initial_snake_parts = [
            SnakeGame.SnakePartType.BODY_LEFT_RIGHT,
            SnakeGame.SnakePartType.BODY_LEFT_RIGHT,
            SnakeGame.SnakePartType.HEAD_RIGHT,
        ]
        for j, part in zip([-1, 0, 1], initial_snake_parts):
            pos = (center_row, center_col + j)
            self.snake.appendleft(SnakeGame.SnakePart(pos, part))
            self.board[pos] = (
                self.HEAD if part == SnakeGame.SnakePartType.HEAD_RIGHT else self.BODY
            )

        for i in range(num_fruit):
            self.spawn_fruit()

    def spawn_fruit(self):
        pos = None
        while pos is None or self.board[pos] != SnakeGame.BLANK:
            pos = (random.randrange(self.height), random.randrange(self.width))

        self.board[pos] = SnakeGame.FRUIT
        self.fruits.append(pos)

        if self.on_new_fruit is not None:
            self.on_new_fruit(pos)

    def tick(self, new_direction):
        if self.game_over:
            return

        self.just_ate_fruit = False
        prev_direction = self.snake_direction
        if new_direction is not None:
            # if (
            #     new_direction.value[0] != -prev_direction.value[0]
            #     or new_direction.value[1] != -prev_direction.value[1]
            # ):
            self.snake_direction = new_direction

        old_head = self.snake.popleft()
        new_head_pos = (
            (old_head.pos[0] + self.snake_direction.value[0]),  # % self.height,
            (old_head.pos[1] + self.snake_direction.value[1]),  # % self.width,
        )

        if not (
            0 <= new_head_pos[0] < self.height and 0 <= new_head_pos[1] < self.width
        ):
            self.game_over = True
            return

        old_head_new_part = self.get_part_type(self.snake_direction, prev_direction)
        old_head = SnakeGame.SnakePart(old_head.pos, old_head_new_part)
        new_head = SnakeGame.SnakePart(
            new_head_pos, self.get_head_type(self.snake_direction)
        )

        self.snake.appendleft(old_head)
        self.snake.appendleft(new_head)

        if self.board[new_head_pos] == SnakeGame.BODY:
            # checking body hit is sufficient, head cannot hit itself
            self.game_over = True
            return
        elif self.board[new_head_pos] == SnakeGame.FRUIT:
            self.score += 1
            self.just_ate_fruit = True
            self.fruits.remove(new_head_pos)
            self.spawn_fruit()
            # old_tail = None
            old_tail = self.snake.pop()
            self.board[old_tail.pos] = SnakeGame.BLANK
        else:
            # no game over, no fruit hit. snake tail moves
            old_tail = self.snake.pop()
            self.board[old_tail.pos] = SnakeGame.BLANK

        # move head
        self.board[old_head.pos] = SnakeGame.BODY
        self.board[new_head_pos] = SnakeGame.HEAD

        if self.on_snake_move is not None:
            self.on_snake_move(new_head, old_head, old_tail)

    @classmethod
    def get_head_type(cls, direction):
        if direction == SnakeGame.Move.UP:
            return SnakeGame.SnakePartType.HEAD_UP
        elif direction == SnakeGame.Move.RIGHT:
            return SnakeGame.SnakePartType.HEAD_RIGHT
        elif direction == SnakeGame.Move.DOWN:
            return SnakeGame.SnakePartType.HEAD_DOWN
        else:
            return SnakeGame.SnakePartType.HEAD_LEFT

    @classmethod
    def get_part_type(cls, head_direction, old_head_direction):
        ahead_diff = head_direction.value
        behind_diff = old_head_direction.value
        if ahead_diff == behind_diff:
            if ahead_diff[0] == 0:
                return cls.SnakePartType.BODY_LEFT_RIGHT
            return cls.SnakePartType.BODY_DOWN_UP

        diff_change = (ahead_diff[0] - behind_diff[0], ahead_diff[1] - behind_diff[1])
        if diff_change == (1, 1):
            return cls.SnakePartType.BODY_RIGHT_DOWN
        elif diff_change == (1, -1):
            return cls.SnakePartType.BODY_LEFT_DOWN
        elif diff_change == (-1, 1):
            return cls.SnakePartType.BODY_RIGHT_UP
        elif diff_change == (-1, -1):
            return cls.SnakePartType.BODY_LEFT_UP


def play_game(game, model=None):
    def cleanup():
        curses.endwin()

    atexit.register(cleanup)

    curses.initscr()
    # +2 for the borders
    win = curses.newwin(game.height + 2, game.width + 2, 0, 0)

    try:
        play_game_helper(game, win, model)
    finally:
        cleanup()
        atexit.unregister(cleanup)


def play_game_helper(game, win, model=None):

    win.keypad(True)  # interpret escape sequences (in particular arrow keys)
    curses.noecho()  # don't echo input characters
    curses.curs_set(0)  # invisible cursor
    win.border(0)
    win.nodelay(True)  # make getch non-blocking

    char_map = {
        SnakeGame.HEAD: "o",
        SnakeGame.BODY: "+",
        SnakeGame.FRUIT: "#",
        SnakeGame.BLANK: " ",
        SnakeGame.SnakePartType.HEAD_UP: "∧",
        SnakeGame.SnakePartType.HEAD_LEFT: "<",
        SnakeGame.SnakePartType.HEAD_RIGHT: ">",
        SnakeGame.SnakePartType.HEAD_DOWN: "∨",
        SnakeGame.SnakePartType.BODY_LEFT_RIGHT: "═",
        SnakeGame.SnakePartType.BODY_DOWN_UP: "║",
        SnakeGame.SnakePartType.BODY_LEFT_DOWN: "╗",
        SnakeGame.SnakePartType.BODY_LEFT_UP: "╝",
        SnakeGame.SnakePartType.BODY_RIGHT_DOWN: "╔",
        SnakeGame.SnakePartType.BODY_RIGHT_UP: "╚",
    }

    def draw_char(pos, c):
        # shift indices since border takes up the first row and column
        win.addch(pos[0] + 1, pos[1] + 1, char_map[c])

    def draw_part(part):
        draw_char(part.pos, part.part)

    def on_new_fruit(fruit):
        draw_char(fruit, SnakeGame.FRUIT)

    def on_snake_move(new_head, old_head, old_tail):
        draw_part(new_head)
        draw_part(old_head)
        if old_tail is not None:
            draw_char(old_tail.pos, SnakeGame.BLANK)

    # register event handlers
    game.on_new_fruit = on_new_fruit
    game.on_snake_move = on_snake_move

    # initial board
    draw_part(game.snake[0])
    for body in itertools.islice(game.snake, 1, None):
        draw_part(body)
    for fruit in game.fruits:
        draw_char(fruit, SnakeGame.FRUIT)

        KEY_ESC = 27
        KEY_SPACE = ord(" ")
        key_map = {
            KEY_LEFT: SnakeGame.Move.LEFT,
            KEY_RIGHT: SnakeGame.Move.RIGHT,
            KEY_UP: SnakeGame.Move.UP,
            KEY_DOWN: SnakeGame.Move.DOWN,
        }
        is_paused = False
        key = None
        while key != KEY_ESC:
            win.border(0)
            win.timeout(100)
            win.addstr(0, 2, f"Score : {game.score} ")
            if game.game_over:
                win.addstr(game.height + 1, 2, "Game over")
            elif is_paused:
                win.addstr(game.height + 1, 2, "Paused")

            if model and not game.game_over:
                game_actions = [
                    SnakeGame.Move.UP,
                    SnakeGame.Move.DOWN,
                    SnakeGame.Move.LEFT,
                    SnakeGame.Move.RIGHT,
                ]
                game.tick(game_actions[choose_action(model, game.board)])
                event = win.getch()
            else:
                event = win.getch()
                key = None if event == -1 else event

                # pause if space bar is pressed
                if key == KEY_SPACE:
                    is_paused = not is_paused
                    continue

                if key == KEY_ESC:
                    break

                if key not in key_map:
                    key = None

                if not game.game_over and not is_paused:
                    game.tick(new_direction=key_map.get(key))


if __name__ == "__main__":
    game = SnakeGame(20, 13)
    play_game(game)