import atexit
import collections
import curses
import itertools
import numpy
import random
from enum import Enum
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN

# Game adapted from https://gist.github.com/sanchitgangwar/2158089


class SnakeGame:

    BLANK = 0
    HEAD = 0.5
    BODY = 0.25
    FRUIT = 1

    class Move(Enum):
        UP = (-1, 0)
        RIGHT = (0, 1)
        DOWN = (1, 0)
        LEFT = (0, -1)

    def __init__(self, width, height, on_new_fruit=None, on_snake_move=None):

        assert width >= 3 and height >= 3

        self.width = width
        self.height = height
        self.board = numpy.zeros((self.height, self.width))
        self.snake = collections.deque()  # left end is head, right end is tail
        self.snake_direction = SnakeGame.Move.RIGHT.value
        self.fruits = []
        self.score = 0
        self.game_over = False
        self.on_new_fruit = on_new_fruit
        self.on_snake_move = on_snake_move

        # snake initial position
        center_row = self.height // 2
        center_col = self.width // 2
        for j, part in zip([-1, 0, 1], [self.BODY, self.BODY, self.HEAD]):
            pos = (center_row, center_col + j)
            self.snake.appendleft(pos)
            self.board[pos] = part

        self.spawn_fruit()

    def spawn_fruit(self):
        pos = None
        while pos is None or self.board[pos] != SnakeGame.BLANK:
            pos = (random.randrange(self.height), random.randrange(self.width))

        self.board[pos] = SnakeGame.FRUIT
        self.fruits.append(pos)

        if self.on_new_fruit is not None:
            self.on_new_fruit(pos)

    def move_snake(self):
        old_head = self.snake[0]
        new_head = (
            (old_head[0] + self.snake_direction[0]) % self.height,
            (old_head[1] + self.snake_direction[1]) % self.width,
        )

        self.snake.appendleft(new_head)

        if self.board[new_head] == SnakeGame.BODY:
            # checking body hit is sufficient, head cannot hit itself
            self.game_over = True
            return
        elif self.board[new_head] == SnakeGame.FRUIT:
            self.score += 1
            self.fruits.remove(new_head)
            self.spawn_fruit()
            old_tail = None
        else:
            # no game over, no fruit hit. snake tail moves
            old_tail = self.snake.pop()
            self.board[old_tail] = SnakeGame.BLANK

        # move head
        self.board[old_head] = SnakeGame.BODY
        self.board[new_head] = SnakeGame.HEAD

        if self.on_snake_move is not None:
            self.on_snake_move(new_head, old_head, old_tail)

    def tick(self, new_direction):
        if new_direction is not None:
            self.snake_direction = new_direction.value
        self.move_snake()


def play_game(game):
    def cleanup():
        curses.endwin()

    atexit.register(cleanup)

    curses.initscr()
    win = curses.newwin(game.width + 2, game.height + 4, 0, 0)

    try:
        play_game_helper(game, win)
    finally:
        cleanup()
        atexit.unregister(cleanup)


def play_game_helper(game, win):

    win.keypad(1)
    curses.noecho()
    curses.curs_set(0)
    win.border(0)
    win.nodelay(1)

    char_map = {
        SnakeGame.HEAD: "*",
        SnakeGame.BODY: "+",
        SnakeGame.FRUIT: "#",
        SnakeGame.BLANK: " ",
    }

    def draw_char(pos, c):
        # shift indices since border takes up the first row and column
        win.addch(pos[0] + 1, pos[1] + 1, char_map[c])

    def on_new_fruit(fruit):
        draw_char(fruit, SnakeGame.FRUIT)

    def on_snake_move(new_head, old_head, old_tail):
        draw_char(new_head, SnakeGame.HEAD)
        draw_char(old_head, SnakeGame.BODY)
        if old_tail is not None:
            draw_char(old_tail, SnakeGame.BLANK)

    # register event handlers
    game.on_new_fruit = on_new_fruit
    game.on_snake_move = on_snake_move

    # initial board
    draw_char(game.snake[0], SnakeGame.HEAD)
    for body in itertools.islice(game.snake, 1, None):
        draw_char(body, SnakeGame.BODY)
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
    key = None
    while key != KEY_ESC:
        win.border(0)
        win.timeout(100)

        win.addstr(0, 2, f"Score : {game.score} ")  # Printing 'Score' and
        if game.game_over:
            win.addstr(game.height + 1, 2, "Game over")
        # win.addstr(0, game.width - len(win_title), win_title)  # 'SNAKE' strings
        # win.timeout(150 - (len(game.snake) // 5 + len(game.snake) // 10) % 120)

        event = win.getch()
        key = None if event == -1 else event

        # pause if space bar is pressed
        if key == KEY_SPACE:
            key = None
            while key != KEY_SPACE:
                key = win.getch()
            key = None
            continue

        if key == KEY_ESC:
            break

        if key not in key_map:
            key = None

        if not game.game_over:
            game.tick(new_direction=key_map.get(key))


game = SnakeGame(10, 10)
play_game(game)
