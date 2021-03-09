import atexit
import collections
import curses
import numpy
import random

from enum import Enum
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN


class SnakeGame:

    BLANK = 0
    HEAD = 0.5
    BODY = 0.25
    FRUIT = 1

    class Move(Enum):
        UP = (0, 1)
        RIGHT = (1, 0)
        DOWN = (0, -1)
        LEFT = (-1, 0)

    def __init__(self, width, height, on_new_fruit = None, on_snake_move = None):

        assert width >= 3 and height >= 3

        self.width = width
        self.height = height
        self.board = numpy.zeros((self.height, self.width))
        self.snake = collections.deque()  # right end is head, left end is tail
        self.snake_direction = SnakeGame.Move.RIGHT
        self.score = 0
        self.on_new_fruit = on_new_fruit
        self.on_snake_move = on_snake_move

        # snake initial position
        center_row = self.height // 2
        center_col = self.width // 2
        for j, part in zip(range(1, -2, -1), [self.HEAD, self.BODY, self.BODY]):
            pos = (center_row, center_col + j)
            self.snake.append(pos)
            self.board[pos] = part

    def spawn_fruit(self):
        while True:
            pos = (random.randrange(self.height), random.randrange(self.width))
            if self.board[pos] == SnakeGame.BLANK:
                self.board[pos] = SnakeGame.FRUIT
                if self.on_new_fruit is not None:
                    self.on_new_fruit(pos)
                break

    def move_snake(self):
        old_head = self.snake[0]
        new_head = (
            old_head[0] + self.snake_direction[0] % self.height,
            old_head[1] + self.snake_direction[1] % self.width,
        )
        self.snake.append(new_head)
        old_tail = self.snake.popleft()

        # update board
        self.board[old_tail] = SnakeGame.BLANK
        self.board[old_head] = SnakeGame.BODY
        if self.board[new_head] == SnakeGame.FRUIT:
            self.score += 1
            self.spawn_fruit()
        self.board[new_head] = SnakeGame.HEAD

        if self.on_snake_move is not None:
            self.on_snake_move(new_head, old_tail)

    def tick(self, new_direction):
        if new_direction is not None:
            self.snake_direction = new_direction
        self.move_snake()



def play_game(game):

    def cleanup():
        curses.endwin()
    atexit.register(cleanup)

    curses.initscr()
    win = curses.newwin(20, 60, 0, 0)

    def on_new_fruit(game, fruit):
        win.addch(fruit[0], fruit[1], "#")

    def on_snake_move(game, new_head, old_head, old_tail):
        win.addch(old_tail[0], old_tail[1], " ")
        win.addch(old_head[0], old_head[1], "+")
        win.addch(new_head[0], new_head[1], "*")


    #
    win.keypad(1)
    curses.noecho()
    curses.curs_set(0)
    win.border(0)
    win.nodelay(1)

    key = KEY_RIGHT  # Initializing values
    score = 0

    snake = [[4, 10], [4, 9], [4, 8]]  # Initial snake co-ordinates
    food = [10, 20]  # First food co-ordinates

    win.addch(food[0], food[1], "*")  # Prints the food

    while key != 27:  # While Esc key is not pressed
        win.border(0)
        win.addstr(0, 2, "Score : " + str(score) + " ")  # Printing 'Score' and
        win.addstr(0, 27, " SNAKE ")  # 'SNAKE' strings
        win.timeout(
            150 - (len(snake) // 5 + len(snake) // 10) % 120
        )  # Increases the speed of Snake as its length increases

        prevKey = key  # Previous key pressed
        event = win.getch()
        key = key if event == -1 else event

        if key == ord(" "):  # If SPACE BAR is pressed, wait for another
            key = -1  # one (Pause/Resume)
            while key != ord(" "):
                key = win.getch()
            key = prevKey
            continue

        if key not in [
            KEY_LEFT,
            KEY_RIGHT,
            KEY_UP,
            KEY_DOWN,
            27,
        ]:  # If an invalid key is pressed
            key = prevKey

        # Calculates the new coordinates of the head of the snake. NOTE: len(snake) increases.
        # This is taken care of later at [1].
        snake.insert(
            0,
            [
                snake[0][0] + (key == KEY_DOWN and 1) + (key == KEY_UP and -1),
                snake[0][1] + (key == KEY_LEFT and -1) + (key == KEY_RIGHT and 1),
            ],
        )

        # If snake crosses the boundaries, make it enter from the other side
        if snake[0][0] == 0:
            snake[0][0] = 18
        if snake[0][1] == 0:
            snake[0][1] = 58
        if snake[0][0] == 19:
            snake[0][0] = 1
        if snake[0][1] == 59:
            snake[0][1] = 1

        # Exit if snake crosses the boundaries (Uncomment to enable)
        # if snake[0][0] == 0 or snake[0][0] == 19 or snake[0][1] == 0 or snake[0][1] == 59: break

        # If snake runs over itself
        if snake[0] in snake[1:]:
            break

        if snake[0] == food:  # When snake eats the food
            food = []
            score += 1
            while food == []:
                food = [
                    randint(1, 18),
                    randint(1, 58),
                ]  # Calculating next food's coordinates
                if food in snake:
                    food = []
            win.addch(food[0], food[1], "*")
        else:
            last = snake.pop()  # [1] If it does not eat the food, length decreases
            win.addch(last[0], last[1], " ")
        win.addch(snake[0][0], snake[0][1], "#")

    curses.endwin()
    print("\nScore - " + str(score))
