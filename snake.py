import collections
import enum
import numpy
import random
from enum import Enum
from typing import Callable, NamedTuple, Optional, Tuple

# Game adapted from https://gist.github.com/sanchitgangwar/2158089


class SnakeGame:

    BLANK = 0
    HEAD = 0.5
    BODY = 0.25
    FRUIT = 1

    class BoardElement(Enum):
        BLANK = enum.auto()
        FRUIT = enum.auto()
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
        part: "SnakeGame.BoardElement"

    class Move(Enum):
        UP = (-1, 0)
        RIGHT = (0, 1)
        DOWN = (1, 0)
        LEFT = (0, -1)

        def is_opposite(self, rhs):
            return self.value[0] == -rhs.value[0] and self.value[1] == -rhs.value[1]

    def __init__(
        self, width, height, num_fruits=1, walls=False, grows=True, reverse_death=False
    ):

        assert width >= 3 and height >= 3
        assert num_fruits > 0

        self.width = width
        self.height = height
        self.walls = walls
        self.grows = grows
        self.reverse_death = reverse_death

        self.board = numpy.zeros((self.height, self.width))
        self.snake = collections.deque()  # left end is head, right end is tail
        self.snake_direction = SnakeGame.Move.RIGHT
        self.fruits = []
        self.score = 0
        self.moves_since_last_fruit = 0
        self.game_over = False
        self.on_new_fruit: Optional[Callable] = None
        self.on_snake_move: Optional[Callable] = None

        # snake initial position
        center_row = self.height // 2
        center_col = self.width // 2
        initial_snake_parts = [
            SnakeGame.BoardElement.BODY_LEFT_RIGHT,
            SnakeGame.BoardElement.BODY_LEFT_RIGHT,
            SnakeGame.BoardElement.HEAD_RIGHT,
        ]
        for j, part in zip([-1, 0, 1], initial_snake_parts):
            pos = (center_row, center_col + j)
            self.snake.appendleft(SnakeGame.SnakePart(pos, part))
            self.board[pos] = (
                self.HEAD if part == SnakeGame.BoardElement.HEAD_RIGHT else self.BODY
            )

        for i in range(num_fruits):
            self.spawn_fruit()

    def spawn_fruit(self):
        pos = None
        while pos is None or self.board[pos] != SnakeGame.BLANK:
            pos = (random.randrange(self.height), random.randrange(self.width))

        self.board[pos] = SnakeGame.FRUIT
        self.fruits.append(pos)

        if self.on_new_fruit is not None:
            self.on_new_fruit(pos)

    def get_board(self):
        if self.walls:
            # for now, just copy the board
            return numpy.copy(self.board)

        rotations = {
            self.Move.RIGHT: 0,
            self.Move.DOWN: 1,
            self.Move.LEFT: 2,
            self.Move.UP: 3,
        }

        center_row = self.height // 2
        center_col = self.width // 2

        head = self.snake[0].pos
        shift = (center_row - head[0], center_col - head[1])
        return numpy.rot90(
            numpy.roll(self.board, shift), rotations[self.snake_direction]
        )

    def tick(self, action: Optional["SnakeGame.Move"] = None):
        if self.game_over:
            return

        self.moves_since_last_fruit += 1
        prev_direction = self.snake_direction
        if action is not None:
            if not self.snake_direction.is_opposite(action):
                self.snake_direction = action
            elif self.reverse_death:
                self.game_over = True
                return

        old_head = self.snake.popleft()

        new_head_row = old_head.pos[0] + self.snake_direction.value[0]
        new_head_col = old_head.pos[1] + self.snake_direction.value[1]

        if not self.walls:
            new_head_row %= self.height
            new_head_col %= self.width
        elif not (0 <= new_head_row < self.height and 0 <= new_head_col < self.width):
            self.game_over = True
            return

        new_head_pos = (new_head_row, new_head_col)

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
            self.moves_since_last_fruit = 0
            self.fruits.remove(new_head_pos)
            self.spawn_fruit()
            old_tail = None
            if not self.grows:
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
            return SnakeGame.BoardElement.HEAD_UP
        elif direction == SnakeGame.Move.RIGHT:
            return SnakeGame.BoardElement.HEAD_RIGHT
        elif direction == SnakeGame.Move.DOWN:
            return SnakeGame.BoardElement.HEAD_DOWN
        else:
            return SnakeGame.BoardElement.HEAD_LEFT

    @classmethod
    def get_part_type(cls, head_direction, old_head_direction):
        ahead_diff = head_direction.value
        behind_diff = old_head_direction.value
        if ahead_diff == behind_diff:
            if ahead_diff[0] == 0:
                return cls.BoardElement.BODY_LEFT_RIGHT
            return cls.BoardElement.BODY_DOWN_UP

        diff_change = (ahead_diff[0] - behind_diff[0], ahead_diff[1] - behind_diff[1])
        if diff_change == (1, 1):
            return cls.BoardElement.BODY_RIGHT_DOWN
        elif diff_change == (1, -1):
            return cls.BoardElement.BODY_LEFT_DOWN
        elif diff_change == (-1, 1):
            return cls.BoardElement.BODY_RIGHT_UP
        elif diff_change == (-1, -1):
            return cls.BoardElement.BODY_LEFT_UP
