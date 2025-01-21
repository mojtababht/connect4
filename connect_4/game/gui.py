import pygame
import sys

class Connect4GUI:
    def __init__(self, game_engine):
        self.game = game_engine
        pygame.init()
        self.SQUARESIZE = 100
        self.RADIUS = self.SQUARESIZE // 2 - 5
        self.width = self.game.COLS * self.SQUARESIZE
        self.height = (self.game.ROWS + 1) * self.SQUARESIZE
        self.size = (self.width, self.height)
        self.screen = pygame.display.set_mode(self.size)
        self.font = pygame.font.SysFont("monospace", 75)
        self.draw_board()

    def draw_board(self):
        for c in range(self.game.COLS):
            for r in range(self.game.ROWS):
                pygame.draw.rect(self.screen, (0, 0, 255), (c * self.SQUARESIZE, (r + 1) * self.SQUARESIZE, self.SQUARESIZE, self.SQUARESIZE))
                pygame.draw.circle(self.screen, (0, 0, 0), (c * self.SQUARESIZE + self.SQUARESIZE // 2, (r + 1) * self.SQUARESIZE + self.SQUARESIZE // 2), self.RADIUS)

        for c in range(self.game.COLS):
            for r in range(self.game.ROWS):
                if self.game.board[r][c] == 1:
                    pygame.draw.circle(self.screen, (255, 0, 0), (c * self.SQUARESIZE + self.SQUARESIZE // 2, self.height - (r + 1) * self.SQUARESIZE + self.SQUARESIZE // 2), self.RADIUS)
                elif self.game.board[r][c] == 2:
                    pygame.draw.circle(self.screen, (255, 255, 0), (c * self.SQUARESIZE + self.SQUARESIZE // 2, self.height - (r + 1) * self.SQUARESIZE + self.SQUARESIZE // 2), self.RADIUS)

        pygame.display.update()

    def display_winner(self):
        winner = self.game.get_winner()
        if winner:
            label = self.font.render(f"Player {winner} wins!", True, (255, 255, 255))
        else:
            label = self.font.render("It's a draw!", True, (255, 255, 255))
        self.screen.blit(label, (40, 10))
        pygame.display.update()
        pygame.time.wait(3000)

    def handel_event(self, event, turn):
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.width, self.SQUARESIZE))
            posx = event.pos[0]
            if turn == 0:
                pygame.draw.circle(self.screen, (255, 0, 0), (posx, self.SQUARESIZE // 2), self.RADIUS)
            else:
                pygame.draw.circle(self.screen, (255, 255, 0), (posx, self.SQUARESIZE // 2), self.RADIUS)
            pygame.display.update()

        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.width, self.SQUARESIZE))
            posx = event.pos[0]
            col = posx // self.SQUARESIZE

            try:
                self.game.make_move(col, 1 if turn == 0 else 2)
                self.draw_board()
                if self.game.is_game_over():
                    self.display_winner()
                    pygame.time.wait(2000)
                    sys.exit()
                turn = 1 - turn
            except ValueError:
                print("Column full! Choose another.")

    def handle_events(self):
        while not self.game.is_game_over():
            for event in pygame.event.get():
                self.handel_event(event, self.game.turn)
