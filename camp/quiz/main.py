import pygame
import sys
import random
import math

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 30

# Multiple choice questions on %6==0 cells
QUESTIONS = [
    {"text": "Why do Large Language Models sometimes produce incorrect or nonsensical answers?",
     "options": ["They don't have enough data to learn from","They predict text based on patterns, not true understanding","They are programmed to make mistakes occasionally","They only work with numbers, not words"],
     "answer": 1},
    {"text": "How can an AI model \"understand\" language without knowing the meaning of words?",
     "options": ["It memorizes every sentence it has seen","It uses statistical relationships between words learned from data","It asks humans for help when confused","It translates words into images first"],
     "answer": 1},
    {"text": "Why is diverse training data important for LLMs?",
     "options": ["To make the model run faster","To reduce bias and improve handling of different languages and cultures","To make the model smaller in size","To limit the model's vocabulary"],
     "answer": 1},
    {"text": "What is a major challenge when deploying LLMs in real-world applications?",
     "options": ["Making the model speak multiple languages","Ensuring the model doesn't produce harmful or biased content","Teaching the model to play games","Reducing the number of parameters"],
     "answer": 1},
    {"text": "How do LLMs \"remember\" what was said earlier in a conversation?",
     "options": ["They store the entire internet in memory","They use context windows that keep track of previous tokens up to a limit","They write everything down on a notepad","They don't remember anything at all"],
     "answer": 1},
    {"text": "Why can't LLMs reliably fact-check themselves?",
     "options": ["They don't have access to real-time databases or the internet during generation","They are designed to lie sometimes","They only work with fictional stories","They forget facts after training"],
     "answer": 0},
    {"text": "What's the difference between a rule-based chatbot and an LLM-based chatbot?",
     "options": ["Rule-based chatbots follow fixed scripts; LLM chatbots generate flexible, context-aware responses","Rule-based chatbots are faster at typing","LLM chatbots only answer yes/no questions","Rule-based chatbots can learn from new data automatically"],
     "answer": 0},
    {"text": "Why do LLMs sometimes \"hallucinate\" facts?",
     "options": ["They get tired after long conversations","They guess based on patterns when they lack specific knowledge","They are programmed to invent stories","They confuse numbers with words"],
     "answer": 1},
    {"text": "How does \"fine-tuning\" improve an LLM's performance?",
     "options": ["By training the model on a smaller, task-specific dataset","By increasing the model's size","By deleting irrelevant data from training","By making the model forget old information"],
     "answer": 0},
    {"text": "Which ethical consideration is important when developing AI language models?",
     "options": ["Making the model as fast as possible","Avoiding bias and preventing misuse","Ensuring the model can generate jokes","Limiting the model's vocabulary to simple words"],
     "answer": 1}
]

# AI quiz dict for %6==3 cells
AI_QUIZ = {
    "1": {
        "question": "What year was the term \"Artificial Intelligence\" first coined?",
        "options": {"A": "1945", "B": "1956", "C": "1969", "D": "1978"},
        "answer": "B"
    },
    "2": {
        "question": "Which game did an AI first defeat a reigning world champion, marking a major milestone in AI?",
        "options": {"A": "Chess", "B": "Go", "C": "Checkers", "D": "Poker"},
        "answer": "A"
    },
    "3": {
        "question": "What is the name of the AI model developed by OpenAI that can generate human-like text?",
        "options": {"A": "AlphaGo", "B": "Watson", "C": "GPT", "D": "DeepBlue"},
        "answer": "C"
    },
    "4": {
        "question": "Which of these is NOT a common application of AI today?",
        "options": {"A": "Voice assistants", "B": "Weather forecasting", "C": "Cooking food", "D": "Image recognition"},
        "answer": "C"
    },
    "5": {
        "question": "What is the term for AI systems that can learn and improve from experience without being explicitly programmed?",
        "options": {"A": "Supervised learning", "B": "Reinforcement learning", "C": "Machine learning", "D": "Deep learning"},
        "answer": "C"
    }
}

class Cell:
    def __init__(self, index, x, y, effect=None):
        self.index = index
        self.x = x
        self.y = y
        self.effect = effect

class Board:
    def __init__(self, cells):
        self.cells = cells
    def draw(self, screen):
        light_blue = (173, 216, 230)
        gold = (255, 215, 0)
        purple = (186, 85, 211)
        black = (0, 0, 0)
        for cell in self.cells:
            if cell.effect == 'question': color = gold
            elif cell.effect == 'ai_quiz': color = purple
            elif cell.effect in ('loss5', 'loss10'): color = black
            else: color = light_blue
            pygame.draw.circle(screen, color, (int(cell.x), int(cell.y)), 20, 2)
    def get_cell(self, index): return self.cells[index]

class Player:
    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.position = 0
        self.score = 0
    def draw(self, screen, board):
        cell = board.get_cell(self.position)
        pygame.draw.circle(screen, self.color, (int(cell.x), int(cell.y)), 10)
    def move(self, steps, board):
        self.position = max(0, min(self.position + steps, len(board.cells)-1))

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Board Dice Game")
        self.clock = pygame.time.Clock()
        self.background = None
        for fn in ('background.jpg','background.png'):
            try:
                bg = pygame.image.load(fn).convert()
                self.background = pygame.transform.scale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT))
                break
            except:
                pass
        self.button_rect = pygame.Rect(50, SCREEN_HEIGHT-80, 200, 50)
        self.font = pygame.font.SysFont(None, 24)
        self.cells = self.create_cells()
        self.board = Board(self.cells)
        self.player = Player("Player1", (255,0,0))
        self.state = 'normal'
        self.current_question = None
        self.question_buttons = []
        self.last_steps = 0
        self.message = ''

    def create_cells(self):
        cells = []
        cx, cy = SCREEN_WIDTH/2, SCREEN_HEIGHT/2
        max_r = min(SCREEN_WIDTH,SCREEN_HEIGHT)*0.45
        turns = 3
        num = 60
        for i in range(num):
            t = (i/(num-1))* (turns*2*math.pi)
            r = max_r*(1 - i/(num-1))
            x = cx + r*math.cos(t)
            y = cy + r*math.sin(t)
            idx = i+1
            effect = None
            if idx%6 ==0: effect='question'
            elif idx%6==3: effect='ai_quiz'
            elif idx%10==0: effect='loss5'
            elif idx%15==0: effect='loss10'
            cells.append(Cell(i,x,y,effect))
        return cells

    def roll_dice(self): return random.randint(1,6)

    def start_question(self):
        self.current_question = random.choice(QUESTIONS)
        opts = self.current_question['options']
        self.question_buttons=[]
        w,h = 500,30
        y0=200
        for i, opt in enumerate(opts):
            rect = pygame.Rect(150, y0+i*(h+10), w, h)
            self.question_buttons.append((rect, i))
        self.state='question'

    def start_ai_quiz(self):
        key = random.choice(list(AI_QUIZ.keys()))
        qdata = AI_QUIZ[key]
        self.current_question = {'text': qdata['question'], 'options': qdata['options'], 'answer': qdata['answer']}
        self.question_buttons=[]
        w,h = 500,30
        y0=200
        for opt_key, opt_text in self.current_question['options'].items():
            rect = pygame.Rect(150, y0* (list(self.current_question['options'].keys()).index(opt_key)+1), w, h)
            self.question_buttons.append((rect, opt_key))
        self.state='ai_quiz'

    def handle_question_click(self, pos):
        for rect, idx in self.question_buttons:
            if rect.collidepoint(pos):
                correct = idx == self.current_question['answer']
                if correct:
                    self.player.move(self.last_steps, self.board)
                    self.message = 'Correct! Movement doubled.'
                else:
                    self.player.move(-1, self.board)
                    self.message = 'Wrong! -1 step.'
                self.state = 'normal'
                return

    def handle_ai_click(self, pos):
        for rect, opt_key in self.question_buttons:
            if rect.collidepoint(pos):
                if opt_key == self.current_question['answer']:
                    self.player.score += 5
                    self.message = 'AI Quiz correct! +5 points.'
                else:
                    self.player.score -= 1
                    self.message = 'AI Quiz wrong! -1 point.'
                self.state = 'normal'
                return

    def apply_losses(self, cell):
        if cell.effect == 'loss5':
            self.player.move(-5, self.board)
            self.message = 'You lost funding, go 5 steps backwards.'
        elif cell.effect == 'loss10':
            self.player.move(-10, self.board)
            self.message = 'Your main researcher went to OpenMind instead of your company.'

    def draw(self):
        if self.background:
            self.screen.blit(self.background, (0,0))
        else:
            self.screen.fill((255,255,255))
        # win condition
        if self.player.position == len(self.cells)-1:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(200)
            overlay.fill((0,0,0))
            self.screen.blit(overlay, (0,0))
            msg = "You won! AI safety solved and humanity is safed. You can go to sleep now."
            surf = self.font.render(msg, True, (255,255,255))
            rect = surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            self.screen.blit(surf, rect)
            pygame.display.flip()
            return

        self.board.draw(self.screen)
        self.player.draw(self.screen, self.board)
        # steps to end
        steps_left = len(self.cells)-1 - self.player.position
        txt = self.font.render(f"{steps_left} steps to singularity", True, (0,0,0))
        self.screen.blit(txt, txt.get_rect(topright=(SCREEN_WIDTH-10, 10)))
        # button
        if self.state == 'normal':
            pygame.draw.rect(self.screen, (200,200,200), self.button_rect)
            txt2 = self.font.render('Roll Dice', True, (0,0,0))
            self.screen.blit(txt2, txt2.get_rect(center=self.button_rect.center))
        # message and score
        msg_surf = self.font.render(self.message, True, (0,0,0))
        self.screen.blit(msg_surf, (10, 10))
        score_surf = self.font.render(f'Score: {self.player.score}', True, (0,0,0))
        self.screen.blit(score_surf, (10, 40))
        # question overlays
        if self.state in ('question', 'ai_quiz'):
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(180)
            overlay.fill((0,0,0))
            self.screen.blit(overlay, (0,0))
            qtxt = self.font.render(self.current_question['text'], True, (255,255,255))
            self.screen.blit(qtxt, (50, 150))
            for rect, val in self.question_buttons:
                pygame.draw.rect(self.screen, (255,255,255), rect)
                # always show text
                opt_text = self.current_question['options'][val] if isinstance(self.current_question['options'], dict) else self.current_question['options'][val]
                otxt = self.font.render(opt_text, True, (0,0,0))
                self.screen.blit(otxt, (rect.x+5, rect.y+5))

        pygame.display.flip()

    def main_loop(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    if self.state == 'normal' and self.button_rect.collidepoint(e.pos):
                        steps = self.roll_dice()
                        self.last_steps = steps
                        self.player.move(steps, self.board)
                        cell = self.board.get_cell(self.player.position)
                        self.message = f'You are {steps} steps closer to singularity'
                        if cell.effect == 'question':
                            self.start_question()
                        elif cell.effect == 'ai_quiz':
                            self.start_ai_quiz()
                        elif cell.effect in ('loss5', 'loss10'):
                            self.apply_losses(cell)
                    elif self.state == 'question':
                        self.handle_question_click(e.pos)
                    elif self.state == 'ai_quiz':
                        self.handle_ai_click(e.pos)

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    Game().main_loop()
