import tkinter as tk
import numpy as np
import random
import matplotlib.pyplot as plt

# Параметри
CELL_SIZE = 10
ROWS, COLS = 70, 70

S, I, R = 0, 1, 2

COLORS = {
    S: "#A1D99B",
    I: "#FC9272",
    R: "#9ECAE1"
}

# Ініціалізація
def initialize_grid(infected_count):
    grid = np.zeros((ROWS, COLS), dtype=int)

    for _ in range(infected_count):
        i = random.randint(0, ROWS - 1)
        j = random.randint(0, COLS - 1)
        grid[i, j] = I

    return grid

# Крок
def step(grid, beta, gamma, alpha):
    new_grid = grid.copy()

    for i in range(ROWS):
        for j in range(COLS):
            if grid[i, j] == S:
                infected_neighbors = 0

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue

                        ni, nj = i + di, j + dj

                        if 0 <= ni < ROWS and 0 <= nj < COLS:
                            if grid[ni, nj] == I:
                                infected_neighbors += 1

                p = 1 - (1 - beta) ** infected_neighbors

                if random.random() < p:
                    new_grid[i, j] = I

            elif grid[i, j] == I:
                if random.random() < gamma:
                    new_grid[i, j] = R

            elif grid[i, j] == R:
                if random.random() < alpha:
                    new_grid[i, j] = S

    return new_grid

# GUI
class Simulation:
    def __init__(self, root):
        self.root = root
        self.root.title("Симуляція епідемій")

        # Заголовок
        tk.Label(root,
                 text="Симуляція поширення інфекційних захворювань",
                 font=("Arial", 16, "bold")).pack(pady=(10, 5), padx=10)

        self.beta = 0.3
        self.gamma = 0.1
        self.alpha = 0.05
        self.initial_infected = 10
        self.delay = 100

        self.grid = initialize_grid(self.initial_infected)
        self.running = False
        self.drawing = False

        # Дані для графіка
        self.S_data = []
        self.I_data = []
        self.R_data = []

        # Канвас
        self.canvas = tk.Canvas(
            root,
            width=COLS * CELL_SIZE,
            height=ROWS * CELL_SIZE,
            bg="white"
        )
        self.canvas.pack(padx=10)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.erase_cell)

        # Параметри
        param_frame = tk.Frame(root)
        param_frame.pack(pady=5, padx=10)

        tk.Label(param_frame, text="β:").grid(row=0, column=0)
        self.beta_entry = tk.Entry(param_frame, width=5)
        self.beta_entry.insert(0, "0.3")
        self.beta_entry.grid(row=0, column=1)

        tk.Label(param_frame, text="γ:").grid(row=0, column=2)
        self.gamma_entry = tk.Entry(param_frame, width=5)
        self.gamma_entry.insert(0, "0.1")
        self.gamma_entry.grid(row=0, column=3)

        tk.Label(param_frame, text="α:").grid(row=0, column=4)
        self.alpha_entry = tk.Entry(param_frame, width=5)
        self.alpha_entry.insert(0, "0.05")
        self.alpha_entry.grid(row=0, column=5)

        tk.Label(param_frame, text="I₀:").grid(row=0, column=6)
        self.infected_entry = tk.Entry(param_frame, width=5)
        self.infected_entry.insert(0, "10")
        self.infected_entry.grid(row=0, column=7)

        tk.Button(param_frame, text="Apply",
                  command=self.apply_params).grid(row=0, column=8, padx=5)

        # Слайдер швидкості
        speed_frame = tk.Frame(root)
        speed_frame.pack(pady=1, padx=10)

        tk.Label(speed_frame, text="Speed").pack()

        self.speed_slider = tk.Scale(
            speed_frame,
            from_=10,
            to=300,
            orient=tk.HORIZONTAL,
            command=self.change_speed
        )
        self.speed_slider.set(150)
        self.speed_slider.pack()

        # Кнопки
        button_frame = tk.Frame(root)
        button_frame.pack(pady=5, padx=10)

        tk.Button(button_frame, text="Start",
                  command=self.start).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Pause",
                  command=self.pause).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Reset",
                  command=self.reset).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save Graph",
                  command=self.save_plot).pack(side=tk.LEFT, padx=5)

        self.draw()
        self.update()

    # Керування
    def start(self):
        self.running = True

    def pause(self):
        self.running = False

    def reset(self):
        self.grid = initialize_grid(self.initial_infected)
        self.S_data.clear()
        self.I_data.clear()
        self.R_data.clear()
        self.draw()

    def apply_params(self):
        try:
            self.beta = float(self.beta_entry.get())
            self.gamma = float(self.gamma_entry.get())
            self.alpha = float(self.alpha_entry.get())
            self.initial_infected = int(self.infected_entry.get())
            self.reset()
        except:
            print("Invalid input")

    def change_speed(self, value):
        min_delay = 10
        max_delay = 300
        self.delay = max_delay - int(value) + min_delay

    # Збір даних
    def collect_data(self):
        self.S_data.append(np.sum(self.grid == S))
        self.I_data.append(np.sum(self.grid == I))
        self.R_data.append(np.sum(self.grid == R))

    # Збереження графіку
    def save_plot(self):
        s = np.array(self.S_data)
        i = np.array(self.I_data)
        r = np.array(self.R_data)

        plt.figure(figsize=(8, 5))

        plt.plot(s, linewidth=2, color='C2', label="Сприйнятливі")
        plt.plot(i, linewidth=2, color='C1', label="Заражені")
        plt.plot(r, linewidth=2, color='C9', label="Одужалі")

        plt.xlabel("Крок часу")
        plt.ylabel("Кількість клітин")
        plt.title("Динаміка SIRS")
        plt.legend()
        plt.grid()

        # Збереження графіка
        filename = "sirs_graph.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")

        print(f"Графік збережено у файл: {filename}")

        plt.show()

    # Малювання
    def on_click(self, event):
        self.drawing = True
        self.paint_cell(event)

    def on_drag(self, event):
        if self.drawing:
            self.paint_cell(event)

    def on_release(self, event):
        self.drawing = False

    def paint_cell(self, event):
        j = event.x // CELL_SIZE
        i = event.y // CELL_SIZE
        if 0 <= i < ROWS and 0 <= j < COLS:
            self.grid[i, j] = I
            self.draw()

    def erase_cell(self, event):
        j = event.x // CELL_SIZE
        i = event.y // CELL_SIZE
        if 0 <= i < ROWS and 0 <= j < COLS:
            self.grid[i, j] = S
            self.draw()

    def draw(self):
        self.canvas.delete("all")

        for i in range(ROWS):
            for j in range(COLS):
                color = COLORS[self.grid[i, j]]

                x1 = j * CELL_SIZE
                y1 = i * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE

                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline="#dddddd"
                )

    def update(self):
        if self.running:
            self.grid = step(self.grid, self.beta, self.gamma, self.alpha)
            self.collect_data()
            self.draw()

        self.root.after(self.delay, self.update)

root = tk.Tk()
app = Simulation(root)
root.mainloop()