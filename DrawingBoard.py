import tkinter as tk
import numpy as np

class DrawingBoard:
    def __init__(self, root):
        self.root = root
        self.array_size = 28  # Input array size for Neural Network
        self.pixel_size = 10  # Each "pixel" size on the canvas (enlarged for visibility)
        self.canvas_size = self.array_size * self.pixel_size  # Total canvas size
        self.brush_radius = 1  # Radius of the brush (in grid cells)

        # Set up the canvas
        self.canvas = tk.Canvas(root, bg='black', width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        # Initialize the grid to store pixel values
        self.grid = np.zeros((self.array_size, self.array_size))  # 28x28 grid, all initialized to 0 (black)
        self.rects = [[None for _ in range(self.array_size)] for _ in range(self.array_size)]
        
        self.draw_grid()

        # Bind mouse events for drawing
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<Button-1>', self.draw)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_board)
        self.clear_button.pack()

        self.display_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.display_label.pack()

    def draw_grid(self):
        """Draw the 28x28 grid on the canvas."""
        for i in range(self.array_size):
            for j in range(self.array_size):
                x0 = j * self.pixel_size
                y0 = i * self.pixel_size
                x1 = x0 + self.pixel_size
                y1 = y0 + self.pixel_size
                # Draw each grid cell (rectangle)
                self.rects[i][j] = self.canvas.create_rectangle(
                    x0, y0, x1, y1, outline='gray', fill='black'
                )

    def draw(self, event):
        """Fill grid cells within the brush radius where the mouse is clicked or dragged."""
        x, y = event.x, event.y
        grid_x = x // self.pixel_size
        grid_y = y // self.pixel_size

        for dx in range(-self.brush_radius, self.brush_radius + 1):
            for dy in range(-self.brush_radius, self.brush_radius + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.array_size and 0 <= ny < self.array_size:
                    self.grid[ny, nx] = 1  # Mark the cell as white (1)
                    self.canvas.itemconfig(self.rects[ny][nx], fill='white')

    def get_array(self):
        """Return the current 28x28 grid as a NumPy array."""
        return self.grid

    def clear_board(self):
        """Clear the canvas and reset the grid."""
        self.grid.fill(0)  # Reset the grid
        for i in range(self.array_size):
            for j in range(self.array_size):
                self.canvas.itemconfig(self.rects[i][j], fill='black')  # Reset the grid cells to black

    def update_text(self, prediction, confidence):
        """Update the display label with the provided text."""
        self.display_label.config(text=f"Prediction  >  {prediction}\nConfidence  >  {confidence}")


if __name__ == "__main__":
    root = tk.Tk()
    root.title(">>AI Drawing Board<<")
    drawing_board = DrawingBoard(root)
    root.mainloop()