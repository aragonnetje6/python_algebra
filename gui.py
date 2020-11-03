"""
User interface for the tree
"""

import tkinter as tk

window = tk.Tk()
window.geometry("1000x750")
frame = tk.Frame(window, bg='white')
frame.pack()

label = tk.Label(frame, text="Hello world")
label.pack()

entry = tk.Entry(frame, width=200)
entry.insert(0, 'Username')
entry.pack(padx=5, pady=5, )

window.title("python_algebra")
window.mainloop()
