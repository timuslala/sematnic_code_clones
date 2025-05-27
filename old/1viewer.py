import tkinter as tk
from tkinter import scrolledtext
import pyperclip
import os

def load_file_content(file_name):
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"File {file_name} not found."

def display_contents(event=None):
    clipboard_data = pyperclip.paste()
    file_ids = clipboard_data.split()
    
    if len(file_ids) == 2:
        file1_content = load_file_content(f"{file_ids[0]}.java")
        file2_content = load_file_content(f"{file_ids[1]}.java")
        
        text_area1.delete(1.0, tk.END)
        text_area1.insert(tk.END, file1_content)
        
        text_area2.delete(1.0, tk.END)
        text_area2.insert(tk.END, file2_content)

app = tk.Tk()
app.title("Clipboard File Viewer")

text_area1 = scrolledtext.ScrolledText(app, width=60, height=80)
text_area1.pack(side=tk.LEFT, padx=10, pady=10)

text_area2 = scrolledtext.ScrolledText(app, width=60, height=80)
text_area2.pack(side=tk.RIGHT, padx=10, pady=10)

# Bind Ctrl+V to display_contents
app.bind('<Control-v>', display_contents)
app.bind('<Control-V>', display_contents)

display_contents()

app.mainloop()