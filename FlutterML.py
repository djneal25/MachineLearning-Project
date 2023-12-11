import pandas as pd
from fuzzywuzzy import process
import tkinter as tk
from tkinter import messagebox
import time

# Loads the CSV file into a pandas DataFrame
df = pd.read_csv('flutter_commands.csv')

column_keywords = {
    'function': 'Function',
    'documentation': 'Documentation',
    'signature': 'Signature',
    'return': 'Return Type',
    'example': 'Example Usage',
    'library': 'Library'
}

def get_best_match(query, choices):
    best_match, score = process.extractOne(query, choices)
    if score > 80:
        return best_match
    return None

def get_function_info(function_name, request):
    function_data = df[df['Function'].str.lower() == function_name.lower()]
    if function_data.empty:
        return f"No data found for function: {function_name}"

    request = request.lower()

    info = ""
    for keyword, col in column_keywords.items():
        if keyword in request:
            info += f"{col}: {function_data[col].iloc[0]}\n"

    if not info:
        for col in df.columns:
            if col != "Function":
                info += f"{col}: {function_data[col].iloc[0]}\n"

    return info


def animate_typing(result_text, text):
    for char in text:
        result_text.config(state='normal')
        result_text.insert(tk.END, char)
        result_text.config(state='disabled')
        result_text.update()
        time.sleep(0.01)

def handle_query():
    user_input = input_entry.get()
    if user_input.lower() == 'exit':
        root.destroy()
    else:
        function_name = get_best_match(user_input, df['Function'].tolist())
        if not function_name:
            messagebox.showinfo("Info", "Could not determine the function name from your query.")
        else:
            result_text.config(state='normal')
            result_text.delete('1.0', tk.END)
            result_text.config(state='disabled')
            input_entry.delete(0, tk.END)
            input_entry.focus()

            function_info = get_function_info(function_name, user_input)
            animate_typing(result_text, f"Function: {function_name}\n")
            animate_typing(result_text, function_info)

def on_enter(event):
    handle_query()

root = tk.Tk()
root.title("Function Info")
root.geometry("400x300")

input_label = tk.Label(root, text="Enter your query or 'exit' to quit:")
input_label.pack()

input_entry = tk.Entry(root)
input_entry.pack()
input_entry.bind('<Return>', on_enter)
input_entry.focus_set()

query_button = tk.Button(root, text="Query", command=handle_query)
query_button.pack()

result_text = tk.Text(root, wrap=tk.WORD, height=10, state='disabled')
result_text.pack()

root.mainloop()