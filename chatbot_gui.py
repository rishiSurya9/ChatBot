import tkinter as tk
from tkinter import scrolledtext
from chatbot_model import pred_class, get_response, words, classes, data

# The main application window
class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot")
        self.root.geometry("400x550")
        
        # The chat area
        self.chat_area = scrolledtext.ScrolledText(root, state='disabled', wrap='word')
        self.chat_area.pack(padx=10, pady=10, fill='both', expand=True)
        
        # The entry area
        self.entry = tk.Entry(root, width=50)
        self.entry.pack(side='left', padx=(10, 0), pady=(0, 10), fill='x', expand=True)
        self.entry.bind("<Return>", self.on_enter_pressed)
        
        # The stop button
        self.stop_button = tk.Button(root, text="Stop", command=self.stop_chatbot)
        self.stop_button.pack(side='bottom', padx=10, pady=10)
        
        # welcome message
        self.display_message("Bot", "Hello! How can I assist you today?")
    
    def display_message(self, sender, message):
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, f"{sender}: {message}\n")
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)
    
    def get_user_input(self):
        return self.entry.get()
    
    def clear_entry(self):
        self.entry.delete(0, tk.END)
    
    def on_enter_pressed(self, event=None):
        self.process_input()
    
    def process_input(self):
        user_input = self.get_user_input()
        if user_input:
            self.display_message("You", user_input)
            intents = pred_class(user_input, words, classes)
            response = get_response(intents, data)
            self.display_message("Bot", response)
            self.clear_entry()

    def stop_chatbot(self):
        self.display_message("Bot", "Terminating the chatbot. Goodbye!")
        self.root.after(2000, self.root.quit)  # Wait for 2 seconds before closing the window

# Run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
