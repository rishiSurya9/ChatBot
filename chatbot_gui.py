import tkinter as tk
from tkinter import Label, Entry, Text, Scrollbar, scrolledtext
from PIL import Image, ImageTk  # To handle JPG images
from chatbot_model import pred_class, get_response, words, classes, data

# The main application window
class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Chatbot")
        self.root.geometry("500x600")
        self.root.configure(bg='#000000')  

        # Logo/Icon 
        img = Image.open("ai.jpg")  # Replace with actual path to your JPG image
        img = img.resize((110, 110), Image.LANCZOS)  # Resize the image to fit
        logo_img = ImageTk.PhotoImage(img)
        logo_label = Label(root, image=logo_img, bg='#000000')  # Set logo background to black
        logo_label.image = logo_img  # Keep a reference to avoid garbage collection
        logo_label.pack(pady=10)

        # The chat area (ScrolledText widget for better user experience)
        self.chat_area = scrolledtext.ScrolledText(root, state='disabled', wrap='word', bg="#333333", fg="#ffffff", font="Helvetica 12", padx=10, pady=10)
        self.chat_area.config(borderwidth=0, relief="flat", insertbackground="white")  # White text cursor
        self.chat_area.pack(padx=10, pady=10, fill='both', expand=True)

        # The entry area
        self.entry_frame = tk.Frame(root, bg='#000000')  # Frame to contain the entry and button
        self.entry_frame.pack(padx=10, pady=(0, 10), fill='x')

        # User input entry field
        self.entry = tk.Entry(self.entry_frame, width=40, bd=0, bg="#ffffff", font="Helvetica 12", fg="#000000", relief="flat")
        self.entry.pack(side='left', padx=(0, 10), pady=10, ipady=5, expand=True)
        self.entry.bind("<Return>", self.on_enter_pressed)  # Trigger on pressing Enter

        # Send button with icon
        send_icon = Image.open("send.jpg")  # Replace with actual path to your send icon image
        send_icon = send_icon.resize((50, 50), Image.LANCZOS)  # Resize to fit button
        send_icon_img = ImageTk.PhotoImage(send_icon)

        self.send_button = tk.Button(self.entry_frame, image=send_icon_img, bg="#4CAF50", bd=0, relief="flat", activebackground="#4CAF50", cursor="hand2")
        self.send_button.image = send_icon_img  # Keep a reference to avoid garbage collection
        self.send_button.pack(side='right', padx=(0, 10), pady=10)
        self.send_button.bind("<Button-1>", self.on_enter_pressed)

        # Stop button to terminate the chatbot
        self.stop_button = tk.Button(root, text="Stop", font=("Helvetica", 12, "bold"), bg="#FF5733", fg="white", relief="flat", bd=0, padx=10, pady=5, cursor="hand2")
        self.stop_button.pack(pady=10)
        self.stop_button.bind("<Button-1>", lambda event: self.stop_chatbot())

        # Display welcome message
        self.display_message("Bot", "Hello! How can I assist you today?", "left")
    
    # Display messages in the chat area with alignment
    def display_message(self, sender, message, align):
        self.chat_area.configure(state='normal')
        if align == "left":
            self.chat_area.insert(tk.END, f"{sender}: {message}\n", "left")  # Left-align for bot
        else:
            self.chat_area.insert(tk.END, f"{sender}: {message}\n", "right")  # Right-align for user

        self.chat_area.tag_configure("left", justify="left", lmargin1=10, lmargin2=20, foreground="#4CAF50")  # Bot response in green
        self.chat_area.tag_configure("right", justify="right", rmargin=10, foreground="#E0E0E0")  # User response in gray

        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)  # Auto-scroll to the bottom
    
    # Retrieve user input from the entry widget
    def get_user_input(self):
        return self.entry.get()
    
    # Clear the entry widget after sending a message
    def clear_entry(self):
        self.entry.delete(0, tk.END)
    
    # Handle the Enter key being pressed
    def on_enter_pressed(self, event=None):
        self.process_input()
    
    # Process user input, call the chatbot logic, and display the response
    def process_input(self):
        user_input = self.get_user_input()
        if user_input:
            self.display_message("You", user_input, "right")  # User message on the right
            intents = pred_class(user_input, words, classes)
            response = get_response(intents, data)
            self.display_message("Bot", response, "left")  # Bot message on the left
            self.clear_entry()

    # Stop the chatbot with a termination message
    def stop_chatbot(self):
        self.display_message("Bot", "Terminating the chatbot. Goodbye!", "left")
        self.root.after(2000, self.root.quit)  # Wait for 2 seconds before closing the window

# Run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
