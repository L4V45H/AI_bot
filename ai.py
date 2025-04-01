import tkinter as tk
from tkinter import scrolledtext, ttk, font
import threading
from vosk import Model, KaldiRecognizer
import pyaudio
from llama_cpp import Llama

class StreamText(scrolledtext.ScrolledText):
    """Виджет с потоковым выводом текста"""
    def stream(self, text: str, delay: int = 1):
        self.insert(tk.END, text)
        self.see(tk.END)
        self.update()

class VoiceAssistant:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        self.setup_audio()
        self.setup_llm()
        self.history = []
        
    def setup_ui(self):
        """Настройка современного интерфейса"""
        self.root.title("Mistral AI Assistant")
        self.root.geometry("900x700")
        self.root.configure(bg="#f5f5f5")
        
        # Шрифты
        self.main_font = font.Font(family="Segoe UI", size=12)
        self.button_font = font.Font(family="Segoe UI", size=11, weight="bold")
        
        # Основной чат
        self.chat_frame = tk.Frame(self.root, bg="#ffffff", bd=2, relief=tk.GROOVE)
        self.chat_frame.pack(pady=15, padx=15, fill=tk.BOTH, expand=True)
        
        self.chat_area = StreamText(
            self.chat_frame, wrap=tk.WORD, width=100, height=30,
            font=self.main_font, bg="#ffffff", fg="#333333",
            padx=10, pady=10
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True)
        
        # Панель ввода
        input_frame = tk.Frame(self.root, bg="#f5f5f5")
        input_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        self.input_entry = tk.Text(
            input_frame, height=4, wrap=tk.WORD,
            font=self.main_font, bg="#ffffff"
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Кнопки
        button_frame = tk.Frame(input_frame, bg="#f5f5f5")
        button_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.send_btn = tk.Button(
            button_frame, text="Отправить (Ctrl+Enter)", 
            command=self.send_text, bg="#4CAF50", fg="white",
            font=self.button_font, padx=10
        )
        self.send_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.record_btn = tk.Button(
            button_frame, text="🎤 Голосовой ввод", 
            command=self.start_recording, bg="#2196F3", fg="white",
            font=self.button_font, padx=10
        )
        self.record_btn.pack(fill=tk.X, pady=5)
        
        self.clear_btn = tk.Button(
            button_frame, text="Очистить чат", 
            command=self.clear_chat, bg="#f44336", fg="white",
            font=self.button_font, padx=10
        )
        self.clear_btn.pack(fill=tk.X)
        
        # Статус бар
        self.status_var = tk.StringVar(value="✅ Система готова")
        tk.Label(
            self.root, textvariable=self.status_var, 
            bg="#f5f5f5", font=self.main_font
        ).pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=5)
        
        # Горячие клавиши
        self.root.bind("<Control-Return>", lambda e: self.send_text())
        
    def setup_audio(self):
        """Инициализация голосового ввода"""
        self.model_stt = Model("vosk-model-small-ru-0.22")
        self.recognizer = KaldiRecognizer(self.model_stt, 16000)
        self.mic = pyaudio.PyAudio()
        self.stream = self.mic.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8192
        )
    
    def setup_llm(self):
        """Оптимизированная инициализация модели"""
        self.llm = Llama(
            model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            n_ctx=4096,
            n_threads=6,
            n_batch=2048,
            n_gpu_layers=0,
            offload_kqv=True,
            verbose=False
        )
    
    def send_text(self):
        """Обработка текстового ввода"""
        text = self.input_entry.get("1.0", tk.END).strip()
        if not text:
            return
            
        self.input_entry.delete("1.0", tk.END)
        self.chat_area.stream(f"Вы: {text}\n\n", delay=0)
        self.history.append({"role": "user", "content": text})
        
        threading.Thread(target=self.generate_response, daemon=True).start()
    
    def start_recording(self):
        """Запуск голосового ввода"""
        self.status_var.set("🎤 Говорите...")
        self.record_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.record_voice, daemon=True).start()
    
    def record_voice(self):
        """Обработка голосового ввода"""
        try:
            while True:
                data = self.stream.read(4096)
                if self.recognizer.AcceptWaveform(data):
                    text = self.recognizer.Result()[14:-3]
                    if text:
                        self.chat_area.stream(f"Вы: {text}\n\n")
                        self.history.append({"role": "user", "content": text})
                        self.generate_response(text)
                    break
        finally:
            self.record_btn.config(state=tk.NORMAL)
            self.status_var.set("✅ Готов к работе")
    
    def generate_response(self, prompt=None):
        """Генерация ответа с потоковым выводом"""
        self.status_var.set("🤖 Mistral генерирует ответ...")
        self.root.config(cursor="watch")
        
        try:
            full_response = ""
            for chunk in self.llm.create_chat_completion(
                messages=self.history[-10:],  # Ограничиваем историю
                max_tokens=1024,
                temperature=0.7,
                stream=True
            ):
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    full_response += delta["content"]
                    self.chat_area.stream(delta["content"])
            
            self.history.append({"role": "assistant", "content": full_response})
            self.chat_area.stream("\n\n")
            
        finally:
            self.status_var.set("✅ Готов к работе")
            self.root.config(cursor="")
    
    def clear_chat(self):
        """Очистка чата"""
        self.chat_area.delete("1.0", tk.END)
        self.history = []

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAssistant(root)
    root.mainloop()