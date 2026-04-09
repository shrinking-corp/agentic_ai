import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk


def _format_exception_message(exc: BaseException) -> str:
    if hasattr(exc, "exceptions"):
        nested_messages = []
        for nested_exc in exc.exceptions:
            nested_message = _format_exception_message(nested_exc)
            if nested_message:
                nested_messages.append(nested_message)
        if nested_messages:
            return " | ".join(nested_messages)

    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


class AgenticAiGui:
    def __init__(self, agent_runtime):
        self.agent_runtime = agent_runtime
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.conversation_history: list[dict[str, str]] = []

        self.root = tk.Tk()
        self.root.title("Agentic AI Chat")
        self.root.geometry("1220x820")
        self.root.configure(bg="#f3efe6")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status = tk.StringVar(value="Ready")
        self.mcp_status = tk.StringVar(value="MCP: initializing...")
        self.tool_indicator = tk.StringVar(value="Tool was not used")

        self._build_layout()
        self._start_backend_initialization()
        self.root.after(100, self._drain_events)

    def _build_layout(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3efe6")
        style.configure("TLabel", background="#f3efe6", foreground="#1d2433")
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 18), background="#f3efe6")
        style.configure("TButton", padding=8)
        style.configure("TCombobox", padding=6)

        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        header = ttk.Label(container, text="Agentic AI Chat", style="Header.TLabel")
        header.grid(row=0, column=0, sticky="w")

        controls = ttk.Frame(container, padding=(0, 12, 0, 12))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(
            controls,
            text="General AI chat. The MCP server is connected in the background.",
        ).grid(row=0, column=0, sticky="w")

        self.clear_button = ttk.Button(controls, text="Clear chat", command=self._clear_chat)
        self.clear_button.grid(row=0, column=1, padx=(16, 0), sticky="e")

        self.tool_indicator_label = tk.Label(
            controls,
            textvariable=self.tool_indicator,
            font=("Segoe UI Semibold", 10),
            bg="#f3efe6",
            fg="#8b1e3f",
        )
        self.tool_indicator_label.grid(row=1, column=0, columnspan=2, pady=(12, 0), sticky="w")

        self.mcp_status_label = tk.Label(
            controls,
            textvariable=self.mcp_status,
            font=("Segoe UI Semibold", 10),
            bg="#f3efe6",
            fg="#8a5a00",
        )
        self.mcp_status_label.grid(row=2, column=0, columnspan=2, pady=(6, 0), sticky="w")

        body = ttk.Frame(container)
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(1, weight=4)
        body.rowconfigure(3, weight=1)

        ttk.Label(body, text="Conversation history").grid(row=0, column=0, sticky="w")
        ttk.Label(body, text="Agent logs").grid(row=0, column=1, sticky="w", padx=(16, 0))

        self.chat_text = tk.Text(body, wrap="word", font=("Segoe UI", 10), bg="#fffdf8", fg="#1d2433")
        self.chat_text.grid(row=1, column=0, sticky="nsew")
        self.chat_text.configure(state="disabled")
        self.chat_text.tag_configure("user", foreground="#7a1f1f", font=("Segoe UI Semibold", 10))
        self.chat_text.tag_configure("assistant", foreground="#1b4d8a", font=("Segoe UI Semibold", 10))
        self.chat_text.tag_configure("body", foreground="#1d2433", font=("Segoe UI", 10))

        self.log_text = tk.Text(body, wrap="word", font=("Consolas", 10), bg="#10151f", fg="#d6e2ff")
        self.log_text.grid(row=1, column=1, rowspan=3, padx=(16, 0), sticky="nsew")
        self.log_text.configure(state="disabled")

        ttk.Label(body, text="New message").grid(row=2, column=0, pady=(16, 0), sticky="w")
        self.input_text = tk.Text(body, height=8, wrap="word", font=("Segoe UI", 10), bg="#fffdf8", fg="#1d2433")
        self.input_text.grid(row=3, column=0, sticky="nsew", pady=(6, 0))
        self.input_text.bind("<Control-Return>", self._handle_ctrl_enter)

        actions = ttk.Frame(body)
        actions.grid(row=4, column=0, pady=(16, 0), sticky="e")
        self.send_button = ttk.Button(actions, text="Send", command=self._start_run)
        self.send_button.pack(anchor="e")

        status_bar = ttk.Label(container, textvariable=self.status)
        status_bar.grid(row=3, column=0, sticky="w", pady=(12, 0))

    def _handle_ctrl_enter(self, _event):
        self._start_run()
        return "break"

    def _start_backend_initialization(self) -> None:
        self.status.set("Initializing MCP server...")
        worker = threading.Thread(target=self._initialize_backend, daemon=True)
        worker.start()

    def _initialize_backend(self) -> None:
        try:
            status = self.agent_runtime.initialize(log_callback=self._enqueue_log)
            self.events.put(("backend_status", status))
            self.events.put(("status", "Ready"))
        except Exception as exc:
            self.events.put(("error", _format_exception_message(exc)))
            self.events.put(("backend_status", {"mcp_available": False, "error": _format_exception_message(exc)}))
            self.events.put(("status", "Initialization failed"))

    def _set_mcp_status(self, mcp_available: bool, tool_names: list[str] | None = None, error: str = "") -> None:
        if mcp_available:
            tool_suffix = f" ({', '.join(tool_names)})" if tool_names else ""
            self.mcp_status.set(f"MCP: connected{tool_suffix}")
            self.mcp_status_label.configure(fg="#1b6e3c")
        else:
            suffix = f" - {error}" if error else ""
            self.mcp_status.set(f"MCP: fallback without tools{suffix}")
            self.mcp_status_label.configure(fg="#8b1e3f")

    def _set_busy(self, is_busy: bool) -> None:
        button_state = "disabled" if is_busy else "normal"
        self.send_button.configure(state=button_state)
        self.clear_button.configure(state=button_state)

    def _clear_chat(self) -> None:
        self.conversation_history.clear()
        self.chat_text.configure(state="normal")
        self.chat_text.delete("1.0", tk.END)
        self.chat_text.configure(state="disabled")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")
        self._set_tool_indicator(False, "")
        self.status.set("Chat was cleared")

    def _append_chat_message(self, role: str, content: str) -> None:
        label = "User" if role == "user" else "Assistant"
        tag = "user" if role == "user" else "assistant"

        self.chat_text.configure(state="normal")
        self.chat_text.insert(tk.END, f"{label}:\n", tag)
        self.chat_text.insert(tk.END, f"{content.strip()}\n\n", "body")
        self.chat_text.see(tk.END)
        self.chat_text.configure(state="disabled")

    def _start_run(self) -> None:
        user_prompt = self.input_text.get("1.0", tk.END).strip()
        if not user_prompt:
            messagebox.showwarning("Error", "Write a message for the assistant first.")
            return

        history_snapshot = list(self.conversation_history)
        self.conversation_history.append({"role": "user", "content": user_prompt})
        self._append_chat_message("user", user_prompt)
        self.input_text.delete("1.0", tk.END)
        self._set_tool_indicator(False, "")
        self.status.set("Processing response...")
        self._set_busy(True)

        worker = threading.Thread(
            target=self._run_in_background,
            args=(history_snapshot, user_prompt),
            daemon=True,
        )
        worker.start()

    def _run_in_background(
        self,
        conversation_history: list[dict[str, str]],
        user_prompt: str,
    ) -> None:
        try:
            result = self.agent_runtime.run_chat(
                user_prompt=user_prompt,
                conversation_history=conversation_history,
                log_callback=self._enqueue_log,
                metadata_callback=self._enqueue_metadata,
            )
            self.events.put(("result", result))
            self.events.put(("status", "Done"))
        except Exception as exc:
            self.events.put(("error", _format_exception_message(exc)))
            self.events.put(("status", "Run failed"))

    def _enqueue_log(self, message: str) -> None:
        self.events.put(("log", message))

    def _enqueue_metadata(self, metadata: dict) -> None:
        self.events.put(("metadata", metadata))

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _set_tool_indicator(self, tool_used: bool, algorithm: str) -> None:
        if tool_used:
            detail_suffix = f" ({algorithm})" if algorithm else ""
            self.tool_indicator.set(f"Tool was used{detail_suffix}")
            self.tool_indicator_label.configure(fg="#1b6e3c")
        else:
            self.tool_indicator.set("Tool was not used")
            self.tool_indicator_label.configure(fg="#8b1e3f")

    def _drain_events(self) -> None:
        while True:
            try:
                event_type, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if event_type == "log":
                self._append_log(payload)
            elif event_type == "metadata":
                detail = payload.get("tool_name") or payload.get("algorithm", "")
                self._set_tool_indicator(payload.get("tool_used", False), detail)
            elif event_type == "backend_status":
                self._set_mcp_status(
                    payload.get("mcp_available", False),
                    payload.get("tool_names", []),
                    payload.get("error", ""),
                )
            elif event_type == "result":
                self.conversation_history.append({"role": "assistant", "content": payload})
                self._append_chat_message("assistant", payload)
            elif event_type == "error":
                self._append_log(f"[Error] {payload}")
                messagebox.showerror("Processing failed", payload)
            elif event_type == "status":
                self.status.set(payload)
                if payload in {"Done", "Run failed"}:
                    self._set_busy(False)

        self.root.after(100, self._drain_events)

    def run(self) -> None:
        self.root.mainloop()

    def _on_close(self) -> None:
        self.agent_runtime.shutdown()
        self.root.destroy()


def launch_gui(agent_runtime) -> None:
    AgenticAiGui(agent_runtime).run()