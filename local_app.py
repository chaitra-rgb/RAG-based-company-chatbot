import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import rag_engine


HOST = "127.0.0.1"
PORT = 8502
BASE_DIR = Path(__file__).resolve().parent


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Company RAG Chatbot</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, sans-serif;
      --border: #d7dde5;
      --text: #172033;
      --muted: #657085;
      --panel: #f7f9fc;
      --accent: #1769aa;
    }
    body { margin: 0; color: var(--text); background: #ffffff; }
    main { max-width: 920px; margin: 0 auto; padding: 28px 18px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: start; }
    h1 { margin: 0 0 6px; font-size: 28px; }
    .status { color: var(--muted); font-size: 14px; }
    .chat { margin-top: 22px; border: 1px solid var(--border); border-radius: 8px; min-height: 380px; padding: 16px; background: var(--panel); }
    .message { max-width: 78%; margin: 10px 0; padding: 11px 13px; border-radius: 8px; line-height: 1.45; white-space: pre-wrap; }
    .user { margin-left: auto; background: #dceeff; }
    .bot { background: #ffffff; border: 1px solid var(--border); }
    form { display: flex; gap: 10px; margin-top: 14px; }
    input { flex: 1; padding: 12px; border: 1px solid var(--border); border-radius: 8px; font-size: 15px; }
    button { border: 1px solid var(--border); background: #ffffff; border-radius: 8px; padding: 10px 14px; cursor: pointer; font-size: 15px; }
    button.primary { background: var(--accent); color: #ffffff; border-color: var(--accent); }
    .actions { display: flex; gap: 8px; margin-top: 10px; }
    .actions button { width: 48px; height: 40px; padding: 0; font-size: 18px; }
    textarea { width: 100%; margin-top: 10px; min-height: 76px; border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
    .hidden { display: none; }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Company RAG Chatbot</h1>
        <div class="status" id="status">Loading knowledge base status...</div>
      </div>
    </header>

    <section class="chat" id="chat"></section>

    <form id="ask-form">
      <input id="question" autocomplete="off" placeholder="Ask about the uploaded company documents">
      <button class="primary" type="submit">Ask</button>
    </form>

    <div class="actions">
      <button title="Like" data-action="like">👍</button>
      <button title="Dislike" data-action="dislike">👎</button>
      <button title="Save" data-action="save">🔖</button>
      <button title="Share" data-action="share">↗</button>
    </div>
    <textarea class="hidden" id="share-text" readonly></textarea>
  </main>

  <script>
    const chat = document.getElementById("chat");
    const form = document.getElementById("ask-form");
    const input = document.getElementById("question");
    const shareText = document.getElementById("share-text");
    let lastAnswer = "";
    let messages = [];

    function addMessage(role, content) {
      const div = document.createElement("div");
      div.className = "message " + (role === "user" ? "user" : "bot");
      div.textContent = content;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      messages.push({ role, content });
    }

    async function loadStatus() {
      const response = await fetch("/status");
      const status = await response.json();
      document.getElementById("status").textContent =
        status.has_index
          ? `Knowledge base ready: ${status.local_chunks} chunks`
          : "No knowledge base found. Process documents in Streamlit or add local_index.json.";
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = input.value.trim();
      if (!question) return;
      addMessage("user", question);
      input.value = "";
      addMessage("assistant", "Searching...");
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ question })
      });
      const data = await response.json();
      chat.lastChild.textContent = data.answer;
      messages[messages.length - 1].content = data.answer;
      lastAnswer = data.answer;
      shareText.classList.add("hidden");
    });

    document.querySelectorAll(".actions button").forEach((button) => {
      button.addEventListener("click", async () => {
        if (!lastAnswer) return;
        const action = button.dataset.action;
        await fetch("/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: new URLSearchParams({ action, answer: lastAnswer, messages: JSON.stringify(messages.slice(-4)) })
        });
        if (action === "share") {
          shareText.value = lastAnswer;
          shareText.classList.remove("hidden");
          shareText.focus();
          shareText.select();
        }
      });
    });

    loadStatus();
  </script>
</body>
</html>
"""


class LocalChatbotHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: str, content_type: str = "text/html") -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, HTML)
            return
        if self.path == "/status":
            self._send(200, json.dumps(rag_engine.index_status()), "application/json")
            return
        self._send(404, "Not found", "text/plain")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        form = parse_qs(self.rfile.read(length).decode("utf-8"))

        if self.path == "/ask":
            question = form.get("question", [""])[0].strip()
            if not question:
                answer = "Enter a question about the company documents."
            elif not rag_engine.has_index():
                answer = "No document index found. Process documents first."
            else:
                answer = rag_engine.answer_question(question, company_name="the company")["answer"]
            self._send(200, json.dumps({"answer": answer}), "application/json")
            return

        if self.path == "/feedback":
            action = form.get("action", [""])[0]
            answer = form.get("answer", [""])[0]
            try:
                messages = json.loads(form.get("messages", ["[]"])[0])
            except json.JSONDecodeError:
                messages = []

            if action == "save":
                rag_engine.save_answer(answer=answer, company_name="the company", messages=messages)
            else:
                rag_engine.save_feedback_action(action=action, company_name="the company", messages=messages)
            self._send(200, json.dumps({"ok": True}), "application/json")
            return

        self._send(404, "Not found", "text/plain")

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), LocalChatbotHandler)
    print(f"Local chatbot running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
