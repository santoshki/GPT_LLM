document.addEventListener("DOMContentLoaded", function () {

    const socket = new WebSocket("ws://127.0.0.1:8000/ws/chat");

    const chatArea = document.getElementById("chat-area");
    const input = document.getElementById("messageInput");
    const sendBtn = document.getElementById("sendBtn");
    const welcomeText = document.getElementById("welcomeText");

    let currentBotMessage = null;
    let firstMessageSent = false;

    function addMessage(content, className) {
        const msg = document.createElement("div");
        msg.classList.add("message", className);
        msg.textContent = content;
        chatArea.appendChild(msg);
        chatArea.scrollTop = chatArea.scrollHeight;
        return msg;
    }

    function sendMessage() {
    const message = input.value.trim();
    if (!message) return;

    const welcome = document.getElementById("welcomeText");

    // Hide heading on first message
    if (welcome && welcome.style.display !== "none") {
        welcome.style.opacity = "0";

        setTimeout(() => {
            welcome.style.display = "none";
        }, 300);
    }

    addMessage(message, "user");

    socket.send(message);
    input.value = "";

    currentBotMessage = addMessage("", "bot");
}

    sendBtn.onclick = sendMessage;

    input.addEventListener("keydown", function (e) {
        if (e.key === "Enter") sendMessage();
    });

    socket.onmessage = function (event) {
        if (currentBotMessage) {
            currentBotMessage.textContent = event.data;
        }
    };

});