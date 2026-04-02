document.addEventListener("DOMContentLoaded", function () {

    const chatArea = document.getElementById("chat-area");
    const input = document.getElementById("messageInput");
    const sendBtn = document.getElementById("sendBtn");
    const stopBtn = document.getElementById("stopBtn");
    const welcomeText = document.getElementById("welcomeText");

    let currentBotMessage = null;
    let typingTimeout = null;
    let controller = null;
    let isStopped = false;

    function addMessage(content, className) {
        const msg = document.createElement("div");
        msg.classList.add("message", className);
        msg.textContent = content;
        chatArea.appendChild(msg);
        chatArea.scrollTop = chatArea.scrollHeight;
        return msg;
    }

    function showStop() {
        sendBtn.style.display = "none";
        stopBtn.style.display = "inline-block";
    }

    function showSend() {
        sendBtn.style.display = "inline-block";
        stopBtn.style.display = "none";
    }

    function stopAll() {
        isStopped = true;

        if (typingTimeout) {
            clearTimeout(typingTimeout);
            typingTimeout = null;
        }

        if (controller) {
            controller.abort();
            controller = null;
        }

        showSend();
    }

    function typeMessage(element, text) {
        let index = 0;
        element.textContent = "";

        function type() {
            if (isStopped) return;

            if (index < text.length) {
                element.textContent += text.charAt(index);
                index++;
                chatArea.scrollTop = chatArea.scrollHeight;

                typingTimeout = setTimeout(type, 10);
            } else {
                showSend();
            }
        }

        type();
    }

    async function sendMessage() {
        const message = input.value.trim();
        if (!message) return;

        isStopped = false;
        showStop();

        if (welcomeText && welcomeText.style.display !== "none") {
            welcomeText.style.opacity = "0";
            setTimeout(() => {
                welcomeText.style.display = "none";
            }, 300);
        }

        addMessage(message, "user");
        input.value = "";

        currentBotMessage = addMessage("Thinking...", "bot");

        try {
            controller = new AbortController();

            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message: message }),
                signal: controller.signal
            });

            const data = await response.json();

            if (!isStopped && currentBotMessage) {
                typeMessage(currentBotMessage, data.response);
            }

        } catch (error) {
            if (error.name !== "AbortError") {
                console.error(error);
                if (currentBotMessage) {
                    currentBotMessage.textContent = "Error getting response.";
                }
            }
            showSend();
        }
    }

    sendBtn.onclick = sendMessage;
    stopBtn.onclick = stopAll;

    input.addEventListener("keydown", function (e) {
        if (e.key === "Enter") sendMessage();
    });

});