document.addEventListener("DOMContentLoaded", function () {

    const chatArea = document.getElementById("chatStream");
    const input = document.getElementById("messageInput");
    const sendBtn = document.getElementById("sendBtn");
    const stopBtn = document.getElementById("stopBtn");
    const newChatBtn = document.querySelector(".new-chat");

    // 🔥 UPDATED: use badge instead of label
    const statusBadge = document.getElementById("statusBadge");

    let currentBotMessage = null;
    let typingTimeout = null;
    let controller = null;
    let isStopped = false;

    // =========================
    // STATUS HANDLER 🔥 (TEXT + COLOR)
    // =========================
    function setStatus(state) {
        if (!statusBadge) return;

        // Reset classes
        statusBadge.classList.remove(
            "status-ready",
            "status-thinking",
            "status-typing"
        );

        switch (state) {
            case "ready":
                statusBadge.textContent = "🟢 Ready";
                statusBadge.classList.add("status-ready");
                break;

            case "thinking":
                statusBadge.textContent = "🟡 Thinking...";
                statusBadge.classList.add("status-thinking");
                break;

            case "typing":
                statusBadge.textContent = "🔵 Typing...";
                statusBadge.classList.add("status-typing");
                break;
        }
    }

    // =========================
    // Add Message
    // =========================
    function addMessage(content, className) {
        const msg = document.createElement("div");
        msg.classList.add("message", className);
        msg.textContent = content;

        chatArea.appendChild(msg);

        chatArea.scrollTo({
            top: chatArea.scrollHeight,
            behavior: "smooth"
        });

        return msg;
    }

    // =========================
    // Button Toggle
    // =========================
    function showStop() {
        sendBtn.style.display = "none";
        if (stopBtn) stopBtn.style.display = "inline-flex";
    }

    function showSend() {
        sendBtn.style.display = "inline-flex";
        if (stopBtn) stopBtn.style.display = "none";
    }

    // =========================
    // Stop Everything
    // =========================
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

        setStatus("ready"); // ✅ reset state
        showSend();
    }

    // =========================
    // Typing Effect
    // =========================
    function typeMessage(element, text) {
        let index = 0;
        element.textContent = "";

        setStatus("typing"); // 🔵

        function type() {
            if (isStopped) return;

            if (index < text.length) {
                element.textContent += text.charAt(index);
                index++;

                chatArea.scrollTop = chatArea.scrollHeight;
                typingTimeout = setTimeout(type, 12);
            } else {
                setStatus("ready"); // 🟢
                showSend();
            }
        }

        type();
    }

    // =========================
    // Send Message
    // =========================
    async function sendMessage() {
        const message = input.value.trim();
        if (!message) return;

        isStopped = false;
        showStop();
        setStatus("thinking"); // 🟡

        // Hide welcome text
        const welcome = document.getElementById("welcomeText");
        if (welcome && welcome.style.display !== "none") {
            welcome.style.opacity = "0";
            setTimeout(() => {
                welcome.style.display = "none";
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
                body: JSON.stringify({ message }),
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
                    currentBotMessage.textContent = "⚠️ Error getting response.";
                }
            }

            setStatus("ready"); // fallback
            showSend();
        }
    }

    // =========================
    // NEW CHAT
    // =========================
    function startNewChat() {

        stopAll();
        isStopped = false;
        currentBotMessage = null;

        chatArea.style.transition = "all 0.2s ease";
        chatArea.style.opacity = "0";
        chatArea.style.transform = "translateY(10px)";

        setTimeout(() => {

            chatArea.innerHTML = "";

            const welcome = document.createElement("div");
            welcome.className = "welcome-text";
            welcome.id = "welcomeText";
            welcome.textContent = "Hi, Good day. How may I help you?";

            chatArea.appendChild(welcome);

            input.value = "";
            input.focus();

            showSend();
            setStatus("ready"); // 🟢 reset

            chatArea.style.opacity = "1";
            chatArea.style.transform = "translateY(0px)";

        }, 200);
    }

    // =========================
    // Events
    // =========================
    sendBtn.addEventListener("click", sendMessage);

    if (stopBtn) {
        stopBtn.addEventListener("click", stopAll);
        stopBtn.style.display = "none";
    }

    if (newChatBtn) {
        newChatBtn.addEventListener("click", startNewChat);
    }

    input.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
            e.preventDefault();
            sendMessage();
        }
    });

    // =========================
    // INIT
    // =========================
    setStatus("ready"); // 🟢 default
    input.focus();

});