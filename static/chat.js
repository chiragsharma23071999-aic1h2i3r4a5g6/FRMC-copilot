const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-question");
const chatMessages = document.getElementById("chat-messages");
const clearChat = document.getElementById("clear-chat");
let chatHistory = [];

function addChatMessage(label, content) {
    const item = document.createElement("p");
    item.style.whiteSpace = "pre-wrap";
    const heading = document.createElement("strong");
    heading.textContent = label + ": ";
    item.append(heading, document.createTextNode(content));
    chatMessages.appendChild(item);
    return item;
}

clearChat.addEventListener("click", () => {
    chatHistory = [];
    chatMessages.replaceChildren();
    chatInput.focus();
});

chatForm.addEventListener("submit", async event => {
    event.preventDefault();
    const question = chatInput.value.trim();
    if (!question) return;
    const send = chatForm.querySelector('button[type="submit"]');
    send.disabled = true;
    clearChat.disabled = true;
    addChatMessage("You", question);
    const waiting = addChatMessage("Copilot", "Thinking...");
    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({question, history: chatHistory.slice(-12)})
        });
        const data = await response.json();
        waiting.remove();
        if (!response.ok) {
            addChatMessage("Error", data.error || "Unable to answer.");
            return;
        }
        addChatMessage("Copilot", data.answer);
        if (data.notice) addChatMessage("Service status", data.notice);
        if (data.sources && data.sources.length) {
            addChatMessage("Reference material", data.sources.map(source =>
                source.title + (source.url ? " — " + source.url : "")
            ).join("\n"));
        }
        chatHistory.push({role: "user", content: question}, {role: "assistant", content: data.answer.slice(0, 8000)});
        chatHistory = chatHistory.slice(-12);
        chatInput.value = "";
    } catch (error) {
        waiting.remove();
        addChatMessage("Error", "Unable to reach the application. Please try again.");
    } finally {
        send.disabled = false;
        clearChat.disabled = false;
        chatInput.focus();
    }
});
