import { useState } from "react";

export function useChat() {
    const [messages, setMessages] = useState([]);
    // TODO: implement chat logic
    return { messages, setMessages };
}
