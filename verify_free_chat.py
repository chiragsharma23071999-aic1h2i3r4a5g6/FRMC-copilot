"""One live free-model request. Prints diagnostics without keys or provider error bodies."""
from chat_service import conversational_answer
from knowledge import reference_documents


if __name__ == "__main__":
    question = "What is SOX? Explain in one short sentence."
    result = conversational_answer(question, reference_documents("What is SOX?"))
    print("Answer mode:", result["answer_mode"])
    print("Model:", result.get("model"))
    print("Response characters:", len(result["answer"]))
    if result.get("notice"):
        print("Status:", result["notice"])
    raise SystemExit(0 if result["answer_mode"] == "free_model" else 1)
