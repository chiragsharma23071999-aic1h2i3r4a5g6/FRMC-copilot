from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.vectorstore.retriever import search_documents
from app.memory.conversation_memory import ConversationMemory


# ============================================================
# INITIALIZE GPT MODEL
# ============================================================

llm = ChatOpenAI(
    model="gpt-5",
    temperature=0
)


# ============================================================
# INITIALIZE CONVERSATION MEMORY
# ============================================================

memory = ConversationMemory()


# ============================================================
# FRMC / SOX SYSTEM PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are an FRMC Compliance Intelligence Assistant.

Your role is to help compliance analysts understand:

- SOX compliance
- Financial Risk and Management Controls
- Internal controls
- Control testing
- Risk assessment
- Audit findings
- Control evidence
- Entity tiering
- ITGC controls
- Segregation of Duties
- Compliance processes

IMPORTANT RULES:

1. Use ONLY the information provided in the Retrieved Context.

2. Never invent facts.

3. Never invent:
   - Control IDs
   - Risk IDs
   - Audit IDs
   - Control owners
   - Evidence
   - Dates
   - Testing results
   - Conclusions

4. If the answer cannot be determined from the Retrieved Context, say:

"I could not find sufficient information in the available FRMC documents."

5. Give a clear, professional and concise answer.

6. When possible, explain the relationship between:
   - Risk
   - Control
   - Evidence
   - Testing
   - Entity Tier

7. Use conversation history to understand follow-up questions such as:
   - "it"
   - "that control"
   - "its owner"
   - "its frequency"
   - "what about its risk?"

8. Never answer from conversation history alone.
   Always verify the answer using Retrieved Context.

9. At the end of every answer, provide a Sources section.

10. The Sources section must contain ONLY filenames that actually
appear in the Retrieved Context.

11. Never invent source filenames.

12. If a retrieved document contains several records, use only
the record relevant to the user's question.

Conversation History:
{history}

Retrieved Context:
{context}

Current User Question:
{question}

Answer:
"""
)


# ============================================================
# FORMAT RETRIEVED DOCUMENTS
# ============================================================

def format_documents(results):
    """
    Convert ChromaDB results into a structured context
    for the LLM while preserving source information.
    """

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    formatted_documents = []

    for i, document in enumerate(documents):

        metadata = {}

        if i < len(metadatas):
            metadata = metadatas[i] or {}

        source = (
            metadata.get("file_name")
            or metadata.get("source")
            or metadata.get("file_path")
            or metadata.get("filename")
            or "Unknown source"
        )

        # Determine document category from filename.
        source_lower = source.lower()

        if "control_register" in source_lower:
            document_type = "CONTROL REGISTER"

        elif "risk_register" in source_lower:
            document_type = "RISK REGISTER"

        elif "evidence" in source_lower:
            document_type = "CONTROL EVIDENCE"

        elif "testing" in source_lower:
            document_type = "CONTROL TESTING"

        elif "tiering" in source_lower:
            document_type = "ENTITY TIERING"

        elif "audit" in source_lower:
            document_type = "INTERNAL AUDIT"

        elif "policy" in source_lower:
            document_type = "SOX POLICY"

        elif "process" in source_lower:
            document_type = "BUSINESS PROCESS"

        else:
            document_type = "ENTERPRISE DOCUMENT"

        formatted_documents.append(
            f"""
==================================================
{document_type}
==================================================

Source File:
{source}

Retrieved Content:
{document}
"""
        )

    return "\n".join(formatted_documents)


# ============================================================
# CREATE CONVERSATION-AWARE RETRIEVAL QUERY
# ============================================================

def create_retrieval_query(question, history):
    """
    Create a standalone retrieval query using conversation
    history when necessary.
    """

    if history == "No previous conversation.":

        return question

    retrieval_prompt = f"""
You are helping an FRMC document retrieval system.

Rewrite the current question into a standalone search query.

Use relevant identifiers and entities from the conversation
history when applicable.

Important entities may include:

- FRMC control IDs
- RISK IDs
- AUD IDs
- Process names
- Control names
- Evidence
- Testing
- Entity tiers

Do not answer the question.

Return ONLY the rewritten search query.

Conversation History:
{history}

Current Question:
{question}
"""

    response = llm.invoke(
        retrieval_prompt
    )

    return response.content.strip()


# ============================================================
# MAIN FRMC RAG FUNCTION
# ============================================================

def ask_frmc(question):
    """
    Retrieve relevant enterprise documents and generate
    a grounded FRMC compliance answer.
    """

    # --------------------------------------------------------
    # Get previous conversation history
    # --------------------------------------------------------

    history = memory.get_history_as_text()

    # --------------------------------------------------------
    # Create conversation-aware retrieval query
    # --------------------------------------------------------

    retrieval_query = create_retrieval_query(
        question,
        history
    )

    print()
    print("Retrieval query:")
    print(retrieval_query)

    # --------------------------------------------------------
    # Retrieve enterprise documents
    # --------------------------------------------------------

    results = search_documents(
        query=retrieval_query,
        top_k=5
    )

    # --------------------------------------------------------
    # Format retrieved documents
    # --------------------------------------------------------

    context = format_documents(
        results
    )

    # --------------------------------------------------------
    # Build final prompt
    # --------------------------------------------------------

    messages = prompt.invoke(
        {
            "history": history,
            "context": context,
            "question": question
        }
    )

    # --------------------------------------------------------
    # Generate grounded answer
    # --------------------------------------------------------

    response = llm.invoke(
        messages
    )

    # --------------------------------------------------------
    # Store conversation
    # --------------------------------------------------------

    memory.add_user_message(
        question
    )

    memory.add_assistant_message(
        response.content
    )

    return response.content


# ============================================================
# COMMAND-LINE APPLICATION
# ============================================================

if __name__ == "__main__":

    print("===================================")
    print("FRMC COMPLIANCE INTELLIGENCE ASSISTANT")
    print("===================================")
    print("Type 'exit' to quit.")
    print()

    while True:

        question = input(
            "Enter your question: "
        )

        if question.lower() == "exit":

            print()

            print(
                "Thank you for using "
                "FRMC Compliance Intelligence Assistant."
            )

            break

        answer = ask_frmc(
            question
        )

        print()

        print("===================================")
        print("ANSWER")
        print("===================================")
        print(answer)

        print()