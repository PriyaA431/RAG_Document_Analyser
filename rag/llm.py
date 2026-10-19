import ollama


class LocalLLM:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name

    def generate_answer(self, question, context):
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the question using the provided context. "
                        "If the answer is not in the context, say so "
                        "Do not invent facts."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {question}"
                    )
                }
            ]
        )

        return response["message"]["content"]