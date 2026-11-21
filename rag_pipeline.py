        semantic_docs = self.faiss_retriever.invoke(query)
        keyword_docs = self.bm25_retriever.invoke(query)

        # Reciprocal Rank Fusion: equal contribution from semantic and keyword search.
        fused = {}
        for weight, results in ((0.5, semantic_docs), (0.5, keyword_docs)):
            seen = set()
            for rank, doc in enumerate(results, start=1):
                # Identical product text identifies the same record across indexes.
                key = str(doc.metadata.get("product_id", ""))
                if not key or key == "nan":
                    key = doc.page_content
                if key in seen:
                    continue
                seen.add(key)
                if key not in fused:
                    fused[key] = {"document": doc, "score": 0.0}
                fused[key]["score"] += weight / (60 + rank)

        if not fused:
            return []

        # RRF orders candidates; the cross-encoder then chooses the best three.
        candidates = sorted(
            fused.values(), key=lambda item: item["score"], reverse=True
        )
        docs = [item["document"] for item in candidates]
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(
            zip(docs, scores), key=lambda item: float(item[1]), reverse=True
        )
        return [doc for doc, _ in ranked[: self.top_n]]


def build_retriever(faiss_store, bm25_retriever):
    print("Building hybrid FAISS + BM25 + RRF retriever...")
    return HybridRerankingRetriever(faiss_store, bm25_retriever)


class LocalRAGChain:
    """Preserves chain.invoke({'input': ...}) -> answer/context interface."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def invoke(self, inputs):
        query = inputs["input"]
        documents = self.retriever.invoke(query)
        if not documents:
            return {
                "input": query,
                "context": [],
                "answer": "Sorry, I couldn't find matching products in the catalog.",
            }

        context = "\n\n".join(doc.page_content for doc in documents)
        messages = PROMPT.format_messages(context=context, input=query)
        response = self.llm.invoke(messages)
        return {"input": query, "context": documents, "answer": response.content}


def build_rag_chain(retriever):
    print(f"Using local Ollama model: {OLLAMA_MODEL}")
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.3)
    return LocalRAGChain(retriever, llm)


def load_retriever():
    embeddings = load_embedding_model()
    faiss_store = load_faiss_store(embeddings)
    bm25_retriever = load_bm25_retriever()
    return build_retriever(faiss_store, bm25_retriever)


def load_rag_chain():
    return build_rag_chain(load_retriever())


def ask(chain, query):
    return chain.invoke({"input": query})


def main():
    print("Starting free local RAG Shopping Assistant...")
    chain = load_rag_chain()
    while True:
        try:
            query = input("\nEnter your shopping question (or exit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        try:
            result = ask(chain, query)
            print("\nAnswer:\n", result["answer"])
            print("Retrieved products:", len(result["context"]))
        except Exception as exc:
            print(f"Error processing question: {exc}")


if __name__ == "__main__":
    main()
