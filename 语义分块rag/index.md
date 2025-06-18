# RAG系列-语义分块RAG（Semantic Chunking RAG）

# 02. 语义分块RAG（Semantic Chunking RAG）

## 方法简介
语义分块RAG通过计算句子间的语义相似度来智能分块，而不是简单的固定长度分块。它使用百分位数、标准差或四分位距等方法找到语义断点，将文本分割成语义连贯的块，提升检索精度。

## 核心代码
```python
import fitz
import os
import numpy as np
import json
from openai import OpenAI

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    # Open the PDF file
    mypdf = fitz.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text

    # Iterate through each page in the PDF
    for page in mypdf:
        # Extract text from the current page and add spacing
        all_text += page.get_text("text") + " "

    # Return the extracted text, stripped of leading/trailing whitespace
    return all_text.strip()

# Initialize the OpenAI client with the base URL and API key
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # Retrieve the API key from environment variables
)

def get_embedding(text, model="BAAI/bge-en-icl"):
    """
    Creates an embedding for the given text using OpenAI.

    Args:
    text (str): Input text.
    model (str): Embedding model name.

    Returns:
    np.ndarray: The embedding vector.
    """
    response = client.embeddings.create(model=model, input=text)
    return np.array(response.data[0].embedding)

def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): First vector.
    vec2 (np.ndarray): Second vector.

    Returns:
    float: Cosine similarity.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def compute_breakpoints(similarities, method="percentile", threshold=90):
    """
    Computes chunking breakpoints based on similarity drops.

    Args:
    similarities (List[float]): List of similarity scores between sentences.
    method (str): 'percentile', 'standard_deviation', or 'interquartile'.
    threshold (float): Threshold value (percentile for 'percentile', std devs for 'standard_deviation').

    Returns:
    List[int]: Indices where chunk splits should occur.
    """
    # Determine the threshold value based on the selected method
    if method == "percentile":
        # Calculate the Xth percentile of the similarity scores
        threshold_value = np.percentile(similarities, threshold)
    elif method == "standard_deviation":
        # Calculate the mean and standard deviation of the similarity scores
        mean = np.mean(similarities)
        std_dev = np.std(similarities)
        # Set the threshold value to mean minus X standard deviations
        threshold_value = mean - (threshold * std_dev)
    elif method == "interquartile":
        # Calculate the first and third quartiles (Q1 and Q3)
        q1, q3 = np.percentile(similarities, [25, 75])
        # Set the threshold value using the IQR rule for outliers
        threshold_value = q1 - 1.5 * (q3 - q1)
    else:
        # Raise an error if an invalid method is provided
        raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")

    # Identify indices where similarity drops below the threshold value
    return [i for i, sim in enumerate(similarities) if sim < threshold_value]

def split_into_chunks(sentences, breakpoints):
    """
    Splits sentences into semantic chunks.

    Args:
    sentences (List[str]): List of sentences.
    breakpoints (List[int]): Indices where chunking should occur.

    Returns:
    List[str]: List of text chunks.
    """
    chunks = []  # Initialize an empty list to store the chunks
    start = 0  # Initialize the start index

    # Iterate through each breakpoint to create chunks
    for bp in breakpoints:
        # Append the chunk of sentences from start to the current breakpoint
        chunks.append(". ".join(sentences[start:bp + 1]) + ".")
        start = bp + 1  # Update the start index to the next sentence after the breakpoint

    # Append the remaining sentences as the last chunk
    chunks.append(". ".join(sentences[start:]))
    return chunks  # Return the list of chunks

def create_embeddings(text_chunks):
    """
    Creates embeddings for each text chunk.

    Args:
    text_chunks (List[str]): List of text chunks.

    Returns:
    List[np.ndarray]: List of embedding vectors.
    """
    # Generate embeddings for each text chunk using the get_embedding function
    return [get_embedding(chunk) for chunk in text_chunks]

def semantic_search(query, text_chunks, chunk_embeddings, k=5):
    """
    Finds the most relevant text chunks for a query.

    Args:
    query (str): Search query.
    text_chunks (List[str]): List of text chunks.
    chunk_embeddings (List[np.ndarray]): List of chunk embeddings.
    k (int): Number of top results to return.

    Returns:
    List[str]: Top-k relevant chunks.
    """
    # Generate an embedding for the query
    query_embedding = get_embedding(query)

    # Calculate cosine similarity between the query embedding and each chunk embedding
    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]

    # Get the indices of the top-k most similar chunks
    top_indices = np.argsort(similarities)[-k:][::-1]

    # Return the top-k most relevant text chunks
    return [text_chunks[i] for i in top_indices]

def generate_response(system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generates a response from the AI model based on the system prompt and user message.

    Args:
    system_prompt (str): The system prompt to guide the AI's behavior.
    user_message (str): The user's message or query.
    model (str): The model to be used for generating the response. Default is "meta-llama/Llama-2-7B-chat-hf".

    Returns:
    dict: The response from the AI model.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response

# 完整调用流程
def semantic_chunking_rag_pipeline(pdf_path, query):
    # 1. 提取PDF文本
    extracted_text = extract_text_from_pdf(pdf_path)

    # 2. 按句子分割
    sentences = extracted_text.split(". ")

    # 3. 生成句子嵌入
    embeddings = [get_embedding(sentence) for sentence in sentences]

    # 4. 计算句子间相似度
    similarities = [cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

    # 5. 计算断点（使用百分位数方法）
    breakpoints = compute_breakpoints(similarities, method="percentile", threshold=90)

    # 6. 分割成语义块
    text_chunks = split_into_chunks(sentences, breakpoints)

    # 7. 创建块嵌入
    chunk_embeddings = create_embeddings(text_chunks)

    # 8. 语义搜索
    top_chunks = semantic_search(query, text_chunks, chunk_embeddings, k=2)

    # 9. 生成回答
    system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"
    user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
    user_prompt = f"{user_prompt}\nQuestion: {query}"

    ai_response = generate_response(system_prompt, user_prompt)
    return ai_response.choices[0].message.content
```

## 代码讲解
- **句子分割**：按句号分割文本成句子
- **嵌入生成**：为每个句子生成向量表示
- **相似度计算**：计算相邻句子的余弦相似度
- **断点检测**：使用百分位数方法找到语义断点
- **语义分块**：根据断点将句子组合成语义块
- **检索生成**：基于语义块进行检索和答案生成

## 主要特点
- 基于语义相似度的智能分块
- 支持多种断点检测方法（百分位数、标准差、四分位距）
- 保持语义连贯性
- 比固定长度分块更精准

## 使用场景
- 长文档处理
- 需要保持语义完整性的场景
- 复杂问答系统
- 学术论文、技术文档等结构化文本

