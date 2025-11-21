# LLM Steering with Activation Engineering

This project implements **Activation Steering** (also known as Activation Addition or Representation Engineering) to control the behavior of a Large Language Model (LLM) at inference time without fine-tuning.

## Topic
The core concept is **Mechanistic Interpretability** and **Steering**. By identifying a direction in the model's activation space that corresponds to a specific concept (in this case, "truthfulness" vs. "hallucination"), we can intervene during the forward pass to shift the model's behavior.

## Implementation Details
The implementation follows the "Mean Difference" method:
1.  **Data Collection**: We define a set of positive prompts (factual statements) and negative prompts (false/hallucinated statements).
2.  **Activation Extraction**: We run these prompts through the model and record the internal activations (hidden states) at each layer.
3.  **Vector Calculation**: We compute the mean activation vector for positive and negative sets and find the difference: $\vec{v}_{steering} = \mu_{pos} - \mu_{neg}$.
4.  **Intervention**: During inference, we inject this steering vector into a specific layer of the model with a tunable coefficient: $x' = x + \alpha \cdot \vec{v}_{steering}$.

## Papers & References
This implementation is based on concepts from:
*   **"Activation Addition: Steering Language Models Without Optimization"** (Turner et al., 2023)
*   **"Representation Engineering: A Top-Down Approach to AI Transparency"** (Zou et al., 2023)

## Models Used
*   **Model**: `google/gemma-3-1b-it` (Gemma 3, 1B parameters, Instruction Tuned)
*   **Library**: Hugging Face `transformers` and `torch`.

## Usage
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the main script:
    ```bash
    python main.py
    ```
