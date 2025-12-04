
import json
import time
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import re
from openai import OpenAI

# Configuration
MODEL = "gpt-4o-mini"
# Client will be initialized in main() after checking for API key
client = None

# Sample texts (12 human + we'll generate 12 AI)
HUMAN_TEXTS = [
    "I've been thinking about this problem for weeks now. The solution isn't obvious, but I think I'm getting closer.",
    "The weather today is absolutely beautiful. I spent the morning in the garden, and it reminded me why I love spring so much.",
    "I tried that new restaurant downtown last night. The food was okay, but honestly, I've had better.",
    "My cat keeps knocking things off my desk. I don't know why she does this, but it's become a daily occurrence.",
    "I'm reading this book about ancient civilizations, and it's fascinating how they built such complex structures.",
    "The traffic was terrible this morning. I was stuck for almost an hour, which made me late for my meeting.",
    "I've been learning to play the guitar, and it's harder than I thought. My fingers hurt, and I can't seem to get the chords right.",
    "I saw the most amazing sunset yesterday. The sky was painted in shades of orange, pink, and purple.",
    "I'm trying to decide what to cook for dinner tonight. I have some chicken in the fridge, but I'm not sure what to make.",
    "I had the weirdest dream last night. I was flying over a city made of glass, and there were talking animals everywhere.",
    "I'm working on a project for work, and it's taking longer than expected. There are so many details to consider.",
    "I went for a walk in the park today, and I saw so many people out enjoying the nice weather.",
]

AI_PROMPTS = [
    "Write a short paragraph explaining how machine learning algorithms work.",
    "Describe the benefits of renewable energy sources in a few sentences.",
    "Explain the concept of cloud computing in simple terms.",
    "Write a brief paragraph about the importance of cybersecurity.",
    "Describe how artificial intelligence is transforming healthcare.",
    "Explain the difference between supervised and unsupervised learning.",
    "Write a short paragraph about the role of data science in business.",
    "Describe the advantages of using Python for data analysis.",
    "Explain how neural networks process information.",
    "Write a brief paragraph about the future of autonomous vehicles.",
    "Describe the impact of social media on modern communication.",
    "Explain the concept of blockchain technology in simple terms.",
]

# Pre-generated AI texts as fallback
FALLBACK_AI_TEXTS = [
    "Machine learning algorithms are computational methods that enable systems to learn and improve from experience without being explicitly programmed. They analyze patterns in data to make predictions or decisions.",
    "Renewable energy sources like solar and wind power offer significant environmental benefits. They produce clean electricity without emitting greenhouse gases, reducing our carbon footprint and helping combat climate change.",
    "Cloud computing allows users to access computing resources over the internet instead of maintaining physical servers. It provides scalability, cost-effectiveness, and enables remote access to data and applications from anywhere.",
    "Cybersecurity is crucial in protecting digital systems from threats. It involves implementing measures to prevent unauthorized access, data breaches, and cyber attacks that could compromise sensitive information.",
    "Artificial intelligence is revolutionizing healthcare by enabling faster diagnosis, personalized treatment plans, and drug discovery. AI systems can analyze medical images and patient data more efficiently than traditional methods.",
    "Supervised learning uses labeled training data to teach algorithms, while unsupervised learning finds patterns in unlabeled data. Both approaches have different applications in data analysis and machine learning.",
    "Data science plays a vital role in business decision-making by extracting insights from large datasets. It helps companies understand customer behavior, optimize operations, and identify new opportunities for growth.",
    "Python offers numerous advantages for data analysis, including extensive libraries like pandas and NumPy. Its simple syntax and powerful tools make it ideal for processing, analyzing, and visualizing complex datasets.",
    "Neural networks process information through interconnected nodes that mimic biological neurons. They learn by adjusting connection weights based on training data, enabling pattern recognition and complex problem-solving.",
    "Autonomous vehicles represent the future of transportation, using sensors and AI to navigate without human intervention. They promise to reduce accidents, improve traffic flow, and provide mobility for those unable to drive.",
    "Social media has transformed modern communication by enabling instant global connectivity. It allows people to share information, connect with others, and participate in discussions regardless of geographical boundaries.",
    "Blockchain technology is a decentralized digital ledger that records transactions across multiple computers. It ensures transparency, security, and immutability, making it valuable for applications like cryptocurrency and supply chain management.",
]


def call_openai_api(prompt, model="gpt-4o-mini", max_tokens=200, max_retries=3):
    """Call OpenAI API."""
    global client
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   API error, retrying... ({attempt+1}/{max_retries})")
                time.sleep(2)
            else:
                print(f"   Error after {max_retries} attempts: {e}")
                return None
    
    return None


def standard_detection_prompt(text):
    """Standard prompt: Is this AI or human?"""
    return f"""Read this text and determine if it was written by a human or generated by AI.

Text: "{text}"

Respond with:
Answer: [Human or AI]
Confidence: [0-100]"""


def unusualness_detection_prompt(text):
    """Unusualness-based anomaly prompt."""
    return f"""Analyze this text and rate how unusual it is compared to typical AI-generated output (0-100, where 0 is very typical of AI and 100 is very unusual/human-like).

Text: "{text}"

Respond with:
Unusualness: [0-100]
Assessment: [Human or AI]
Explanation: [brief reason]"""


def parse_response(response_text, prompt_type):
    """Extract prediction and score from LLM response."""
    if not response_text:
        return "Unknown", 50.0
    
    text_lower = response_text.lower()
    
    # Extract prediction - only check the line containing "Answer:" or "Assessment:"
    prediction = "Unknown"
    lines = response_text.splitlines()
    answer_line = ""
    
    for line in lines:
        if "answer:" in line.lower() or "assessment:" in line.lower():
            answer_line = line.lower()
            break
    
    if answer_line:
        if "human" in answer_line:
            prediction = "Human"
        elif "ai" in answer_line:
            prediction = "AI"
    
    # Extract score
    score = None
    for keyword in ["confidence:", "unusualness:", "score:"]:
        if keyword in text_lower:
            idx = text_lower.find(keyword)
            remaining = text_lower[idx + len(keyword):]
            numbers = re.findall(r'\d+', remaining)
            if numbers:
                score = min(100, max(0, float(numbers[0])))
                break
    
    if score is None:
        # Try to find any number 0-100
        numbers = re.findall(r'\b([0-9]{1,2}|100)\b', response_text)
        if numbers:
            score = min(100, max(0, float(numbers[0])))
    
    if score is None:
        score = 50.0
    
    return prediction, score


def run_study():
    """Main function."""
    print("=" * 60)
    print("Feasibility Study: Anomaly-Based Detection")
    print("=" * 60)
    
    # Step 1: Generate AI texts
    print("\n[1/4] Generating AI texts...")
    ai_texts = []
    
    for i, prompt in enumerate(AI_PROMPTS):
        print(f"   Generating {i+1}/12...")
        ai_text = call_openai_api(prompt, model=MODEL, max_tokens=150)
        
        # If API failed, use fallback
        if ai_text is None:
            print(f"   Using fallback text {i+1}...")
            ai_text = FALLBACK_AI_TEXTS[i]
        
        ai_texts.append(ai_text)
        time.sleep(1)  # Rate limiting
    
    # Step 2: Run detections
    print("\n[2/4] Running standard prompt detection...")
    standard_results = []
    all_texts = [(t, "Human") for t in HUMAN_TEXTS] + [(t, "AI") for t in ai_texts]
    
    for i, (text, label) in enumerate(all_texts):
        print(f"   Processing {i+1}/24 (standard)...")
        prompt = standard_detection_prompt(text)
        response = call_openai_api(prompt, model=MODEL, max_tokens=100)
        
        # If API failed, use default
        if response is None:
            response = "Answer: Unknown\nConfidence: 50"
        
        pred, score = parse_response(response, "standard")
        standard_results.append({
            "text": text[:50] + "...",  # Truncate for display
            "true_label": label,
            "prediction": pred,
            "score": score,
            "raw_response": response[:100] + "..." if len(response) > 100 else response
        })
        time.sleep(1)
    
    print("\n[3/4] Running unusualness prompt detection...")
    unusualness_results = []
    for i, (text, label) in enumerate(all_texts):
        print(f"   Processing {i+1}/24 (unusualness)...")
        prompt = unusualness_detection_prompt(text)
        response = call_openai_api(prompt, model=MODEL, max_tokens=150)
        
        # If API failed, use default
        if response is None:
            response = "Unusualness: 50\nAssessment: Unknown\nExplanation: API unavailable"
        
        pred, score = parse_response(response, "unusualness")
        unusualness_results.append({
            "text": text[:50] + "...",
            "true_label": label,
            "prediction": pred,
            "score": score,
            "raw_response": response[:100] + "..." if len(response) > 100 else response
        })
        time.sleep(1)
    
    # Step 3: Compute metrics
    print("\n[4/4] Computing metrics and visualizations...")
    
    def calc_metrics(results):
        y_true = [1 if r["true_label"] == "Human" else 0 for r in results]
        y_pred = [1 if r["prediction"] == "Human" else 0 for r in results]
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return accuracy, cm, [r["score"] for r in results]
    
    acc_std, cm_std, scores_std = calc_metrics(standard_results)
    acc_un, cm_un, scores_un = calc_metrics(unusualness_results)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nStandard Prompt:")
    print(f"  Accuracy: {acc_std:.2%}")
    print(f"  Confusion Matrix: TN={cm_std[0][0]}, FP={cm_std[0][1]}, FN={cm_std[1][0]}, TP={cm_std[1][1]}")
    print(f"  Avg Score: {np.mean(scores_std):.1f}")
    
    print(f"\nUnusualness Prompt:")
    print(f"  Accuracy: {acc_un:.2%}")
    print(f"  Confusion Matrix: TN={cm_un[0][0]}, FP={cm_un[0][1]}, FN={cm_un[1][0]}, TP={cm_un[1][1]}")
    print(f"  Avg Score: {np.mean(scores_un):.1f}")
    
    print(f"\nImprovement: {acc_un - acc_std:.2%}")
    print("=" * 60)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histograms
    human_std = [r["score"] for r in standard_results if r["true_label"] == "Human"]
    ai_std = [r["score"] for r in standard_results if r["true_label"] == "AI"]
    axes[0, 0].hist(human_std, alpha=0.5, label="Human", bins=10, color="blue")
    axes[0, 0].hist(ai_std, alpha=0.5, label="AI", bins=10, color="red")
    axes[0, 0].set_xlabel("Confidence Score")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Standard Prompt: Score Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    human_un = [r["score"] for r in unusualness_results if r["true_label"] == "Human"]
    ai_un = [r["score"] for r in unusualness_results if r["true_label"] == "AI"]
    axes[0, 1].hist(human_un, alpha=0.5, label="Human", bins=10, color="blue")
    axes[0, 1].hist(ai_un, alpha=0.5, label="AI", bins=10, color="red")
    axes[0, 1].set_xlabel("Unusualness Score")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Unusualness Prompt: Score Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion matrices
    import seaborn as sns
    sns.heatmap(cm_std, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0],
                xticklabels=["AI", "Human"], yticklabels=["AI", "Human"])
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("True")
    axes[1, 0].set_title("Standard Prompt: Confusion Matrix")
    
    sns.heatmap(cm_un, annot=True, fmt="d", cmap="Blues", ax=axes[1, 1],
                xticklabels=["AI", "Human"], yticklabels=["AI", "Human"])
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("True")
    axes[1, 1].set_title("Unusualness Prompt: Confusion Matrix")
    
    plt.tight_layout()
    plt.savefig("feasibility_results.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved: feasibility_results.png")
    
    # Save JSON
    results = {
        "standard_accuracy": float(acc_std),
        "unusualness_accuracy": float(acc_un),
        "standard_confusion_matrix": cm_std.tolist(),
        "unusualness_confusion_matrix": cm_un.tolist(),
        "standard_results": standard_results,
        "unusualness_results": unusualness_results
    }
    
    with open("feasibility_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✓ Saved: feasibility_results.json")
    
    return results


if __name__ == "__main__":
    # Check for OpenAI API key in environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not found.")
        print("\nPlease set your OpenAI API key:")
        print("  Windows PowerShell: $env:OPENAI_API_KEY=\"your_key_here\"")
        print("  Windows CMD: setx OPENAI_API_KEY \"your_key_here\"")
        print("  Linux/Mac: export OPENAI_API_KEY=\"your_key_here\"")
        print("\nNote: After using setx, you may need to restart your terminal.")
        sys.exit(1)
    
    # Initialize client with the key from environment variable
    client = OpenAI(api_key=api_key)
    
    try:
        results = run_study()
        print("\n✓ Study completed!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
