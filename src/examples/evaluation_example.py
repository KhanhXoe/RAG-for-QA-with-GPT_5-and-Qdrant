from typing import Dict, Any
from src.evaluation import LLMEvaluator, ManualEvaluator, ScriptEvaluator
from src.reformulator import LLMQueryReformulator

# Define custom evaluation functions
def evaluate_reformulator(input_data: Dict[str, Any], 
                         output_data: Dict[str, Any]) -> tuple[float, str]:
    query = input_data["args"][0]  # Original query
    result = output_data["result"]  # ReformulatedQuery object
    
    # Example evaluation logic
    score = 0.0
    feedback = []
    
    # Check if refined query is longer than original
    if len(result.refined_text) > len(query):
        score += 0.3
        feedback.append("Query expanded appropriately")
    
    # Check if we have a reasonable number of keywords
    if 2 <= len(result.keywords) <= 5:
        score += 0.3
        feedback.append("Good number of keywords")
    
    # Check if keywords appear in refined query
    keywords_in_query = sum(1 for k in result.keywords if k.lower() in result.refined_text.lower())
    score += 0.4 * (keywords_in_query / len(result.keywords))
    feedback.append(f"{keywords_in_query}/{len(result.keywords)} keywords in refined query")
    
    return score, ". ".join(feedback)

async def main():
    # Create evaluators
    llm_evaluator = LLMEvaluator()
    manual_evaluator = ManualEvaluator()
    script_evaluator = ScriptEvaluator({
        "reformulator": evaluate_reformulator
    })
    
    # Test with different evaluators
    queries = [
        "what is machine learning?",
        "tell me about neural networks"
    ]
    
    for query in queries:
        print(f"\nProcessing query: {query}")
        
        # Test LLM evaluator
        reformulator = LLMQueryReformulator(evaluator=llm_evaluator)
        result, log = await reformulator.execute(query)
        print(f"LLM Evaluation: {log.evaluation}")
        
        # Test script evaluator
        reformulator = LLMQueryReformulator(evaluator=script_evaluator)
        result, log = await reformulator.execute(query)
        print(f"Script Evaluation: {log.evaluation}")

if __name__ == "__main__":
    asyncio.run(main()) 