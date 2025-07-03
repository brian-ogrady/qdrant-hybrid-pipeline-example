import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import datetime


EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator for an e-commerce search engine. Your task is to assess the relevance of a retrieved document for a given user query. The document represents a product listing.

Please rate the relevance on a scale from 1 to 5 based on the following criteria:
1: **Irrelevant**: The document is completely unrelated to the query.
2: **Slightly Relevant**: The document is in the correct category but is not a good match for the query specifics.
3: **Relevant**: The document is a reasonable match for the query.
4. **Highly Relevant**: The document is a very good and direct match for the query.
5: **Perfect Match**: The document is the exact product the user was looking for.

You must provide your response in a JSON format with two keys: "score" (an integer from 1 to 5) and "reasoning" (a brief explanation).

**User Query**: "{query}"

**Retrieved Document**: "{document}"

Your JSON response:
"""

def call_openai_api(prompt: str, client: OpenAI) -> dict:
    """
    Calls the OpenAI API to get a relevance grade.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert search relevance evaluator. Respond in valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        grade_str = response.choices[0].message.content
        return json.loads(grade_str)

    except Exception as e:
        print(f"An error occurred during the OpenAI API call: {e}")
        return {"score": 0, "reasoning": "API call failed."}


def calculate_ndcg_for_query(scores: list) -> float:
    """Calculates the nDCG score for a single list of relevance scores."""
    dcg = sum([score / np.log2(i + 2) for i, score in enumerate(scores)])
    ideal_scores = sorted(scores, reverse=True)
    idcg = sum([score / np.log2(i + 2) for i, score in enumerate(ideal_scores)])
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def create_evaluation_report(graded_results: dict) -> dict:
    """
    Analyzes graded results and creates a comprehensive evaluation report.
    """
    detailed_grades = {}
    all_ndcg_scores = []

    for query, scores in graded_results.items():
        ndcg = calculate_ndcg_for_query(scores)
        all_ndcg_scores.append(ndcg)
        detailed_grades[query] = {
            "llm_scores": scores,
            "ndcg": round(ndcg, 4)
        }
    
    average_ndcg = np.mean(all_ndcg_scores)

    report = {
        "evaluation_summary": {
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "total_queries_evaluated": len(graded_results),
            "average_ndcg_score": round(average_ndcg, 4)
        },
        "detailed_grades": detailed_grades
    }
    
    print("\n--- Final Search Quality Grade ---")
    print(f"  Average nDCG Score: {average_ndcg:.4f}")
    print("----------------------------------")
    
    return report


def evaluate_search_results(results_path: str, text_column: str):
    """
    Loads search results, grades them using an LLM, and saves a final report.
    """
    try:
        openai_client = OpenAI()
    except Exception as e:
        print(f"Failed to initialize OpenAI client. Is OPENAI_API_KEY set? Error: {e}")
        return

    try:
        with open(results_path, 'r') as f:
            search_results = json.load(f)
        print(f"Loaded {len(search_results)} queries from {results_path}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading results file: {e}")
        return

    graded_results = {}
    for query, results in tqdm(search_results.items(), desc="LLM Judging"):
        scores = []
        for res in results:
            document_text = res.get("payload", {}).get(text_column, "")
            prompt = EVALUATION_PROMPT_TEMPLATE.format(query=query, document=document_text)
            llm_grade = call_openai_api(prompt, openai_client)
            scores.append(llm_grade.get("score", 0))
        graded_results[query] = scores

    final_report = create_evaluation_report(graded_results)

    report_output_path = "evaluation_report.json"
    with open(report_output_path, 'w') as f:
        json.dump(final_report, f, indent=4)
    print(f"\nEvaluation complete. Report saved to {report_output_path}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate search results using an LLM as a judge.")
    parser.add_argument(
        "--results-path",
        type=str,
        required=True,
        help="Path to the search_results.json file."
    )
    parser.add_argument(
        "--text-column",
        type=str,
        required=True,
        help="The name of the column that contains the document text within the payload."
    )
    args = parser.parse_args()

    evaluate_search_results(args.results_path, args.text_column)


if __name__ == "__main__":
    main()
