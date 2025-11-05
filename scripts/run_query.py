import argparse
import logging
import os
import json
from src.utils import setup_logging
from src.llm_handler import LLMHandler
from src.query_handler import generate_global_answer

def main(args):
    """
    Main orchestration script for the querying pipeline.
    """
    setup_logging()
    logging.info(f"Starting GraphRAG Query Pipeline for query: '{args.query}'")

    try:
        # 1. Load Community Summaries
        if not os.path.exists(args.summaries_path):
            logging.critical(f"Summaries file not found: {args.summaries_path}")
            logging.critical("Please run the indexing pipeline first: `python scripts/run_indexing.py`")
            return

        with open(args.summaries_path, 'r', encoding='utf-8') as f:
            community_summaries = json.load(f)
        
        if not community_summaries:
            logging.error("Community summaries file is empty.")
            return
            
        logging.info(f"Loaded {len(community_summaries)} community summaries.")

        # 2. Initialize LLM
        llm = LLMHandler(model_name=args.model_name)

        # 3. Generate Global Answer
        final_answer = generate_global_answer(args.query, community_summaries, llm)
        
        # 4. Print Answer
        print("\n" + "="*80)
        print(f"Query: {args.query}")
        print("\nFinal Global Answer:\n")
        print(final_answer)
        print("="*80)
        
        logging.info("GraphRAG Query Pipeline COMPLETED successfully.")

    except Exception as e:
        logging.critical(f"Query pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG Querying Pipeline")
    
    parser.add_argument(
        "--query", 
        type=str, 
        required=True,
        help="The global query to ask the document corpus."
    )
    parser.add_argument(
        "--summaries_path", 
        type=str, 
        default="output/community_summaries.json",
        help="Path to the pre-computed community summaries JSON file."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Hugging Face model to use for all LLM tasks."
    )
    
    args = parser.parse_args()
    main(args)
