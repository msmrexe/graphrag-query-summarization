import argparse
import logging
import os
import json
import networkx as nx
from src.utils import setup_logging
from src.data_processing import download_pdf, load_and_split_docs
from src.llm_handler import LLMHandler
from src.graph_pipeline import (
    extract_entities_relationships,
    build_knowledge_graph,
    detect_communities,
    summarize_communities,
)

def main(args):
    """
    Main orchestration script for the indexing pipeline.
    """
    setup_logging()
    logging.info("Starting GraphRAG Indexing Pipeline...")

    try:
        # 1. Download PDF
        if args.pdf_url:
            download_pdf(args.pdf_url, args.pdf_path)
        
        if not os.path.exists(args.pdf_path):
            logging.critical(f"PDF file not found at: {args.pdf_path}. Exiting.")
            return

        # 2. Load and Split
        df_text_units = load_and_split_docs(args.pdf_path)
        if df_text_units is None or df_text_units.empty:
            logging.critical("No text units extracted. Exiting.")
            return

        # 3. Initialize LLM
        llm = LLMHandler(model_name=args.model_name)

        # 4. Extract Entities
        elements = extract_entities_relationships(df_text_units, llm)

        # 5. Build Graph
        graph = build_knowledge_graph(elements)

        # 6. Detect Communities
        communities = detect_communities(graph)

        # 7. Summarize Communities
        community_summaries = summarize_communities(communities, graph, llm)

        # 8. Save Artifacts
        os.makedirs(args.output_dir, exist_ok=True)
        
        graph_path = os.path.join(args.output_dir, "knowledge_graph.gml")
        summaries_path = os.path.join(args.output_dir, "community_summaries.json")
        
        nx.write_gml(graph, graph_path)
        logging.info(f"Knowledge graph saved to {graph_path}")
        
        with open(summaries_path, 'w', encoding='utf-8') as f:
            json.dump(community_summaries, f, indent=4)
        logging.info(f"Community summaries saved to {summaries_path}")
        
        logging.info("GraphRAG Indexing Pipeline COMPLETED successfully.")

    except Exception as e:
        logging.critical(f"Indexing pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG Indexing Pipeline")
    
    parser.add_argument(
        "--pdf_url", 
        type=str, 
        default="https://arxiv.org/pdf/2404.16130v1",
        help="URL of the PDF to download."
    )
    parser.add_argument(
        "--pdf_path", 
        type=str, 
        default="data/2404.16130v1.pdf",
        help="Local path to the PDF file."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output",
        help="Directory to save output artifacts (graph, summaries)."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Hugging Face model to use for all LLM tasks."
    )
    
    args = parser.parse_args()
    main(args)
