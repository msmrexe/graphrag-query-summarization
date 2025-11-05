import logging
import re
import networkx as nx
import igraph as ig
import leidenalg
from typing import List, Dict, Any, Tuple
from .llm_handler import LLMHandler
import pandas as pd

# PROMPT for entity and relationship extraction
ENTITY_EXTRACTION_PROMPT = """
Extract all entities and their relationships from the text.
Respond in the following format, and only this format:

Entities:
- <Entity 1>
- <Entity 2>
...

Relationships:
- <Entity 1> -> <Relationship> -> <Entity 2>
- <Entity 3> -> <Relationship> -> <Entity 4>
...
"""

# PROMPT for summarizing a community
COMMUNITY_SUMMARY_PROMPT = """
Based on the following entities and relationships from a text community, write a concise, one-paragraph summary that captures the main topic of this community.

Data:
{community_data}

Summary:
"""

def extract_entities_relationships(
    df_text_units: pd.DataFrame, llm: LLMHandler
) -> List[str]:
    """
    Uses the LLM to extract entities and relationships from each text unit.
    """
    logging.info(f"Extracting entities from {len(df_text_units)} text units...")
    elements = []
    for index, row in df_text_units.iterrows():
        chunk = row['text_unit']
        logging.info(f"Processing chunk {index + 1}/{len(df_text_units)}...")
        try:
            response = llm.get_response(ENTITY_EXTRACTION_PROMPT, chunk)
            elements.append(response)
            logging.debug(f"Chunk {index} elements:\n{response}")
        except Exception as e:
            logging.error(f"Failed on chunk {index}: {e}")
            elements.append("Error: Failed to extract elements.")
    logging.info("Finished extracting entities and relationships.")
    return elements

def build_knowledge_graph(element_summaries: List[str]) -> nx.Graph:
    """
    Builds a NetworkX graph from the list of element summaries.
    This function robustly parses the LLM's text output.
    """
    logging.info("Building knowledge graph...")
    G = nx.Graph()
    
    # Regex to find (Source Entity) -> (Relationship) -> (Target Entity)
    # Allows for entities and relationships to contain spaces
    rel_pattern = re.compile(r"^\s*-\s*(.+?)\s*->\s*(.+?)\s*->\s*(.+?)\s*$", re.MULTILINE)
    
    for i, summary in enumerate(element_summaries):
        if "Relationships:" not in summary:
            logging.warning(f"No 'Relationships:' block found in summary {i}, skipping.")
            continue

        # Find all relationship matches
        for match in rel_pattern.finditer(summary):
            try:
                source = match.group(1).strip()
                relation = match.group(2).strip()
                target = match.group(3).strip()

                if not source or not relation or not target:
                    logging.warning(f"Skipping malformed relationship in summary {i}")
                    continue

                # Add nodes and edge to the graph
                G.add_node(source)
                G.add_node(target)
                G.add_edge(source, target, label=relation)
                
            except Exception as e:
                logging.error(f"Error parsing line in summary {i}: {e}")

    logging.info(f"Knowledge graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def detect_communities(graph: nx.Graph) -> List[List[str]]:
    """
    Detects communities in the graph using the Leiden algorithm.
    Handles the conversion from NetworkX to iGraph and maps node names.
    """
    logging.info("Detecting communities using Leiden algorithm...")
    
    # iGraph works on integer IDs. We must map our string node names.
    # We process each disconnected component separately for better results.
    
    all_communities = []
    
    for component in nx.connected_components(graph):
        subgraph_nx = graph.subgraph(component)
        
        # 1. Create iGraph object from the NetworkX subgraph
        subgraph_ig = ig.Graph.from_networkx(subgraph_nx)
        
        # 2. Create the mapping from iGraph int ID back to NetworkX string name
        # The 'name' vertex attribute is automatically created by from_networkx
        int_to_name_map = {v.index: v["name"] for v in subgraph_ig.vs}

        try:
            # 3. Run Leiden algorithm
            partition = leidenalg.find_partition(
                subgraph_ig, 
                leidenalg.ModularityVertexPartition
            )
            
            logging.info(f"Found {len(partition)} communities in component.")
            
            # 4. Map integer-based communities back to node names
            for int_community in partition:
                named_community = [int_to_name_map[node_id] for node_id in int_community]
                all_communities.append(named_community)
                
        except Exception as e:
            # This handles errors seen in the original notebook, e.g., if a
            # component is trivial (single node) and causes issues.
            logging.warning(f"Could not process component: {e}. Using nodes as single community.")
            if len(component) > 0:
                 all_communities.append(list(component))

    logging.info(f"Total communities found: {len(all_communities)}")
    return all_communities

def summarize_communities(
    communities: List[List[str]], graph: nx.Graph, llm: LLMHandler
) -> List[Dict[str, Any]]:
    """
    Generates a textual summary for each community.
    """
    logging.info(f"Summarizing {len(communities)} communities...")
    community_summaries = []
    
    for i, community_nodes in enumerate(communities):
        logging.info(f"Summarizing community {i + 1}/{len(communities)}...")
        
        # Build a text representation of the community
        community_data = "Entities:\n"
        for node in community_nodes:
            community_data += f"- {node}\n"
        
        community_data += "\nRelationships:\n"
        # Get the subgraph for this community to find internal edges
        subgraph = graph.subgraph(community_nodes)
        for u, v, data in subgraph.edges(data=True):
            community_data += f"- {u} -> {data['label']} -> {v}\n"
            
        try:
            # Generate summary
            summary = llm.get_response(
                COMMUNITY_SUMMARY_PROMPT.format(community_data=community_data),
                "" # No additional content needed, prompt is self-contained
            )
            
            community_summaries.append({
                "community_id": i,
                "nodes": community_nodes,
                "summary": summary
            })
            logging.debug(f"Community {i} summary: {summary}")
        except Exception as e:
            logging.error(f"Failed to summarize community {i}: {e}")
            community_summaries.append({
                "community_id": i,
                "nodes": community_nodes,
                "summary": "Error: Failed to generate summary."
            })
            
    logging.info("Finished summarizing communities.")
    return community_summaries
