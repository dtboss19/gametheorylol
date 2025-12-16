"""
Game Tree Visualization for League of Legends Draft Scenarios.

Creates a visual game tree showing:
- Decision nodes for each pick
- Terminal nodes with payoffs
- SPNE path (highlighted in green)
- Actual picks path (highlighted in blue)
- Available champions at each decision point

Usage:
    python visualize_game_tree.py [--max-candidates N] [--format png|pdf|svg]
    
    --max-candidates: Maximum champions to show at each decision level (default: 8)
                      Lower values create simpler trees, higher values show more options
    --format: Output format - png (default), pdf, or svg
    
Requirements:
    Install one of:
    - pip install graphviz  (recommended, requires Graphviz system package)
    - pip install matplotlib networkx  (alternative)
    
Example:
    python visualize_game_tree.py --max-candidates 10 --format pdf
    
Note: The visualization will calculate optimal picks automatically, but for best results,
      run the full analysis in test_game3_nash.py first to populate the cache.
"""

import sys
import math
import re
from typing import List, Dict, Tuple, Optional, Set
from copy import deepcopy

# Import existing modules
from nash_equilibrium import SubgamePerfectNashEquilibrium
from composition_scorer import CompositionScorer
from test_worlds2025_scenarios import EXTENDED_CHAMPION_POOL, ALL_EXTENDED_CHAMPIONS, create_draft_from_scenario
from get_worlds_data import load_champion_name_map, normalize_champion_name, TEAM_1, TEAM_2, ROLE_ORDER, load_player_comfort_data

# Try to import visualization libraries
HAS_GRAPHVIZ = False
HAS_MATPLOTLIB = False

try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import networkx as nx
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class GameTreeVisualizer:
    """Visualizes game trees for draft scenarios."""
    
    def __init__(self, spne: SubgamePerfectNashEquilibrium, 
                 blue_players: List[str], red_players: List[str],
                 alpha: float = 0.02):
        self.spne = spne
        self.blue_players = blue_players
        self.red_players = red_players
        self.alpha = alpha
        self.node_counter = 0
        self.nodes = {}  # node_id -> (label, type, data)
        self.edges = []  # (from_id, to_id, label, style)
        self.spne_path = []  # List of node IDs in SPNE path
        self.actual_path = []  # List of node IDs in actual picks path
        
    def calculate_payoff(self, blue_team: List[str], red_team: List[str]) -> float:
        """Calculate payoff for a complete team composition."""
        return self.spne.calculate_payoff(
            blue_team, red_team,
            self.blue_players, self.red_players, 'na1', use_fast_mode=False
        )
    
    def calculate_win_rate(self, payoff: float) -> Tuple[float, float]:
        """Calculate win rates from payoff."""
        blue_win = 1 / (1 + math.exp(-(payoff + self.alpha)))
        red_win = 1 - blue_win
        return blue_win, red_win
    
    def build_tree(self, 
                   draft,
                   available_champions: List[str],
                   draft_order: List[str],
                   optimal_blue: List[str],
                   optimal_red: List[str],
                   actual_blue: List[str],
                   actual_red: List[str],
                   max_candidates_per_level: int = 10,
                   blue_is_team1: bool = False):
        """
        Build the game tree structure.
        
        Args:
            draft: DraftState object
            available_champions: List of available champions
            draft_order: List like ['blue', 'blue', 'red']
            optimal_blue: Optimal Blue team picks (for SPNE path)
            optimal_red: Optimal Red team picks (for SPNE path)
            actual_blue: Actual Blue team picks
            actual_red: Actual Red team picks
            max_candidates_per_level: Maximum champions to show at each level
        """
        # Initialize teams from locked picks
        # blue_is_team1: True if Blue is team1, False if Blue is team2
        if blue_is_team1:
            blue_team = [draft.team1_picks.get(role) for role in ROLE_ORDER]
            red_team = [draft.team2_picks.get(role) for role in ROLE_ORDER]
        else:
            blue_team = [draft.team2_picks.get(role) for role in ROLE_ORDER]
            red_team = [draft.team1_picks.get(role) for role in ROLE_ORDER]
        
        # Create root node
        root_id = self._create_node("ROOT", "root", {
            "blue_team": blue_team.copy(),
            "red_team": red_team.copy(),
            "available": available_champions,
            "depth": 0
        })
        
        # Track which picks are being made
        # Determine pick roles based on draft order length and scenario
        pick_roles = []
        if len(draft_order) == 2:
            # Game 3: Blue MID, Red MID
            pick_roles = ['MID', 'MID']
        elif len(draft_order) == 3:
            # Game 1: Blue BOT, Blue SUPPORT, Red TOP
            pick_roles = ['BOT', 'SUPPORT', 'TOP']
        else:
            # Fallback: assume MID for all
            pick_roles = ['MID'] * len(draft_order)
        
        # Build tree recursively - initialize dict to track Red's best payoff for each Blue combo
        blue_combo_best_red_payoff = {}
        self._build_tree_recursive(
            root_id, blue_team, red_team, available_champions,
            draft_order, pick_roles, 0,
            optimal_blue, optimal_red, actual_blue, actual_red,
            max_candidates_per_level, draft.bans,
            parent_on_spne=True,  # Root is start of all paths
            parent_on_actual=True,
            blue_combo_best_red_payoff=blue_combo_best_red_payoff
        )
        
        return root_id
    
    def _create_node(self, label: str, node_type: str, data: Dict) -> str:
        """Create a node and return its ID."""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        self.nodes[node_id] = (label, node_type, data)
        return node_id
    
    def _build_tree_recursive(self,
                              parent_id: str,
                              blue_team: List[str],
                              red_team: List[str],
                              available: List[str],
                              draft_order: List[str],
                              pick_roles: List[str],
                              depth: int,
                              optimal_blue: List[str],
                              optimal_red: List[str],
                              actual_blue: List[str],
                              actual_red: List[str],
                              max_candidates: int,
                              bans: List[str],
                              parent_on_spne: bool = False,
                              parent_on_actual: bool = False,
                              blue_combo_best_red_payoff: Dict = None):
        """Recursively build the game tree."""
        if depth >= len(draft_order):
            # Terminal node - calculate payoff
            payoff = self.calculate_payoff(blue_team, red_team)
            blue_win, red_win = self.calculate_win_rate(payoff)
            
            # Check if this is SPNE or actual path
            is_spne = (blue_team == optimal_blue and red_team == optimal_red)
            is_actual = (blue_team == actual_blue and red_team == actual_red)
            
            # Check if this is Red's best response for this Blue combination
            # Create a key for this Blue combination (BOT + SUPPORT)
            role_idx_bot = ROLE_ORDER.index("BOT")
            role_idx_support = ROLE_ORDER.index("SUPPORT")
            blue_combo_key = (blue_team[role_idx_bot], blue_team[role_idx_support])
            
            is_reds_best = False
            if blue_combo_best_red_payoff is not None:
                # Check if this payoff matches the worst (minimum) payoff for this Blue combo
                if blue_combo_key in blue_combo_best_red_payoff:
                    best_payoff = blue_combo_best_red_payoff[blue_combo_key]
                    # Use small epsilon for float comparison
                    is_reds_best = abs(payoff - best_payoff) < 0.0001
            
            # Label terminal node - only mark as "Red's best" if it's actually Red's best response
            if is_reds_best:
                label = f"Payoff: {payoff:+.3f}\nBlue: {blue_win:.1%}\nRed: {red_win:.1%}\n(Red's best)"
            else:
                label = f"Payoff: {payoff:+.3f}\nBlue: {blue_win:.1%}\nRed: {red_win:.1%}"
            
            node_id = self._create_node(label, "terminal", {
                "blue_team": blue_team.copy(),
                "red_team": red_team.copy(),
                "payoff": payoff,
                "blue_win": blue_win,
                "red_win": red_win,
                "is_spne": is_spne,
                "is_actual": is_actual,
                "is_reds_best": is_reds_best,
                "depth": depth
            })
            
            # Determine edge style based on path
            # Use parent's path status to determine edge style
            edge_style = "terminal"
            if parent_on_spne and is_spne:
                edge_style = "spne"
                self.spne_path.append(node_id)
            elif parent_on_actual and is_actual:
                edge_style = "actual"
                self.actual_path.append(node_id)
            
            self.edges.append((parent_id, node_id, "", edge_style))
            
            return
        
        # Determine current player and role
        current_player = draft_order[depth]
        current_role = pick_roles[depth] if depth < len(pick_roles) else "MID"
        is_blue = (current_player == 'blue')
        
        # Get available champions for this role
        if current_role in EXTENDED_CHAMPION_POOL:
            # Use extended pool for this role
            candidates = [c for c in EXTENDED_CHAMPION_POOL[current_role] if c in available]
        else:
            # Fallback: use all available
            candidates = [c for c in available]
        
        # Remove already picked champions
        all_picked = [c for c in blue_team + red_team if c is not None]
        candidates = [c for c in candidates if c not in all_picked and c not in bans]
        
        # Special handling for Red's final pick: 
        # - For SPNE path: only show Red's BEST response (to prove it's optimal)
        # - For non-SPNE paths: show multiple Red responses (including best) to prove they're worse
        if not is_blue and depth == len(draft_order) - 1:
            # This is Red's final pick - find Red's best response (minimum payoff for Blue)
            best_red_champ = None
            best_red_payoff = float('inf')  # Red wants minimum (most negative)
            all_red_responses = []
            
            # Get the Blue combination key (BOT + SUPPORT)
            role_idx_bot = ROLE_ORDER.index("BOT")
            role_idx_support = ROLE_ORDER.index("SUPPORT")
            blue_combo_key = (blue_team[role_idx_bot], blue_team[role_idx_support])
            
            for champ in candidates:
                test_blue = blue_team.copy()
                test_red = red_team.copy()
                role_idx = ROLE_ORDER.index(current_role)
                test_red[role_idx] = champ
                
                payoff = self.calculate_payoff(test_blue, test_red)
                all_red_responses.append((champ, payoff))
                if payoff < best_red_payoff:
                    best_red_payoff = payoff
                    best_red_champ = champ
            
            # Store Red's best payoff for this Blue combination
            if blue_combo_best_red_payoff is None:
                blue_combo_best_red_payoff = {}
            blue_combo_best_red_payoff[blue_combo_key] = best_red_payoff
            
            # Check if this Blue combination is the SPNE
            is_spne_blue_combo = (blue_team[role_idx_bot] == optimal_blue[role_idx_bot] and 
                                  blue_team[role_idx_support] == optimal_blue[role_idx_support])
            
            if is_spne_blue_combo:
                # For SPNE path: only show Red's best response (proves it's optimal)
                if best_red_champ:
                    candidates = [best_red_champ]
                else:
                    candidates = candidates[:1] if candidates else []
            else:
                # For non-SPNE paths: show multiple Red responses to prove they're worse
                # Sort by payoff (ascending - worst for Blue first)
                all_red_responses.sort(key=lambda x: x[1])
                
                # Show top N worst responses (including Red's best response)
                # This proves that even Red's best response gives Blue a worse payoff than SPNE
                worst_responses = [c for c, _ in all_red_responses[:max_candidates]]
                
                # Ensure Red's best response is included
                if best_red_champ and best_red_champ not in worst_responses:
                    worst_responses.append(best_red_champ)
                
                candidates = worst_responses
        
        # Always limit candidates to top N based on payoff
        # This keeps the tree manageable while showing best options
        elif len(candidates) > max_candidates:
            # Score candidates and take top N
            scored_candidates = []
            for champ in candidates:
                test_blue = blue_team.copy()
                test_red = red_team.copy()
                if is_blue:
                    role_idx = ROLE_ORDER.index(current_role)
                    test_blue[role_idx] = champ
                else:
                    role_idx = ROLE_ORDER.index(current_role)
                    test_red[role_idx] = champ
                
                # Quick heuristic: calculate payoff (will be cached)
                payoff = self.calculate_payoff(test_blue, test_red)
                scored_candidates.append((champ, payoff))
            
            # Sort by payoff (descending for Blue, ascending for Red)
            if is_blue:
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
            else:
                scored_candidates.sort(key=lambda x: x[1])
            
            # Always take top N, but ensure SPNE/actual picks are included
            top_candidates = [c for c, _ in scored_candidates[:max_candidates]]
            
            # Ensure optimal/actual picks are always included if they exist
            role_idx = ROLE_ORDER.index(current_role)
            if is_blue:
                optimal_champ = optimal_blue[role_idx]
                actual_champ = actual_blue[role_idx]
            else:
                optimal_champ = optimal_red[role_idx]
                actual_champ = actual_red[role_idx]
            
            # Add optimal/actual if not already in top N
            final_candidates = top_candidates.copy()
            if optimal_champ and optimal_champ in candidates and optimal_champ not in final_candidates:
                final_candidates.append(optimal_champ)
            if actual_champ and actual_champ in candidates and actual_champ not in final_candidates:
                final_candidates.append(actual_champ)
            
            candidates = final_candidates[:max_candidates + 2]  # Allow a couple extra if needed
        else:
            # Even if we have fewer candidates, still prioritize by payoff
            scored_candidates = []
            for champ in candidates:
                test_blue = blue_team.copy()
                test_red = red_team.copy()
                if is_blue:
                    role_idx = ROLE_ORDER.index(current_role)
                    test_blue[role_idx] = champ
                else:
                    role_idx = ROLE_ORDER.index(current_role)
                    test_red[role_idx] = champ
                
                payoff = self.calculate_payoff(test_blue, test_red)
                scored_candidates.append((champ, payoff))
            
            # Sort by payoff
            if is_blue:
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
            else:
                scored_candidates.sort(key=lambda x: x[1])
            
            candidates = [c for c, _ in scored_candidates]
        
        # Create decision nodes for each candidate
        for champ in candidates:
            # Create new team states
            new_blue = blue_team.copy()
            new_red = red_team.copy()
            new_available = [c for c in available if c != champ]
            
            if is_blue:
                role_idx = ROLE_ORDER.index(current_role)
                new_blue[role_idx] = champ
                # If this is the first Blue pick (JUNGLE), fix it for subsequent picks
                if current_role == "JUNGLE":
                    # Store the fixed jungle for later
                    pass
            else:
                role_idx = ROLE_ORDER.index(current_role)
                new_red[role_idx] = champ
            
            # Check if this is on SPNE or actual path
            # Must be on path from parent AND this pick must match
            role_idx = ROLE_ORDER.index(current_role)
            
            if is_blue:
                # Blue's pick - must match optimal/actual AND parent was on path
                champ_matches_spne = (champ == optimal_blue[role_idx])
                champ_matches_actual = (champ == actual_blue[role_idx])
                is_on_spne = parent_on_spne and champ_matches_spne
                is_on_actual = parent_on_actual and champ_matches_actual
            else:  # Red pick
                # Red's pick - must match optimal/actual AND parent was on path
                champ_matches_spne = (champ == optimal_red[role_idx])
                champ_matches_actual = (champ == actual_red[role_idx])
                is_on_spne = parent_on_spne and champ_matches_spne
                is_on_actual = parent_on_actual and champ_matches_actual
            
            # Create node with player indicator (colored and bolded)
            # For Graphviz, use HTML-like labels with proper syntax
            if is_blue:
                # Graphviz HTML-like label format
                player_label_html = "<<b><font color='blue'>Blue</font></b><br/><font point-size='10'>{}</font><br/><font point-size='10'>{}</font>>"
                player_label_plain = "Blue"
            else:
                player_label_html = "<<b><font color='red'>Red</font></b><br/><font point-size='10'>{}</font><br/><font point-size='10'>{}</font>>"
                player_label_plain = "Red"
            label = player_label_html.format(current_role, champ) if is_blue else player_label_html.format(current_role, champ)
            label_plain = f"{player_label_plain}\n{current_role}\n{champ}"  # For matplotlib
            node_id = self._create_node(label, "decision", {
                "champion": champ,
                "role": current_role,
                "player": current_player,
                "blue_team": new_blue.copy(),
                "red_team": new_red.copy(),
                "available": new_available,
                "depth": depth + 1,
                "is_on_spne": is_on_spne,
                "is_on_actual": is_on_actual,
                "label_plain": label_plain,  # Store plain label for matplotlib
                "is_blue": is_blue,  # Store for color formatting
                "player_label_plain": player_label_plain  # Store for reference
            })
            
            # Add edge
            edge_style = "spne" if is_on_spne else ("actual" if is_on_actual else "normal")
            self.edges.append((parent_id, node_id, champ, edge_style))
            
            # Track path nodes (will be extended when we reach terminal)
            if is_on_spne:
                self.spne_path.append(node_id)
            if is_on_actual:
                self.actual_path.append(node_id)
            
            # Store parent reference for path building
            if is_on_spne or is_on_actual:
                # We'll add terminal nodes to path in the recursive call
                pass
            
            # Recurse - pass down path status and best Red payoff dict
            self._build_tree_recursive(
                node_id, new_blue, new_red, new_available,
                draft_order, pick_roles, depth + 1,
                optimal_blue, optimal_red, actual_blue, actual_red,
                max_candidates, bans,
                parent_on_spne=is_on_spne,
                parent_on_actual=is_on_actual,
                blue_combo_best_red_payoff=blue_combo_best_red_payoff
            )
    
    def visualize_graphviz(self, output_file: str = "game_tree", format: str = "png"):
        """Visualize using Graphviz."""
        if not HAS_GRAPHVIZ:
            raise ImportError("graphviz library not available. Install with: pip install graphviz")
        
        # Create graph
        dot = graphviz.Digraph(comment='Game Tree', format=format)
        dot.attr(rankdir='TB')  # Top to bottom
        dot.attr('node', shape='box', style='rounded')
        dot.attr('graph', splines='ortho', nodesep='0.5', ranksep='1.0')
        dot.attr('node', fontname='Arial')  # Use font that supports bold
        
        # Add nodes
        for node_id, (label, node_type, data) in self.nodes.items():
            # Determine color and style
            color = "white"
            style = "rounded"
            penwidth = "1"
            
            if node_type == "root":
                color = "lightgray"
                style = "filled,rounded"
            elif node_type == "terminal":
                color = "lightyellow"
                style = "filled,rounded"
                if data.get("is_spne"):
                    color = "lightgreen"
                    penwidth = "3"
                elif data.get("is_actual"):
                    color = "lightblue"
                    penwidth = "2"
                elif data.get("is_reds_best"):
                    # Highlight Red's best response terminal nodes
                    color = "lightcoral"
                    penwidth = "2"
            elif node_type == "decision":
                player = data.get("player", "")
                if player == "blue":
                    color = "lightcyan"
                else:
                    color = "lightpink"
                
                if data.get("is_on_spne"):
                    penwidth = "3"
                elif data.get("is_on_actual"):
                    penwidth = "2"
            
            # Use HTML-like labels for Graphviz (they need to be wrapped properly)
            # Check if label contains HTML-like syntax
            if label.startswith("<<") and label.endswith(">>"):
                # This is an HTML-like label - Graphviz will handle it
                dot.node(node_id, label, fillcolor=color, style=style, penwidth=penwidth)
            else:
                # Plain text label
                dot.node(node_id, label, fillcolor=color, style=style, penwidth=penwidth)
        
        # Add edges
        for from_id, to_id, edge_label, edge_style in self.edges:
            color = "black"
            penwidth = "1"
            style = "solid"
            
            if edge_style == "spne":
                color = "green"
                penwidth = "3"
            elif edge_style == "actual":
                color = "blue"
                penwidth = "2"
            elif edge_style == "terminal":
                color = "gray"
            
            # Remove edge labels - names are already in the boxes
            dot.edge(from_id, to_id, color=color, penwidth=penwidth, style=style)
        
        # Render
        dot.render(output_file, cleanup=True)
        print(f"Game tree saved to {output_file}.{format}")
        return dot
    
    def visualize_matplotlib(self, output_file: str = "game_tree.png", figsize=(24, 18), game_title: str = "Game Tree"):
        """Visualize using matplotlib and networkx with improved layout."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib and networkx libraries not available. Install with: pip install matplotlib networkx")
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes and build node data
        node_data = {}
        levels = {}
        for node_id, (label, node_type, data) in self.nodes.items():
            depth = data.get("depth", 0)
            if depth not in levels:
                levels[depth] = []
            levels[depth].append(node_id)
            G.add_node(node_id)
            node_data[node_id] = (label, node_type, data)
        
        # Add edges
        for from_id, to_id, edge_label, edge_style in self.edges:
            G.add_edge(from_id, to_id, label=edge_label, style=edge_style)
        
        # Use hierarchical layout with better spacing
        pos = {}
        max_nodes_per_level = max(len(nodes) for nodes in levels.values()) if levels else 1
        
        # Calculate dynamic spacing based on tree size
        y_spacing = max(3.0, 4.0 - len(levels) * 0.2)
        x_spacing = max(2.0, max_nodes_per_level * 0.3)
        
        for depth, node_ids in sorted(levels.items()):
            y = -depth * y_spacing
            x_start = -(len(node_ids) - 1) * x_spacing / 2
            for i, node_id in enumerate(node_ids):
                x = x_start + i * x_spacing
                pos[node_id] = (x, y)
        
        # Create figure with better styling
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        # Prepare node properties
        node_colors_map = {}
        node_sizes_map = {}
        node_labels_map = {}
        
        for node_id, (label, node_type, data) in node_data.items():
            # Clean label for display
            clean_label = label.replace('\n', '\\n')
            node_labels_map[node_id] = clean_label
            
            if node_type == "root":
                node_colors_map[node_id] = "#E0E0E0"  # Light gray
                node_sizes_map[node_id] = 3000
            elif node_type == "terminal":
                if data.get("is_spne"):
                    node_colors_map[node_id] = "#90EE90"  # Light green
                    node_sizes_map[node_id] = 2000
                elif data.get("is_actual"):
                    node_colors_map[node_id] = "#87CEEB"  # Sky blue
                    node_sizes_map[node_id] = 2000
                elif data.get("is_reds_best"):
                    # Highlight Red's best response terminal nodes
                    node_colors_map[node_id] = "#F08080"  # Light coral - Red's best response
                    node_sizes_map[node_id] = 1800
                else:
                    node_colors_map[node_id] = "#FFFACD"  # Lemon chiffon
                    node_sizes_map[node_id] = 1500
            elif node_type == "decision":
                player = data.get("player", "")
                if player == "blue":
                    node_colors_map[node_id] = "#E0F7FA"  # Light cyan
                else:
                    node_colors_map[node_id] = "#FCE4EC"  # Light pink
                node_sizes_map[node_id] = 1800
        
        # Convert to lists in correct order
        node_colors = [node_colors_map.get(node_id, "white") for node_id in G.nodes()]
        node_sizes = [node_sizes_map.get(node_id, 1000) for node_id in G.nodes()]
        node_labels = {node_id: node_labels_map.get(node_id, "") for node_id in G.nodes()}
        
        # Prepare edge properties
        edge_colors_list = []
        edge_widths_list = []
        for from_id, to_id in G.edges():
            edge_data = G[from_id][to_id]
            edge_style = edge_data.get('style', 'normal')
            if edge_style == "spne":
                edge_colors_list.append("#228B22")  # Forest green
                edge_widths_list.append(4.0)
            elif edge_style == "actual":
                edge_colors_list.append("#1E90FF")  # Dodger blue
                edge_widths_list.append(3.0)
            else:
                edge_colors_list.append("#808080")  # Gray
                edge_widths_list.append(1.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                             alpha=0.9, ax=ax, edgecolors='black', linewidths=1.5)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors_list, width=edge_widths_list, 
                              alpha=0.7, arrows=True, arrowsize=25, arrowstyle='->', ax=ax)
        
        # Draw labels with better formatting
        for node_id, (x, y) in pos.items():
            label = node_labels.get(node_id, "")
            if label:
                # Get node data for color formatting
                node_info = node_data.get(node_id, (None, None, {}))
                node_type = node_info[1] if len(node_info) > 1 else None
                data = node_info[2] if len(node_info) > 2 else {}
                
                # Use plain label if available (for decision nodes)
                if node_type == "decision" and "label_plain" in data:
                    label = data["label_plain"].replace('\n', '\\n')
                
                # Split label for multi-line display
                lines = label.split('\\n')
                for i, line in enumerate(lines):
                    # Color and bold the first line if it's "Blue" or "Red" in decision nodes
                    if node_type == "decision" and i == 0:
                        if data.get("is_blue", False):
                            color = 'blue'
                        else:
                            color = 'red'
                        ax.text(x, y + 0.15 - i*0.12, line, 
                               ha='center', va='center', fontsize=7, 
                               fontweight='bold', color=color,
                               bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', alpha=0.8, edgecolor='none'))
                    else:
                        ax.text(x, y + 0.15 - i*0.12, line, 
                               ha='center', va='center', fontsize=7, 
                               fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Add edge labels (only for decision edges, and only if not too cluttered)
        if len(G.edges()) < 50:  # Only show edge labels for smaller trees
            edge_labels = {}
            for from_id, to_id, edge_data in G.edges(data=True):
                edge_label = edge_data.get('label', '')
                edge_style = edge_data.get('style', 'normal')
                if edge_label and edge_style != "terminal":
                    edge_labels[(from_id, to_id)] = edge_label
            if edge_labels:
                nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#90EE90', label='SPNE'),
            mpatches.Patch(color='#87CEEB', label='Actual'),
            mpatches.Patch(color='#F08080', label="Red's Best Response (Blockade)"),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        
        ax.set_title(f"{game_title} - Draft Scenario", fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Game tree saved to {output_file}")
        plt.close()


def visualize_game3_tree(max_candidates: int = 8, output_format: str = "png"):
    """Create visualization for Game 3 scenario."""
    print("=" * 70)
    print("GAME TREE VISUALIZATION - GAME 3")
    print("=" * 70)
    
    # Load data
    load_champion_name_map()
    load_player_comfort_data("worlds2025.db")
    
    # Create draft state
    from test_game3_nash import create_game3_draft_state, GAME_3_T1_ACTUAL, GAME_3_KT_ACTUAL
    draft = create_game3_draft_state()
    
    # Get available champions
    all_picked = []
    for role in ROLE_ORDER:
        if draft.team1_picks.get(role):
            all_picked.append(draft.team1_picks[role])
        if draft.team2_picks.get(role):
            all_picked.append(draft.team2_picks[role])
    
    taken = set(draft.bans + all_picked)
    available = [c for c in ALL_EXTENDED_CHAMPIONS if c not in taken]
    
    # Initialize scorer and SPNE
    scorer = CompositionScorer()
    spne = SubgamePerfectNashEquilibrium(
        scorer,
        w1=0.15, w2=0.25, w3=0.6,
        use_memoization=True,
        skip_api_calls=False,
        show_progress=False,
        beam_width=1000,
        fast_heuristic=False
    )
    
    # Get players
    blue_players = [TEAM_2['players'][role] for role in ROLE_ORDER]
    red_players = [TEAM_1['players'][role] for role in ROLE_ORDER]
    
    # Get optimal and actual picks
    # First, normalize actual picks
    t1_actual_jungle = normalize_champion_name(GAME_3_T1_ACTUAL["JUNGLE"])
    t1_actual_mid = normalize_champion_name(GAME_3_T1_ACTUAL["MID"])
    kt_actual_mid = normalize_champion_name(GAME_3_KT_ACTUAL["MID"])
    
    # Build base teams from locked picks
    blue_team_base = [draft.team2_picks.get(role) for role in ROLE_ORDER]
    red_team_base = [draft.team1_picks.get(role) for role in ROLE_ORDER]
    
    # Fix JUNGLE to actual (as done in the analysis)
    blue_team_base[1] = t1_actual_jungle  # JUNGLE is fixed
    
    actual_blue = blue_team_base.copy()
    actual_blue[1] = t1_actual_jungle  # JUNGLE
    actual_blue[2] = t1_actual_mid  # MID
    
    actual_red = red_team_base.copy()
    actual_red[2] = kt_actual_mid  # MID
    
    # For optimal picks, we need to calculate them or use a simplified approach
    # Option 1: Run a quick SPNE calculation (simplified)
    # Option 2: Use actual picks as placeholder (for testing)
    print("\nCalculating optimal picks for visualization...")
    print("(This may take a moment - using cached calculations when possible)\n")
    
    # Quick calculation: find best Blue MID pick, then Red's best response
    # Note: JUNGLE is fixed to actual in the code, so we'll use that
    fixed_jungle = t1_actual_jungle
    blue_team_base[1] = fixed_jungle  # Ensure JUNGLE is set
    valid_mid_blue = [c for c in EXTENDED_CHAMPION_POOL["MID"] if c in available]
    valid_mid_red = [c for c in EXTENDED_CHAMPION_POOL["MID"] if c in available]
    
    best_blue_mid = None
    best_payoff = float('-inf')
    best_red_response = None
    
    print(f"Evaluating {len(valid_mid_blue)} Blue MID options...")
    for blue_mid in valid_mid_blue[:min(20, len(valid_mid_blue))]:  # Limit for speed
        test_blue = blue_team_base.copy()
        test_blue[1] = fixed_jungle
        test_blue[2] = blue_mid
        
        # Find Red's best response
        best_red_mid = None
        best_red_payoff = float('inf')
        for red_mid in valid_mid_red:
            if red_mid == blue_mid:
                continue
            test_red = red_team_base.copy()
            test_red[2] = red_mid
            payoff = spne.calculate_payoff(test_blue, test_red, blue_players, red_players, 'na1', use_fast_mode=True)
            if payoff < best_red_payoff:
                best_red_payoff = payoff
                best_red_mid = red_mid
        
        # Blue wants maximum payoff
        if best_red_payoff > best_payoff:
            best_payoff = best_red_payoff
            best_blue_mid = blue_mid
            best_red_response = best_red_mid
    
    if best_blue_mid:
        optimal_blue = blue_team_base.copy()
        optimal_blue[1] = fixed_jungle
        optimal_blue[2] = best_blue_mid
        optimal_red = red_team_base.copy()
        optimal_red[2] = best_red_response
        print(f"Optimal picks calculated: Blue MID={best_blue_mid}, Red MID={best_red_response}")
    else:
        print("Using actual picks as optimal (fallback)")
        optimal_blue = actual_blue.copy()
        optimal_red = actual_red.copy()
    
    # Create visualizer
    visualizer = GameTreeVisualizer(spne, blue_players, red_players)
    
    # Build tree
    print("Building game tree...")
    # Note: JUNGLE is fixed, so draft order is just Blue MID, then Red MID
    draft_order = ['blue', 'red']  # Blue picks MID, Red picks MID
    root_id = visualizer.build_tree(
        draft, available, draft_order,
        optimal_blue, optimal_red,
        actual_blue, actual_red,
        max_candidates_per_level=max_candidates,
        blue_is_team1=False  # Game 3: T1 (Blue) is team2
    )
    
    print(f"Tree built: {len(visualizer.nodes)} nodes, {len(visualizer.edges)} edges")
    print(f"SPNE path: {len(visualizer.spne_path)} nodes")
    print(f"Actual path: {len(visualizer.actual_path)} nodes")
    
    # Visualize
    print("\nGenerating visualization...")
    visualization_success = False
    if HAS_GRAPHVIZ:
        try:
            visualizer.visualize_graphviz("game3_tree", output_format)
            visualization_success = True
        except Exception as e:
            print(f"Graphviz failed (system executable not found): {e}")
            print("Falling back to matplotlib...")
            if HAS_MATPLOTLIB:
                try:
                    visualizer.visualize_matplotlib("game3_tree.png", game_title="Game Tree - Game 3")
                    visualization_success = True
                except Exception as e2:
                    print(f"Matplotlib also failed: {e2}")
            else:
                print("Matplotlib not available. Install with: pip install matplotlib networkx")
    elif HAS_MATPLOTLIB:
        try:
            visualizer.visualize_matplotlib("game3_tree.png", game_title="Game Tree - Game 3")
            visualization_success = True
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("ERROR: No visualization library available!")
        print("Install one of:")
        print("  pip install graphviz  (requires system Graphviz: https://graphviz.org/download/)")
        print("  pip install matplotlib networkx")
    
    if not visualization_success:
        print("\nNote: To use Graphviz, install the system package:")
        print("  Windows: Download from https://graphviz.org/download/")
        print("  Or use: winget install graphviz")
        print("  Then add Graphviz bin folder to PATH")
    
    scorer.close()


  
def visualize_game1_tree(max_candidates: int = 8, output_format: str = "png"):
    """Create visualization for Game 1 scenario."""
    print("=" * 70)
    print("GAME TREE VISUALIZATION - GAME 1")
    print("=" * 70)
    
    # Load data
    load_champion_name_map()
    load_player_comfort_data("worlds2025.db")
    
    # Create draft state
    from test_game1_nash import create_game1_draft_state, GAME_1_T1_ACTUAL, GAME_1_KT_ACTUAL
    draft = create_game1_draft_state()
    
    # Get available champions
    all_picked = []
    for role in ROLE_ORDER:
        if draft.team1_picks.get(role):
            all_picked.append(draft.team1_picks[role])
        if draft.team2_picks.get(role):
            all_picked.append(draft.team2_picks[role])
    
    taken = set(draft.bans + all_picked)
    available = [c for c in ALL_EXTENDED_CHAMPIONS if c not in taken]
    
    # Initialize scorer and SPNE
    scorer = CompositionScorer()
    spne = SubgamePerfectNashEquilibrium(
        scorer,
        w1=0.15, w2=0.25, w3=0.6,  # Match test_game1_nash.py weights
        use_memoization=True,
        skip_api_calls=False,
        show_progress=False,
        beam_width=1000,
        fast_heuristic=False
    )
    
    # Get players
    # Game 1: KT is Blue (team1), T1 is Red (team2)
    blue_players = [TEAM_1['players'][role] for role in ROLE_ORDER]
    red_players = [TEAM_2['players'][role] for role in ROLE_ORDER]
    
    # Normalize actual picks
    kt_actual_bot = normalize_champion_name(GAME_1_KT_ACTUAL["BOT"])
    kt_actual_support = normalize_champion_name(GAME_1_KT_ACTUAL["SUPPORT"])
    t1_actual_top = normalize_champion_name(GAME_1_T1_ACTUAL["TOP"])
    
    # Build base teams from locked picks
    # KT is Blue (team1), T1 is Red (team2)
    blue_team_base = [draft.team1_picks.get(role) for role in ROLE_ORDER]
    red_team_base = [draft.team2_picks.get(role) for role in ROLE_ORDER]
    
    actual_blue = blue_team_base.copy()
    actual_blue[3] = kt_actual_bot  # BOT
    actual_blue[4] = kt_actual_support  # SUPPORT
    
    actual_red = red_team_base.copy()
    actual_red[0] = t1_actual_top  # TOP
    
    # Calculate optimal picks using the same method as test_game1_nash.py
    print("\nCalculating optimal picks for visualization...")
    print("(This may take a moment - using cached calculations when possible)\n")
        
    # Get valid champions for each role
    valid_bot = [c for c in EXTENDED_CHAMPION_POOL.get("BOT", []) if c in available]
    valid_support = [c for c in EXTENDED_CHAMPION_POOL.get("SUPPORT", []) if c in available]
    valid_top = [c for c in EXTENDED_CHAMPION_POOL.get("TOP", []) if c in available]
    
    best_blue_bot = None
    best_blue_support = None
    best_payoff = float('-inf')
    best_red_top = None
    
    print(f"Evaluating {len(valid_bot)} BOT × {len(valid_support)} SUPPORT options...")
    print("Note: This uses the same calculation as test_game1_nash.py (use_fast_mode=False)")
    # Test ALL combinations (matching test_game1_nash.py)
    for bot in valid_bot:
        for support in valid_support:
            if bot == support:
                continue
            test_blue = blue_team_base.copy()
            test_blue[3] = bot
            test_blue[4] = support
            
            # Find Red's best response (test ALL Red TOP picks)
            best_red_top = None
            best_red_payoff = float('inf')
            for top in valid_top:
                if top in test_blue:
                    continue
                test_red = red_team_base.copy()
                test_red[0] = top
                # Use use_fast_mode=False to match test_game1_nash.py
                payoff = spne.calculate_payoff(test_blue, test_red, blue_players, red_players, 'na1', use_fast_mode=False)
                if payoff < best_red_payoff:
                    best_red_payoff = payoff
                    best_red_top = top
            
            # Blue wants maximum payoff
            if best_red_payoff > best_payoff:
                best_payoff = best_red_payoff
                best_blue_bot = bot
                best_blue_support = support
    
    if best_blue_bot:
        optimal_blue = blue_team_base.copy()
        optimal_blue[3] = best_blue_bot
        optimal_blue[4] = best_blue_support
        optimal_red = red_team_base.copy()
        optimal_red[0] = best_red_top
        print(f"Optimal picks calculated: Blue BOT={best_blue_bot}, SUPPORT={best_blue_support}, Red TOP={best_red_top}")
    else:
        print("Using actual picks as optimal (fallback)")
        optimal_blue = actual_blue.copy()
        optimal_red = actual_red.copy()
    
    # Create visualizer
    visualizer = GameTreeVisualizer(spne, blue_players, red_players)
    
    # Build tree
    print("Building game tree...")
    # Draft order: Blue picks BOT + SUPPORT, then Red picks TOP
    draft_order = ['blue', 'blue', 'red']
    root_id = visualizer.build_tree(
        draft, available, draft_order,
        optimal_blue, optimal_red,
        actual_blue, actual_red,
        max_candidates_per_level=max_candidates,
        blue_is_team1=True  # Game 1: KT (Blue) is team1
    )
    
    print(f"Tree built: {len(visualizer.nodes)} nodes, {len(visualizer.edges)} edges")
    print(f"SPNE path: {len(visualizer.spne_path)} nodes")
    print(f"Actual path: {len(visualizer.actual_path)} nodes")
    
    # Visualize
    print("\nGenerating visualization...")
    visualization_success = False
    if HAS_GRAPHVIZ:
        try:
            visualizer.visualize_graphviz("game1_tree", output_format)
            visualization_success = True
        except Exception as e:
            print(f"Graphviz failed (system executable not found): {e}")
            print("Falling back to matplotlib...")
            if HAS_MATPLOTLIB:
                try:
                    visualizer.visualize_matplotlib("game1_tree.png", game_title="Game Tree - Game 1")
                    visualization_success = True
                except Exception as e2:
                    print(f"Matplotlib also failed: {e2}")
            else:
                print("Matplotlib not available. Install with: pip install matplotlib networkx")
    elif HAS_MATPLOTLIB:
        try:
            visualizer.visualize_matplotlib("game1_tree.png", game_title="Game Tree - Game 1")
            visualization_success = True
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("ERROR: No visualization library available!")
        print("Install one of:")
        print("  pip install graphviz  (requires system Graphviz: https://graphviz.org/download/)")
        print("  pip install matplotlib networkx")
    
    if not visualization_success:
        print("\nNote: To use Graphviz, install the system package:")
        print("  Windows: Download from https://graphviz.org/download/")
        print("  Or use: winget install graphviz")
        print("  Then add Graphviz bin folder to PATH")
    
    scorer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize game tree")
    parser.add_argument("--game", type=int, choices=[1, 3], default=3,
                       help="Game number to visualize (default: 3)")
    parser.add_argument("--max-candidates", type=int, default=4,
                       help="Maximum champions to show at each decision level (default: 4)")
    parser.add_argument("--format", type=str, default="png",
                       choices=["png", "pdf", "svg"],
                       help="Output format (default: png)")
    args = parser.parse_args()
    
    if args.game == 1:
        visualize_game1_tree(max_candidates=args.max_candidates, output_format=args.format)
    else:
        visualize_game3_tree(max_candidates=args.max_candidates, output_format=args.format)

