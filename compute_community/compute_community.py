import networkx as nx
from collections import defaultdict

global_communities = []

def calculate_importance():
    degrees = defaultdict(int)
    with open("c:/lwp/experenment/mymodel/compute_community/graph.txt", "r") as file:
        for line in file:
            u, v = map(int, line.strip().split())
            degrees[u] += 1
            degrees[v] += 1

    # Write the results to a file
    with open("c:/lwp/experenment/mymodel/compute_community/node_importance.txt", "w") as output_file:
        for node, degree in sorted(degrees.items()):
            output_file.write(f"{node} {degree}\n")

    print("Degree statistics saved to node_importance.txt.")

    # Set importance values for all nodes
    all_nodes = set(range(1, max(degrees.keys()) + 1))
    existing_nodes = set(degrees.keys())
    missing_nodes = all_nodes - existing_nodes
    with open("c:/lwp/experenment/mymodel/compute_community/node_importance.txt", "a") as output_file:
        for node in missing_nodes:
            output_file.write(f"{node} 0\n")

    print("Importance values for missing nodes set to 0.")

calculate_importance()


def compute_maximal_k_core(graph, k):
    return nx.k_core(graph, k=k)

def find_min_weight_node(nodes, node_importance):
    return min(nodes, key=node_importance.get)

def dfs(Ck, node, k, cohesive_nodes):
    if node not in Ck:
        return
    cohesive_nodes.add(node)
    neighbors = list(Ck.neighbors(node))  # Make a copy of the list of neighbors
    
    for neighbor in neighbors:
        if neighbor in Ck:  # Check if the neighbor exists in the graph
            Ck.remove_edge(node, neighbor)
    # Update neighbors after removing edges
    neighbors = list(Ck.neighbors(node))  # Update the list of neighbors
    for neighbor in neighbors:    
        if neighbor in Ck and len(list(Ck.neighbors(neighbor))) < k:
            dfs(Ck, neighbor, k, cohesive_nodes)
    Ck.remove_node(node)


def compute_influential_communities(G, node_importance, k):
    global global_communities
    Ck = compute_maximal_k_core(G, k)
    iteration = 1
    while Ck.number_of_nodes() > 0:
        connected_components = list(nx.connected_components(Ck))
        min_influence_communities = []
        min_importance = float('inf')
        for community in connected_components:
            community_importance = min(node_importance[node] for node in community)
            if community_importance < min_importance:
                min_influence_communities = [list(community)]
                min_importance = community_importance
            elif community_importance == min_importance:
                min_influence_communities.append(list(community))
        min_influence_community = max(min_influence_communities, key=len)
        min_weight_node = find_min_weight_node(min_influence_community, node_importance)

        # DFS to remove non-cohesive nodes
        cohesive_nodes = set()
        dfs(Ck, min_weight_node, k, cohesive_nodes)

        G.remove_nodes_from(cohesive_nodes)
        Ck = compute_maximal_k_core(G, k)
        # Check if min_influence_community still exists after node removal
        community_still_exists = any(node in Ck for node in min_influence_community)
        if not community_still_exists:
            global_communities.append(min_influence_community)
            print(f"find a k-influential community with size {len(min_influence_community)}")
        print(f"Iteration {iteration}: Remaining nodes: {Ck.number_of_nodes()}, Remaining edges: {Ck.number_of_edges()}")
        iteration += 1
    print(f"Total influential communities found: {len(global_communities)}")

def write_communities_to_file(communities, filename):
    with open(filename, 'w') as f:
        for i, community in enumerate(communities):  
            f.write(f"Community {i+1}:\n")
            f.write(" ".join(map(str, community)) + "\n")

# Read graph information from graph.txt
graph_file = "c:/lwp/experenment/mymodel/compute_community/graph.txt"
G = nx.read_edgelist(graph_file)

# Read node importance information from node_importance.txt
importance_file = "c:/lwp/experenment/mymodel/compute_community/node_importance.txt"
node_importance = {}
with open(importance_file, 'r') as f:
    for line in f:
        node, importance = line.strip().split()
        node_importance[node] = float(importance)

# Example usage
k = 7  # Specify the k value
compute_influential_communities(G, node_importance, k)

# Output the communities to a file
write_communities_to_file(global_communities, "c:/lwp/experenment/mymodel/compute_community/community.txt")
