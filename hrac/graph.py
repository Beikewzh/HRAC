import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class GraphEncoderDecoder:
    def __init__(self, state_dim, subgoal_dim):
        self.encoder = MLP(state_dim, subgoal_dim)
        self.decoder = lambda g_u, g_v: torch.dot(g_u, g_v)

    def encode(self, state):
        return self.encoder(state)

    def decode(self, g_u, g_v):
        return self.decoder(g_u, g_v)

class Graph:
    def __init__(self, num_nodes, state_dim, subgoal_dim, epsilon_d):
        self.num_nodes = num_nodes
        self.state_dim = state_dim
        self.epsilon_d = epsilon_d
        
        self.nodes = np.zeros((num_nodes, state_dim))
        self.adj_matrix = np.zeros((num_nodes, num_nodes))
        self.encoder_decoder = GraphEncoderDecoder(state_dim, subgoal_dim)
        self.history = np.zeros(num_nodes)

    def find_empty_node(self):
        for i in range(self.num_nodes):
            if np.all(self.nodes[i] == 0):
                return i
        return None

    def add_node(self, new_state, prev_state_index):
#        new_state_tensor = torch.tensor(new_state, dtype=torch.float32)
#        new_state_feature = self.encoder_decoder.encode(new_state_tensor).detach().numpy()
        self.history = self.history + 1

        min_dis = np.linalg.norm(new_state - self.nodes[0])
        min_index = 0
        for i, node in enumerate(self.nodes):
            if not np.all(node == 0):
                if np.linalg.norm(new_state - node) <= min_dis:
                    min_index = i
                    min_dis = np.linalg.norm(new_state - node)
                
        if min_dis <= self.epsilon_d:
            if (i != prev_state_index) and (prev_state_index is not None):
                self.adj_matrix[i, prev_state_index] += 1
                self.adj_matrix[prev_state_index, i] += 1
            self.history[i] = 0
            return i  # State is already represented in the graph

        new_node_index = self.find_empty_node()
        if new_node_index is not None:
            self.nodes[new_node_index] = new_state
            self.history[new_node_index] = 0
        else:
            new_node_index = np.argmax(self.history)
            self.nodes[new_node_index] = new_state
            self.history[new_node_index] = 0
            self.adj_matrix[new_node_index, :] = 0
            self.adj_matrix[:, new_node_index] = 0

        # Debug prints
        #print(f"New node index: {new_node_index}")
        #print(f"Previous node index: {prev_state_index}")

        if prev_state_index is not None:
            self.adj_matrix[new_node_index, prev_state_index] += 1
            self.adj_matrix[prev_state_index, new_node_index] += 1

        return new_node_index

    def normalize_adj_matrix(self):
        max_val = np.max(np.abs(self.adj_matrix))
        if max_val != 0:
            return self.adj_matrix / max_val
        else:
            return self.adj_matrix


    def compute_loss(self):
        loss = 0
        normalized_adj = self.normalize_adj_matrix()

        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if not np.all(self.nodes[u] == 0) and not np.all(self.nodes[v] == 0):
                    g_u = self.encoder_decoder.encode(torch.tensor(self.nodes[u], dtype=torch.float32))
                    g_v = self.encoder_decoder.encode(torch.tensor(self.nodes[v], dtype=torch.float32))
                    predicted_similarity = self.encoder_decoder.decode(g_u, g_v)
                    true_similarity = normalized_adj[u, v]

                    loss += (predicted_similarity - true_similarity) ** 2
        return loss

    def train_encoder_decoder(self, optimizer):
        optimizer.zero_grad()
        loss = self.compute_loss()
        loss.backward()
        optimizer.step()
        return loss.item()

    def print_graph(self):
        print("Nodes in the graph:")
        for i, node in enumerate(self.nodes):
            if not np.all(node == 0):
                print(f"Node {i}: {node}")
        print("\nAdjacency Matrix:")
        print(self.adj_matrix)

########################################################## Test the graph ##########################################################
if __name__ == "__main__":
    graph = Graph(num_nodes=10, state_dim=10, subgoal_dim=3, epsilon_d=10)
    optimizer = optim.Adam(graph.encoder_decoder.encoder.parameters(), lr=0.001)

    prev_node_index = None
    count = 0 
    for _ in range(100):
        state = np.random.rand(10) * 10
        new_node_index = graph.add_node(state, prev_node_index)
        prev_node_index = new_node_index

#        if count % 5 == 0:
#            loss = graph.train_encoder_decoder(optimizer)
#            print(f"Training loss: {loss}")
#        count += 1

    graph.print_graph()
