import os
import json

def get_samples(file_path):
    with open(file_path, 'r', encoding = 'utf-8') as f:
        tree = json.loads(f.read())
    return tree 


def build_nodes_from_tree(tree):
    nodes = []
    if "root" in tree.keys():
        root = tree.get("root")
    else:
        root = tree
    root.pop("text")
    if "token_logprobs" in root.keys():
        root.pop("token_logprobs")
        root.pop("tokens")
        root.pop("top_logprobs")
    if "children" in root.keys():
        children = root["children"]
        root.pop("children")
        nodes.append(root)
        for child in children:
            child_nodes = build_nodes_from_tree(child)
            if len(child_nodes) > 0:
                nodes.extend(child_nodes)
        return nodes
    else:
        return [root]
def save_nodes(nodes):
    print(f"Number of nodes: {len(nodes)}")
    nodes = sorted(nodes, key= lambda x: int(x["P"]), reverse=True)
    with open("nodes.jsonl", "w", encoding="utf-8") as f:
        for node in nodes:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")
    print(f"Save nodes successfully!")

if __name__ == "__main__":
    tree = get_samples("results/local/deepseek/DeepSeek-R1-Distill-Qwen-1.5B_4-4-4_e839ebaf.json")
    nodes = build_nodes_from_tree(tree)
    save_nodes(nodes)

