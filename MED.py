import functions
import re

# Define the path to your files
datasets = {
    "med": {
        "documents": "med/MED.all",
        "queries": "med/MED.QRY",
        "qrels": "med/MED.REL"
    }
}

def parse_documents(content):
    documents = re.split(r'\.I \d+', content)[1:]  # Skip the first empty split
    parsed_docs = []
    for doc in documents:
        # Since MED.all has only .I and .W, we consider the entire text as the document
        text = re.search(r'\.W\s+([\s\S]+?)(\.I|$)', doc)
        parsed_docs.append((text.group(1) if text else '').strip())
    return parsed_docs

def parse_queries(content):
    queries = re.split(r'\.I \d+', content)[1:]  # Skip the first empty split
    parsed_queries = []
    for query in queries:
        # Since MED.QRY has only .I and .W, we consider the entire text as the query
        text = re.search(r'\.W\s+([\s\S]+?)(\.I|$)', query)
        parsed_queries.append((text.group(1) if text else '').strip())
    return parsed_queries

def parse_qrels(filepath):
    qrels = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            qid = int(parts[0])
            did = int(parts[2])  # The relevant document ID is the third element
            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].add(did)
    return qrels

def process_dataset():
    tf_methods = ['n', 'l', 'a', 'b']
    idf_methods = ['n', 't']
    normalizations = ['n', 'c']
    
    weighting_schemes = [(tf_d, idf_d, norm_d, tf_q, idf_q, norm_q) 
                         for tf_d in tf_methods for idf_d in idf_methods for norm_d in normalizations
                         for tf_q in tf_methods for idf_q in idf_methods for norm_q in normalizations]
    
    results = []

    for dataset_name, dataset_files in datasets.items():
        print(f"Processing {dataset_name.upper()} dataset...")

        # Read files
        doc_content = functions.read_file(dataset_files["documents"])
        query_content = functions.read_file(dataset_files["queries"])
        qrels = parse_qrels(dataset_files["qrels"])

        # Parse files
        documents = parse_documents(doc_content)
        queries = parse_queries(query_content)

        # For the MED dataset, stopwords are not provided
        stopwords = set()

        for stemming in [False, True]:
            # Preprocess documents and queries
            processed_documents = [functions.preprocess_text(doc, stopwords, stemming=stemming) for doc in documents]
            processed_queries = [functions.preprocess_text(query, stopwords, stemming=stemming) for query in queries]

            for scheme in weighting_schemes:
                doc_scheme = scheme[:3]
                query_scheme = scheme[3:]
                
                similarities = functions.compute_tfidf_and_similarity(processed_documents, processed_queries, doc_scheme, query_scheme)
                map_score = functions.calculate_map(similarities, qrels)
                
                results.append((map_score, scheme, dataset_name, "stemming" if stemming else "no stemming"))
    
    # Sort results by MAP score
    results.sort(reverse=True, key=lambda x: x[0])

    # Output results
    print("\nRanking of weighting schemes by MAP score:")
    for score, scheme, dataset_name, stemming in results:
        doc_scheme_str = ''.join(scheme[:3])
        query_scheme_str = ''.join(scheme[3:])
        print(f"{dataset_name.upper()} - {doc_scheme_str}.{query_scheme_str} ({stemming}): MAP Score = {score:.4f}")

# Run the function
if __name__ == "__main__":
    process_dataset()
