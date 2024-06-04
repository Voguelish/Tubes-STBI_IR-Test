import functions
import re

# Define the path to your files
datasets = {
    "time": {
        "documents": "time/TIME.ALL",
        "queries": "time/TIME.QUE",
        "qrels": "time/TIME.REL",
        "stopwords": "time/TIME.STP"
    }
}

def parse_documents(content):
    documents = re.split(r'\*TEXT \d+', content)[1:]  # Skip the first empty split
    parsed_docs = []
    for doc in documents:
        text = re.search(r'PAGE \d+\n\n([\s\S]+?)(?=\*TEXT|$)', doc)
        parsed_docs.append((text.group(1) if text else '').strip())
    return parsed_docs

def parse_queries(content):
    queries = re.findall(r'\*FIND\s+\d+\n\n([\s\S]+?)(?=\*FIND|$)', content)
    return [query.strip() for query in queries]

def parse_qrels(filepath):
    qrels = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # Skip lines with insufficient data
            qid = int(parts[0])
            dids = list(map(int, parts[1:]))
            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].update(dids)
    return qrels

def parse_stopwords(filepath):
    with open(filepath, 'r') as file:
        stopwords = set(word.strip().upper() for word in file)
    return stopwords

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

        # Load stopwords
        stopwords = parse_stopwords(dataset_files["stopwords"])

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
