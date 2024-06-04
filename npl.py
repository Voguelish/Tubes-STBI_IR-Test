import functions
import re

# Define the path to your files
datasets = {
    "npl": {
        "documents": "npl/doc-text",
        "queries": "npl/query-text",
        "stopwords": "npl/term-vocab",
        "qrels": "npl/rlv-ass"
    }
}

def parse_documents(content):
    documents = re.split(r'\s*/\s*', content)  # Split by '/'
    parsed_docs = []
    for doc in documents:
        doc = doc.strip()  # Remove leading and trailing whitespace
        if doc:  # Skip empty documents
            parsed_docs.append(doc)
    return parsed_docs

def parse_queries(content):
    queries = re.split(r'\s*/\s*', content)  # Split by '/'
    parsed_queries = []
    for query in queries:
        query = query.strip()  # Remove leading and trailing whitespace
        if query:  # Skip empty queries
            parsed_queries.append(query)
    return parsed_queries

def parse_qrels(content):
    qrels = {}
    relevancy_data = re.split(r'\s*/\s*', content)  # Split by '/'
    qid = 0
    for data in relevancy_data:
        data = data.strip()  # Remove leading and trailing whitespace
        if data:  # Skip empty data
            qid += 1
            qrels[qid] = set(map(int, data.split()))
    return qrels

def parse_stopwords(content):
    stopwords = []
    terms = re.split(r'\s*/\s*', content)  # Split by '/'
    for term in terms:
        term = term.strip()  # Remove leading and trailing whitespace
        if term:  # Skip empty terms
            # Remove the leading numbers and whitespace
            term = re.sub(r'^\d+\s*', '', term)
            stopwords.append(term)
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
        stopwords_content = functions.read_file(dataset_files["stopwords"])
        qrels_content = functions.read_file(dataset_files["qrels"])

        # Parse files
        documents = parse_documents(doc_content)
        queries = parse_queries(query_content)
        stopwords = parse_stopwords(stopwords_content)
        qrels = parse_qrels(qrels_content)

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