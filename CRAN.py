import functions
import re

datasets = {
    "cran": {
        "documents" : "cran/cran.all.1400",
        "queries": "cran/cran.qry",
        "stopwords": None,
        "qrels": "cran/cranqrel"
    }
}

def parse_documents(content):
    documents = re.split(r'\.I \d+', content)[1:]  # Skip the first empty split
    parsed_docs = []
    for doc in documents:
        title = re.search(r'\.T\s+([\s\S]+?)\.(A|B|W|N|X)', doc)
        body = re.search(r'\.B\s+([\s\S]+?)\.(T|A|W|N|X)', doc)
        abstract = re.search(r'\.W\s+([\s\S]+?)\.(T|A|B|N|X|$)', doc)
        text = (title.group(1) if title else '') + ' ' + \
               (body.group(1) if body else '') + ' ' + \
               (abstract.group(1) if abstract else '')
        parsed_docs.append(text.replace('\n', ' ').strip())
    return parsed_docs

def parse_queries(content):
    queries = re.split(r'\.I \d+', content)[1:]  # Skip the first empty split
    parsed_queries = []
    for query in queries:
        text = re.search(r'\.W\s+([\s\S]+?)(?=\.I|$)', query)  # Look ahead to find the next .I or end of string
        if text:
            parsed_query = text.group(1).replace('\n', ' ').strip()  # Replace newlines with spaces and strip
            parsed_queries.append(parsed_query)
    return parsed_queries

def parse_qrels(filepath):
    qrels = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            qid = int(parts[0])
            did = int(parts[1])
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

        # For the CRAN dataset, stopwords are not provided
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