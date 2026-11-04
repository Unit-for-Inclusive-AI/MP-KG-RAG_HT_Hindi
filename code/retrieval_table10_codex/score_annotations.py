"""Score the saved rankings after real pooled relevance grades are supplied."""
import argparse
import csv
import json
import math
from pathlib import Path


def score(rankings, grades, post_ids):
    values = []
    no_grade2 = 0
    no_positive = 0
    for row in rankings:
        post = row['ID']
        if post not in post_ids:
            continue
        pool = [g for (pid, _), g in grades.items() if pid == post]
        retrieved = [grades[(post, eid)] for eid in row['hits']]
        relevant = pool.count(2)
        hits = retrieved.count(2)
        dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(retrieved))
        ideal = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(sorted(pool, reverse=True)[:5]))
        no_grade2 += relevant == 0
        no_positive += ideal == 0
        values.append((hits / 5, hits / relevant if relevant else 0, dcg / ideal if ideal else 0))
    return {'P@5': sum(v[0] for v in values)/len(values),
            'R@5': sum(v[1] for v in values)/len(values),
            'nDCG@5': sum(v[2] for v in values)/len(values),
            'n_posts': len(values), 'no_grade2_in_pool': no_grade2,
            'no_positive_grade_in_pool': no_positive}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('annotations', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    rankings = json.loads((root/'rankings.json').read_text())
    sample = json.loads((root/'sample_100.json').read_text())
    with args.annotations.open(newline='', encoding='utf-8') as f:
        pool = list(csv.DictReader(f))
    grades = {}
    for row in pool:
        key = row['post_id'], row['evidence_id']
        if key in grades:
            raise ValueError(f'Duplicate annotation key {key}')
        a, b, final = (row[k].strip() for k in ('annotator_1_grade', 'annotator_2_grade', 'adjudicated_grade'))
        if a not in {'0', '1', '2'} or b not in {'0', '1', '2'}:
            raise ValueError(f'Both annotators must supply a real grade 0, 1, or 2 for {key}')
        if final:
            if final not in {'0', '1', '2'}:
                raise ValueError(f'Invalid adjudicated grade for {key}')
            grade = int(final)
        elif a == b:
            grade = int(a)
        else:
            raise ValueError(f'Annotator disagreement needs adjudication for {key}')
        grades[key] = grade
    required = {(r['ID'], eid) for r in rankings for eid in r['hits']}
    if set(grades) != required:
        raise ValueError('Annotation keys do not match the complete retrieved passage pool')
    results = []
    for name in dict.fromkeys(r['retriever'] for r in rankings):
        for group in ['All', 'Romanized Hinglish', 'Devanagari Hindi']:
            ids = {str(r['ID']) for r in sample if group == 'All' or r['script'] == group}
            results.append({'retriever': name, 'script': group,
                            **score([r for r in rankings if r['retriever'] == name], grades, ids)})
    path = root/'annotated_metrics.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(path)


if __name__ == '__main__':
    main()
