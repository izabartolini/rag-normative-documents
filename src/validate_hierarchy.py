import json
from collections import Counter
from pathlib import Path

def validate(elements):
    counts = Counter(e['path'] for e in elements)
    duplicates = {p: n for p, n in counts.items() if n > 1}
    orphans = [e['path'] for e in elements
               if len(e['path'].split(' > ')) > 2
               and e['path'].rsplit(' > ', 1)[0] not in counts]
    empty = [e['path'] for e in elements if not e['text'].strip()]
    print('Total:', len(elements))
    print('Caminhos distintos:', len(counts))
    print('Caminhos duplicados:', len(duplicates))
    print('Caminhos sem pai:', len(orphans))
    print('Textos vazios:', len(empty))
    print('Tipos:', dict(Counter(e['type'] for e in elements)))
    for p, n in duplicates.items():
        print('Duplicado:', p, n)
    for p in orphans:
        print('Sem pai:', p)
    for p in empty:
        print('Vazio:', p)


if __name__ == '__main__':
    validate(json.loads(Path('data/processed/lgpd_hierarchy.json').read_text(encoding='utf-8')))
