import sqlite3

# Get all champions from database
conn = sqlite3.connect('lolchampiontags.db')
cursor = conn.cursor()
cursor.execute('SELECT name FROM champions')
db_champs = {row[0].lower() for row in cursor.fetchall()}
conn.close()

# Extended champion pool
extended = [
    'Ksante', 'Sion', 'Rumble', 'Ambessa', 'RekSai', 'Renekton', 
    'Ornn', 'Mordekaiser', 'Aatrox', 'Jax', 'Poppy', 'Camille', 
    'Galio', 'Yorick', 'Kled', 'Jayce', 'Gwen', 'Gragas', 'Aurora', 'Gnar',
    'XinZhao', 'JarvanIV', 'Wukong', 'Trundle', 'Vi', 'Qiyana', 
    'Pantheon', 'Naafiri', 'DrMundo', 'Nocturne', 'Sejuani', 'Skarner', 
    'Poppy', 'Viego', 'Maokai', 'Nidalee',
    'Ryze', 'Orianna', 'Taliyah', 'Aurora', 'Azir', 'Galio', 
    'Cassiopeia', 'Viktor', 'Yone', 'Akali', 'Annie', 'Sylas', 
    'Anivia', 'LeBlanc', 'Hwei', 'Syndra', 'Mel', 'Ziggs', 'Smolder', 
    'Ahri', 'Swain', 'Zoe',
    'Corki', 'KaiSa', 'Yunara', 'Caitlyn', 'Ezreal', 'Sivir', 
    'Varus', 'Xayah', 'Lucian', 'Ziggs', 'Draven', 'Jhin', 'Ashe', 
    'Kalista', 'Vayne', 'MissFortune', 'Smolder', 'Jinx',
    'Alistar', 'Rakan', 'Neeko', 'Braum', 'Leona', 'Bard', 
    'Nautilus', 'Rell', 'Poppy', 'Karma', 'Nami', 'Lulu', 
    'Renata', 'TahmKench'
]

# Remove duplicates and check
extended_unique = list(set(extended))
missing = []

for champ in extended_unique:
    champ_lower = champ.lower()
    champ_normalized = champ_lower.replace(' ', '').replace("'", '').replace('.', '')
    
    # Check exact match
    if champ_lower not in db_champs:
        # Check normalized match
        found = False
        for db_champ in db_champs:
            db_normalized = db_champ.replace(' ', '').replace("'", '').replace('.', '')
            if champ_normalized == db_normalized:
                found = True
                break
        if not found:
            missing.append(champ)

print(f"Total unique champions in extended pool: {len(extended_unique)}")
print(f"Champions in database: {len(db_champs)}")
print(f"\nMissing champions ({len(missing)}):")
for champ in sorted(missing):
    print(f"  - {champ}")

