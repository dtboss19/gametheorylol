"""
Import CSV data with role columns into the database.
Handles adding new columns if they don't exist and updating all records.
"""

import sqlite3
import csv
import os
from pathlib import Path

# Database and CSV file paths (relative to project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "lolchampiontags.db")
CSV_PATH = "lol_champion_tags.csv"

def add_role_columns_if_needed(conn):
    """Add role columns to the database if they don't exist."""
    cursor = conn.cursor()
    
    # Check which columns exist
    cursor.execute("PRAGMA table_info(champions)")
    existing_columns = [row[1] for row in cursor.fetchall()]
    
    role_columns = [
        ('is_top', 'INTEGER DEFAULT 0'),
        ('is_jungle', 'INTEGER DEFAULT 0'),
        ('is_mid', 'INTEGER DEFAULT 0'),
        ('is_adc', 'INTEGER DEFAULT 0'),
        ('is_support', 'INTEGER DEFAULT 0'),
        ('is_ranged', 'INTEGER DEFAULT 0')
    ]
    
    for col_name, col_type in role_columns:
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE champions ADD COLUMN {col_name} {col_type}")
                print(f"Added column: {col_name}")
            except sqlite3.OperationalError as e:
                print(f"Error adding column {col_name}: {e}")
    
    conn.commit()

def get_column_mapping():
    """Map CSV column names to database column names."""
    return {
        'champion': 'name',
        'champion+H2A1:G2': 'name',  # Handle Excel artifact
        'damage_type': 'damage_type',
        'engage': 'engage_type',
        'bruiser_position': 'bruiser_position',
        'has_poke': 'has_poke',
        'has_aoe': 'has_aoe_combo',
        'has_aoe_combo': 'has_aoe_combo',
        'has_wave_clear': 'has_wave_clear',
        'is_team_fight': 'is_team_fight',
        'is_split_pusher': 'is_split_pusher',
        'is_scaling': 'is_scaling',
        'peel': 'has_peel',
        'has_peel': 'has_peel',
        'is_hypercarry': 'is_hypercarry',
        'global': 'global_ability',
        'global_ability': 'global_ability',
        # Role columns
        'is_top': 'is_top',
        'is_jungle': 'is_jungle',
        'is_mid': 'is_mid',
        'is_adc': 'is_adc',
        'is_support': 'is_support',
        'is_ranged': 'is_ranged'
    }

def convert_value(value):
    """Convert CSV value to appropriate database value."""
    if value is None or value == '' or str(value).upper() == 'NULL':
        return None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return value

def import_csv_data():
    """Import or update data from CSV file."""
    conn = sqlite3.connect(DB_PATH)
    
    # Add role columns if needed
    add_role_columns_if_needed(conn)
    
    cursor = conn.cursor()
    column_mapping = get_column_mapping()
    
    # Get all database columns
    cursor.execute("PRAGMA table_info(champions)")
    db_columns = [row[1] for row in cursor.fetchall()]
    
    with open(CSV_PATH, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        # Get CSV column names
        csv_columns = csv_reader.fieldnames
        print(f"CSV columns found: {csv_columns}")
        
        imported_count = 0
        updated_count = 0
        
        for row in csv_reader:
            # Get champion name (handle different column name variations)
            champion_name = None
            for key in row.keys():
                if 'champion' in key.lower():
                    champion_name = row[key]
                    break
            
            if not champion_name or not champion_name.strip():
                continue
            
            champion_name = champion_name.strip()
            
            # Build data dictionary
            data = {}
            for csv_col, db_col in column_mapping.items():
                if csv_col in row and db_col in db_columns:
                    value = convert_value(row[csv_col])
                    data[db_col] = value
            
            # Check if champion exists
            cursor.execute("SELECT name FROM champions WHERE LOWER(name) = LOWER(?)", (champion_name,))
            exists = cursor.fetchone()
            
            if exists:
                # Update existing record
                set_clause = ", ".join([f"{col} = ?" for col in data.keys()])
                values = list(data.values()) + [champion_name]
                cursor.execute(f"""
                    UPDATE champions 
                    SET {set_clause}
                    WHERE LOWER(name) = LOWER(?)
                """, values)
                updated_count += 1
            else:
                # Insert new record
                columns = ", ".join(data.keys())
                placeholders = ", ".join(["?" for _ in data])
                values = list(data.values())
                cursor.execute(f"""
                    INSERT INTO champions ({columns})
                    VALUES ({placeholders})
                """, values)
                imported_count += 1
        
        conn.commit()
        
        print(f"\nImport complete!")
        print(f"  New records: {imported_count}")
        print(f"  Updated records: {updated_count}")
        print(f"  Total processed: {imported_count + updated_count}")
        
        # Verify import
        cursor.execute("SELECT COUNT(*) FROM champions")
        total = cursor.fetchone()[0]
        print(f"  Total champions in database: {total}")
        
        # Show sample with role columns
        cursor.execute("""
            SELECT name, is_top, is_jungle, is_mid, is_adc, is_support, is_ranged 
            FROM champions 
            WHERE is_top = 1 OR is_jungle = 1 OR is_mid = 1 OR is_adc = 1 OR is_support = 1
            LIMIT 5
        """)
        samples = cursor.fetchall()
        if samples:
            print("\nSample champions with role data:")
            for sample in samples:
                roles = []
                if sample[1]: roles.append("Top")
                if sample[2]: roles.append("Jungle")
                if sample[3]: roles.append("Mid")
                if sample[4]: roles.append("ADC")
                if sample[5]: roles.append("Support")
                print(f"  {sample[0]}: {', '.join(roles) if roles else 'No roles'} (Ranged: {bool(sample[6])})")
    
    conn.close()

if __name__ == "__main__":
    import_csv_data()


