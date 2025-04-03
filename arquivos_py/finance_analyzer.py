import os
import sqlite3
import re
from datetime import datetime
import pandas as pd
import questionary
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pdfplumber
import ofxparse
import time
from contextlib import contextmanager

# Configurações
torch.manual_seed(42)

@contextmanager
def sqlite_db_connection(db_path):
    """Gerencia conexões com o banco de dados de forma segura"""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    finally:
        if conn:
            conn.close()

class FinanceManager:
    def __init__(self, db_path='finances.db', model_path='bert_model'):
        self.db_path = db_path
        self.model_path = model_path
        self._ensure_db_integrity()
        self.categories = self._load_categories()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.categories)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()

    def _ensure_db_integrity(self):
        """Garante que o banco tenha a estrutura correta"""
        with sqlite_db_connection(self.db_path) as conn:
            # Cria tabelas se não existirem
            conn.execute('''
                CREATE TABLE IF NOT EXISTS categories (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE
                )''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY,
                    date TEXT,
                    description TEXT,
                    value REAL,
                    category_id INTEGER,
                    account TEXT,
                    FOREIGN KEY(category_id) REFERENCES categories(id)
                )''')
            
            # Insere categorias padrão
            default_cats = ["Alimentação", "Transporte", "Moradia", "Lazer", "Saúde", "Educação", "Outros"]
            for cat in default_cats:
                conn.execute('INSERT OR IGNORE INTO categories (name) VALUES (?)', (cat,))
            conn.commit()

    def _load_categories(self):
        """Carrega categorias do banco"""
        with sqlite_db_connection(self.db_path) as conn:
            cursor = conn.execute('SELECT name FROM categories')
            return [row[0] for row in cursor.fetchall()]

    def _load_model(self):
        """Carrega o modelo BERT"""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForSequenceClassification.from_pretrained(self.model_path).to(self.device)
        except:
            self.tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
            self.model = BertForSequenceClassification.from_pretrained(
                'neuralmind/bert-base-portuguese-cased',
                num_labels=len(self.categories))
            self.model.to(self.device)

    def show_menu(self):
        """Menu principal interativo"""
        while True:
            action = questionary.select(
                "O que você deseja fazer?",
                choices=[
                    {"name": "Analisar novo extrato", "value": "analyze"},
                    {"name": "Treinar modelo", "value": "train"},
                    {"name": "Gerenciar categorias", "value": "categories"},
                    {"name": "Visualizar transações", "value": "view"},
                    {"name": "Corrigir categorias", "value": "correct"},
                    {"name": "Sair", "value": "exit"}
                ]).ask()

            if action == "analyze":
                self.analyze_statement()
            elif action == "train":
                self.train_model()
            elif action == "categories":
                self.manage_categories()
            elif action == "view":
                self.view_transactions()
            elif action == "correct":
                self.correct_categories()
            elif action == "exit":
                break

    def analyze_statement(self):
        """Processa um novo extrato"""
        file_path = questionary.path("Caminho do arquivo (PDF/CSV/OFX):").ask()
        account = questionary.text("Nome da conta:").ask()
        
        if not file_path or not os.path.exists(file_path):
            print("Arquivo não encontrado!")
            return
        
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.pdf':
                transactions = self._parse_pdf(file_path)
            elif ext == '.csv':
                transactions = self._parse_csv(file_path)
            elif ext == '.ofx':
                transactions = self._parse_ofx(file_path)
            else:
                print("Formato não suportado!")
                return
        except Exception as e:
            print(f"Erro ao processar arquivo: {str(e)}")
            return
        
        if not transactions:
            print("Nenhuma transação encontrada!")
            return
        
        print(f"\n{len(transactions)} transações encontradas (amostra):")
        for t in transactions[:5]:
            print(f"{t['date']} | {t['description'][:30]:<30} | R$ {t['value']:>9.2f}")

        if questionary.confirm("Processar estas transações?").ask():
            categorized = []
            for t in transactions:
                try:
                    t['category'] = self._predict_category(t['description'])
                    t['account'] = account
                    categorized.append(t)
                except Exception as e:
                    print(f"Erro ao categorizar transação: {str(e)}")
                    continue
            
            self._save_transactions(categorized)
            print(f"\n✅ {len(categorized)} transações salvas!")

    def _parse_pdf(self, file_path):
        """Extrai transações de PDF"""
        transactions = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        # Padrão para extratos bancários (ajuste conforme necessário)
                        pattern = r"(\d{2}/\d{2})\s+([^\n]+?)\s+(-?\d{1,3}(?:\.\d{3})*(?:,\d{2}))"
                        for match in re.finditer(pattern, text):
                            date, desc, value = match.groups()
                            transactions.append({
                                'date': datetime.strptime(date, '%d/%m').strftime('%Y-%m-%d'),
                                'description': desc.strip(),
                                'value': float(value.replace('.', '').replace(',', '.'))
                            })
        except Exception as e:
            print(f"Erro ao ler PDF: {str(e)}")
        return transactions

    def _parse_csv(self, file_path):
        """Processa arquivos CSV"""
        transactions = []
        try:
            # Tenta múltiplos encodings
            for encoding in ['utf-8', 'latin1', 'utf-16']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python', on_bad_lines='warn')
                    break
                except UnicodeDecodeError:
                    continue
            
            # Normaliza nomes de colunas
            df.columns = df.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.lower().str.strip()
            
            # Mapeamento de colunas
            col_map = {
                'data': 'date',
                'valor': 'value',
                'descricao': 'description',
                'descrição': 'description',
                'historico': 'description',
                'identificador': 'id'
            }
            df = df.rename(columns={col: col_map[col] for col in col_map if col in df.columns})
            
            # Processamento
            for _, row in df.iterrows():
                try:
                    # Data
                    date_str = str(row.get('date', '')).strip()
                    date_obj = None
                    for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y']:
                        try:
                            date_obj = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    if not date_obj:
                        continue
                    
                    # Valor
                    value_str = str(row.get('value', '0')).replace('R$', '').replace(' ', '').strip()
                    value = float(value_str.replace('.', '').replace(',', '.'))
                    
                    # Descrição
                    desc = str(row.get('description', '')).strip()
                    if 'id' in df.columns:
                        desc += f" ({row['id']})"
                    
                    transactions.append({
                        'date': date_obj.strftime('%Y-%m-%d'),
                        'description': desc,
                        'value': value
                    })
                except Exception as e:
                    continue
        except Exception as e:
            print(f"Erro ao processar CSV: {str(e)}")
        return transactions

    def _parse_ofx(self, file_path):
        """Processa arquivos OFX"""
        transactions = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ofx = ofxparse.parse(f)
                for txn in ofx.account.statement.transactions:
                    transactions.append({
                        'date': txn.date.strftime('%Y-%m-%d'),
                        'description': txn.payee or '',
                        'value': float(txn.amount)
                    })
        except Exception as e:
            print(f"Erro ao processar OFX: {str(e)}")
        return transactions

    def _predict_category(self, description):
        """Prediz categoria usando BERT"""
        inputs = self.tokenizer(description, return_tensors="pt", truncation=True, max_length=64).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.categories[torch.argmax(outputs.logits).item()]

    def _save_transactions(self, transactions):
        """Salva transações no banco"""
        with sqlite_db_connection(self.db_path) as conn:
            for t in transactions:
                # Obtém ID da categoria
                cat_id = conn.execute(
                    'SELECT id FROM categories WHERE name = ?', 
                    (t['category'],)
                ).fetchone()
                
                if not cat_id:
                    # Se categoria não existir, cria nova
                    conn.execute('INSERT OR IGNORE INTO categories (name) VALUES (?)', (t['category'],))
                    cat_id = conn.execute(
                        'SELECT id FROM categories WHERE name = ?', 
                        (t['category'],)
                    ).fetchone()
                
                # Insere transação
                conn.execute('''
                    INSERT INTO transactions 
                    (date, description, value, category_id, account)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    t['date'],
                    t['description'],
                    t['value'],
                    cat_id[0],
                    t['account']
                ))
            conn.commit()

    def train_model(self):
        """Treina o modelo com dados existentes"""
        try:
            with sqlite_db_connection(self.db_path) as conn:
                df = pd.read_sql('''
                    SELECT t.description, c.name as category 
                    FROM transactions t
                    JOIN categories c ON t.category_id = c.id
                    WHERE c.name IS NOT NULL
                ''', conn)
            
            if len(df) < 20:
                print(f"⚠️  Necessário mínimo de 20 transações categorizadas (atualmente: {len(df)})")
                print("Use a opção 'Corrigir categorias' primeiro")
                return
            
            print(f"\nIniciando treinamento com {len(df)} exemplos...")
            
            # Prepara dados
            texts = df['description'].tolist()
            labels = self.label_encoder.transform(df['category'])
            encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=64)
            
            # Dataset PyTorch
            class Dataset(torch.utils.data.Dataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels = labels
                
                def __getitem__(self, idx):
                    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                    item['labels'] = torch.tensor(self.labels[idx])
                    return item
                
                def __len__(self):
                    return len(self.labels)
            
            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(
                    output_dir='./results',
                    per_device_train_batch_size=8,
                    num_train_epochs=3,
                    save_strategy='no',
                    logging_dir='./logs',
                ),
                train_dataset=Dataset(encodings, labels)
            )
            
            trainer.train()
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
            print("\n✅ Modelo treinado e salvo com sucesso!")
        
        except Exception as e:
            print(f"Erro durante o treinamento: {str(e)}")

    def manage_categories(self):
        """Gerencia categorias"""
        while True:
            action = questionary.select(
                "Gerenciar categorias:",
                choices=[
                    {"name": "Listar categorias", "value": "list"},
                    {"name": "Adicionar categoria", "value": "add"},
                    {"name": "Remover categoria", "value": "remove"},
                    {"name": "Voltar", "value": "back"}
                ]).ask()
            
            if action == "list":
                self._list_categories()
            elif action == "add":
                self._add_category()
            elif action == "remove":
                self._remove_category()
            elif action == "back":
                break

    def _list_categories(self):
        """Lista categorias existentes"""
        try:
            with sqlite_db_connection(self.db_path) as conn:
                categories = conn.execute('''
                    SELECT c.name, COUNT(t.id) 
                    FROM categories c
                    LEFT JOIN transactions t ON c.id = t.category_id
                    GROUP BY c.name
                    ORDER BY c.name
                ''').fetchall()
            
            print("\nCategorias disponíveis:")
            for name, count in categories:
                print(f"- {name} ({count} transações)")
        except Exception as e:
            print(f"Erro ao listar categorias: {str(e)}")

    def _add_category(self):
        """Adiciona nova categoria"""
        name = questionary.text("Nome da nova categoria:").ask()
        if name:
            try:
                with sqlite_db_connection(self.db_path) as conn:
                    conn.execute('INSERT OR IGNORE INTO categories (name) VALUES (?)', (name,))
                    conn.commit()
                    print(f"✅ Categoria '{name}' adicionada!")
                    self.categories.append(name)
                    self.label_encoder.fit(self.categories)
            except sqlite3.IntegrityError:
                print("❌ Esta categoria já existe!")
            except Exception as e:
                print(f"Erro ao adicionar categoria: {str(e)}")

    def _remove_category(self):
        """Remove categoria existente"""
        try:
            with sqlite_db_connection(self.db_path) as conn:
                categories = conn.execute('SELECT id, name FROM categories ORDER BY name').fetchall()
            
            if not categories:
                print("Nenhuma categoria disponível!")
                return
            
            choice = questionary.select(
                "Selecione a categoria para remover:",
                choices=[{"name": name, "value": id} for id, name in categories] + 
                        [{"name": "Cancelar", "value": None}]
            ).ask()
            
            if choice and questionary.confirm(f"Tem certeza que deseja remover esta categoria? As transações associadas ficarão sem categoria.").ask():
                with sqlite_db_connection(self.db_path) as conn:
                    conn.execute('UPDATE transactions SET category_id = NULL WHERE category_id = ?', (choice,))
                    conn.execute('DELETE FROM categories WHERE id = ?', (choice,))
                    conn.commit()
                    print("✅ Categoria removida!")
                    self.categories = self._load_categories()
        except Exception as e:
            print(f"Erro ao remover categoria: {str(e)}")

    def view_transactions(self):
        """Visualiza transações"""
        limit = questionary.text("Quantas transações exibir? (padrão: 50)", default="50").ask()
        try:
            limit = int(limit)
        except:
            limit = 50
        
        try:
            with sqlite_db_connection(self.db_path) as conn:
                df = pd.read_sql('''
                    SELECT t.date, t.description, t.value, c.name as category, t.account
                    FROM transactions t
                    LEFT JOIN categories c ON t.category_id = c.id
                    ORDER BY t.date DESC
                    LIMIT ?
                ''', conn, params=(limit,))
            
            if df.empty:
                print("Nenhuma transação encontrada!")
                return
            
            print("\nÚltimas transações:")
            print(df.to_string(index=False))
        except Exception as e:
            print(f"Erro ao visualizar transações: {str(e)}")

    def correct_categories(self):
        """Interface para correção manual"""
        try:
            with sqlite_db_connection(self.db_path) as conn:
                # Busca transações sem categoria
                uncategorized = conn.execute('''
                    SELECT t.id, t.date, t.description, t.value
                    FROM transactions t
                    WHERE t.category_id IS NULL
                    ORDER BY t.date DESC
                    LIMIT 50
                ''').fetchall()
                
                if not uncategorized:
                    print("Nenhuma transação sem categoria!")
                    return
                
                # Busca categorias disponíveis
                categories = conn.execute('''
                    SELECT id, name FROM categories ORDER BY name
                ''').fetchall()
                
                if not categories:
                    print("Nenhuma categoria disponível para classificação!")
                    return
                
                print("\nTransações sem categoria:")
                for txn_id, date, desc, value in uncategorized:
                    print(f"\nID: {txn_id} | {date} | {desc[:50]} | R$ {value:.2f}")
                    
                    choice = questionary.select(
                        "Selecione a categoria:",
                        choices=[{"name": cat, "value": id} for id, cat in categories] +
                                [{"name": "Pular", "value": None}]
                    ).ask()
                    
                    if choice:
                        conn.execute('''
                            UPDATE transactions 
                            SET category_id = ? 
                            WHERE id = ?
                        ''', (choice, txn_id))
                        conn.commit()
                        print("✅ Categoria atualizada!")
                
                print("\nCorreção concluída!")
        except Exception as e:
            print(f"Erro durante correção: {str(e)}")

if __name__ == '__main__':
    print("\n💼 FinancIA - Gestão Financeira Inteligente\n")
    
    # Configuração inicial segura
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Verifica se precisa recriar o banco
            db_needs_rebuild = False
            try:
                with sqlite_db_connection('finances.db') as conn:
                    conn.execute("SELECT 1 FROM categories LIMIT 1")
                    conn.execute("SELECT 1 FROM transactions LIMIT 1")
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                db_needs_rebuild = True
            
            if db_needs_rebuild:
                print("⚠️  Banco de dados inválido ou corrompido. Recriando...")
                try:
                    if os.path.exists('finances.db'):
                        backup_name = f'finances_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
                        # Usa os.replace que é atômico no Windows
                        os.replace('finances.db', backup_name)
                        print(f"Backup criado: {backup_name}")
                except PermissionError:
                    if attempt == max_retries - 1:
                        print("❌ Não foi possível acessar o banco de dados. Feche outros programas que possam estar usando-o.")
                        exit(1)
                    time.sleep(retry_delay)
                    continue
            
            # Inicia o gerenciador
            manager = FinanceManager()
            manager.show_menu()
            break
            
        except PermissionError as e:
            if attempt == max_retries - 1:
                print(f"❌ Erro persistente ao acessar o banco de dados: {str(e)}")
                print("Por favor, feche todos os programas que possam estar usando o arquivo finances.db")
                exit(1)
            time.sleep(retry_delay)
        except Exception as e:
            print(f"❌ Erro inesperado: {str(e)}")
            exit(1)