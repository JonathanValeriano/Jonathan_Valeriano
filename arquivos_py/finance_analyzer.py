import os
import sqlite3
from datetime import datetime
import pandas as pd
import questionary  # Biblioteca para interfaces CLI amigáveis
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Configurações iniciais
torch.manual_seed(42)

class FinanceManager:
    def __init__(self, db_path='finances.db', model_path='bert_model'):
        self.db_path = db_path
        self.model_path = model_path
        self.categories = self._load_categories()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.categories)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._init_db()
        self._load_model()

    def _init_db(self):
        """Inicializa o banco de dados"""
        with sqlite3.connect(self.db_path) as conn:
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
            
            # Insere categorias padrão se não existirem
            for cat in self.categories:
                conn.execute('INSERT OR IGNORE INTO categories (name) VALUES (?)', (cat,))

    def _load_categories(self) -> list:
        """Carrega categorias do banco de dados"""
        if not os.path.exists(self.db_path):
            return ["Alimentação", "Transporte", "Moradia", "Lazer", "Saúde", "Educação", "Outros"]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT name FROM categories')
            return [row[0] for row in cursor.fetchall()] or ["Alimentação", "Transporte", "Moradia", "Lazer", "Saúde", "Educação", "Outros"]

    def _load_model(self):
        """Carrega o modelo BERT"""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForSequenceClassification.from_pretrained(self.model_path).to(self.device)
        except:
            self.tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
            self.model = BertForSequenceClassification.from_pretrained(
                'neuralmind/bert-base-portuguese-cased',
                num_labels=len(self.categories)
            self.model.to(self.device)

    def show_menu(self):
        """Exibe o menu principal"""
        while True:
            action = questionary.select(
                "O que você deseja fazer?",
                choices=[
                    {"name": "Analisar novo extrato", "value": "analyze"},
                    {"name": "Treinar modelo", "value": "train"},
                    {"name": "Adicionar/Editar categorias", "value": "categories"},
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
        
        if not os.path.exists(file_path):
            print("Arquivo não encontrado!")
            return
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.pdf', '.csv', '.ofx']:
            print("Formato não suportado!")
            return
        
        transactions = self._parse_file(file_path)
        if not transactions:
            print("Nenhuma transação encontrada!")
            return
        
        print(f"\n{len(transactions)} transações encontradas:")
        for t in transactions[:5]:
            print(f"{t['date']} | {t['description'][:30]:<30} | R$ {t['value']:>9.2f}")

        if questionary.confirm("Processar estas transações?").ask():
            categorized = []
            for t in transactions:
                t['category'] = self._predict_category(t['description'])
                t['account'] = account
                categorized.append(t)
            
            self._save_transactions(categorized)
            print(f"\n✅ {len(categorized)} transações salvas!")

    def train_model(self):
        """Treina o modelo com dados existentes"""
        with sqlite3.connect(self.db_path) as conn:
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
        
        dataset = Dataset(encodings, labels)
        
        # Configura treinamento
        training_args = TrainingArguments(
            output_dir='./results',
            per_device_train_batch_size=8,
            num_train_epochs=3,
            save_strategy='no',
            logging_dir='./logs',
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        
        # Executa treinamento
        trainer.train()
        
        # Salva modelo
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        print("\n✅ Modelo treinado e salvo com sucesso!")

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
        """Lista todas as categorias"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT id, name FROM categories')
            categories = cursor.fetchall()
        
        print("\nCategorias disponíveis:")
        for cat_id, name in categories:
            count = conn.execute('SELECT COUNT(*) FROM transactions WHERE category_id = ?', (cat_id,)).fetchone()[0]
            print(f"{cat_id}: {name} ({count} transações)")

    def _add_category(self):
        """Adiciona nova categoria"""
        name = questionary.text("Nome da nova categoria:").ask()
        if name:
            with sqlite3.connect(self.db_path) as conn:
                try:
                    conn.execute('INSERT INTO categories (name) VALUES (?)', (name,))
                    print(f"✅ Categoria '{name}' adicionada!")
                    self.categories.append(name)
                    self.label_encoder.fit(self.categories)
                except sqlite3.IntegrityError:
                    print("❌ Esta categoria já existe!")

    def _remove_category(self):
        """Remove categoria existente"""
        with sqlite3.connect(self.db_path) as conn:
            categories = conn.execute('SELECT id, name FROM categories').fetchall()
        
        if not categories:
            print("Nenhuma categoria disponível!")
            return
        
        choices = [{"name": name, "value": id} for id, name in categories]
        cat_id = questionary.select("Selecione a categoria para remover:", choices).ask()
        
        if questionary.confirm("Tem certeza? As transações serão marcadas como sem categoria").ask():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('UPDATE transactions SET category_id = NULL WHERE category_id = ?', (cat_id,))
                conn.execute('DELETE FROM categories WHERE id = ?', (cat_id,))
                print("✅ Categoria removida!")
                self.categories = self._load_categories()

    def view_transactions(self):
        """Visualiza transações"""
        limit = questionary.text("Quantas transações exibir? (padrão: 50)", default="50").ask()
        try:
            limit = int(limit)
        except:
            limit = 50
        
        with sqlite3.connect(self.db_path) as conn:
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

    def correct_categories(self):
        """Interface para correção manual de categorias"""
        with sqlite3.connect(self.db_path) as conn:
            uncategorized = conn.execute('''
                SELECT t.id, t.date, t.description, t.value 
                FROM transactions t
                WHERE t.category_id IS NULL
                LIMIT 100
            ''').fetchall()
            
            if not uncategorized:
                print("Nenhuma transação sem categoria!")
                return
            
            categories = conn.execute('SELECT id, name FROM categories').fetchall()
            if not categories:
                print("Nenhuma categoria definida!")
                return
            
            print("\nTransações sem categoria:")
            for txn in uncategorized[:10]:
                txn_id, date, desc, value = txn
                print(f"\nID: {txn_id} | {date} | {desc[:50]} | R$ {value:.2f}")
                
                choices = [{"name": cat, "value": id} for id, cat in categories]
                choices.append({"name": "Pular", "value": None})
                
                cat_id = questionary.select(
                    "Selecione a categoria:",
                    choices=choices
                ).ask()
                
                if cat_id:
                    conn.execute('UPDATE transactions SET category_id = ? WHERE id = ?', (cat_id, txn_id))
                    print("✅ Categoria atualizada!")
            
            print("\nCorreção concluída!")

    # ... (métodos auxiliares _parse_file, _predict_category, _save_transactions)

if __name__ == '__main__':
    print("\n💼 FinancIA - Gestão Financeira Inteligente\n")
    manager = FinanceManager()
    manager.show_menu()