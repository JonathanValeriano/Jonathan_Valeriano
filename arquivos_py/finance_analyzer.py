import os
import sqlite3
import re
from datetime import datetime
import pandas as pd
import pdfplumber
import ofxparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse
from typing import List, Dict, Optional, Union

def print_file_debug_info(file_path: str):
    """Exibe informações do arquivo para debug"""
    try:
        with open(file_path, 'rb') as f:
            print("\n=== Debug do Arquivo ===")
            print(f"Primeiras 3 linhas do arquivo {file_path}:")
            for _ in range(3):
                line = f.readline().decode('utf-8', errors='replace').strip()
                print(line.replace('\t', '|'))  # Mostra tabs como pipes para visualização
            print("=======================\n")
    except Exception as e:
        print(f"Não foi possível ler o arquivo para debug: {str(e)}")

class FinanceManager:
    def __init__(self, db_path='finances.db', model_path='bert_model'):
        self.db_path = db_path
        self.model_path = model_path
        self.categories = [
            "Alimentação", "Transporte", "Moradia",
            "Lazer", "Saúde", "Educação", "Investimento", "Receita", "Outros"
        ]
        self._init_db()
        self._load_model()

    def _init_db(self):
        """Inicializa o banco de dados SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY,
                    date TEXT,
                    description TEXT,
                    value REAL,
                    category_id INTEGER,
                    account TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_transactions_date 
                ON transactions(date)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_transactions_category 
                ON transactions(category)
            ''')

    def _load_model(self):
        """Carrega ou inicializa o modelo de categorização"""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.vectorizer = data['vectorizer']
            self.model = data['model']
        else:
            self.vectorizer = TfidfVectorizer(max_features=1000)
            self.model = RandomForestClassifier(n_estimators=100)
            self._train_with_examples()

    def _train_with_examples(self):
        """Treina o modelo com exemplos básicos"""
        examples = [
            ("SUPERMERCADO ABC", "Alimentação"),
            ("POSTO IPIRANGA", "Transporte"),
            ("ALUGUEL APTO", "Moradia"),
            ("CINEMA", "Lazer"),
            ("HOSPITAL", "Saúde"),
            ("FACULDADE", "Educação"),
            ("APLICAÇÃO RDB", "Investimento"),
            ("TRANSFERÊNCIA RECEBIDA", "Receita"),
            ("DONATIVO", "Outros")
        ]
        descriptions = [ex[0] for ex in examples]
        labels = [self.categories.index(ex[1]) for ex in examples]
        
        X = self.vectorizer.fit_transform(descriptions)
        self.model.fit(X, labels)
        self._save_model()

    def _save_model(self):
        """Salva o modelo atual"""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.model,
            'categories': self.categories
        }, self.model_path)

    def process_file(self, file_path: str, account_name: str) -> List[Dict]:
        """Processa um arquivo de extrato (PDF, OFX ou CSV)"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            print_file_debug_info(file_path)
        
        if ext == '.pdf':
            transactions = self._parse_pdf(file_path)
        elif ext == '.ofx':
            transactions = self._parse_ofx(file_path)
        elif ext == '.csv':
            transactions = self._parse_csv(file_path)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {ext}")
        
        categorized = []
        for t in transactions:
            t['category'] = self._categorize_transaction(t)
            t['account'] = account_name
            categorized.append(t)
        
        self._save_transactions(categorized)
        return categorized

    def _parse_pdf(self, file_path: str) -> List[Dict]:
        """Extrai transações de um PDF"""
        transactions = []
        
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                pattern = r"(\d{2}/\d{2})\s+(.*?)\s+([\d.,-]+)\s+([\d.,-]+)"
                for match in re.finditer(pattern, text):
                    date, description, value, balance = match.groups()
                    value = float(value.replace('.', '').replace(',', '.'))
                    
                    transactions.append({
                        'date': datetime.strptime(date, '%d/%m').strftime('%Y-%m-%d'),
                        'description': description.strip(),
                        'value': value
                    })
        
        return transactions

    def _parse_ofx(self, file_path: str) -> List[Dict]:
        """Extrai transações de um arquivo OFX"""
        with open(file_path) as f:
            ofx = ofxparse.parse(f)
        
        transactions = []
        for transaction in ofx.account.statement.transactions:
            transactions.append({
                'date': transaction.date.strftime('%Y-%m-%d'),
                'description': transaction.payee,
                'value': float(transaction.amount)
            })
        
        return transactions

    def _parse_csv(self, file_path: str) -> List[Dict]:
        """Processa arquivos CSV com tratamento robusto"""
        try:
            # Tenta ler com diferentes encodings e separadores
            for encoding in ['utf-8', 'latin1', 'utf-16']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python', on_bad_lines='warn')
                    break
                except UnicodeDecodeError:
                    continue
            
            # Normaliza nomes de colunas
            df.columns = df.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.lower().str.strip()
            
            # Mapeamento de colunas alternativas
            col_mapping = {
                'data': 'date',
                'valor': 'value',
                'descricao': 'description',
                'descrição': 'description',
                'historico': 'description',
                'identificador': 'id',
                'idtransacao': 'id'
            }
            
            # Renomeia colunas
            df = df.rename(columns={col: col_mapping[col] for col in col_mapping if col in df.columns})
            
            # Verifica colunas obrigatórias
            if 'date' not in df.columns or 'value' not in df.columns or 'description' not in df.columns:
                missing = [col for col in ['date', 'value', 'description'] if col not in df.columns]
                raise ValueError(f"Colunas obrigatórias faltando: {missing}")
            
            transactions = []
            date_formats = ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d']
            
            for _, row in df.iterrows():
                try:
                    # Processa data
                    date_str = str(row['date']).strip()
                    date_obj = None
                    for fmt in date_formats:
                        try:
                            date_obj = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    if not date_obj:
                        raise ValueError(f"Formato de data não reconhecido: {date_str}")
                    
                    # Processa valor
                    value_str = str(row['value']).replace('R$', '').replace(' ', '').strip()
                    value = float(value_str.replace('.', '').replace(',', '.'))
                    
                    # Descrição
                    desc = str(row['description']).strip()
                    if 'id' in df.columns:
                        desc += f" ({row['id']})"
                    
                    transactions.append({
                        'date': date_obj.strftime('%Y-%m-%d'),
                        'description': desc,
                        'value': value
                    })
                except Exception as e:
                    print(f"Erro ao processar linha: {row.to_dict()} - Erro: {str(e)}")
                    continue
            
            return transactions
        
        except Exception as e:
            print(f"\nERRO CRÍTICO ao processar CSV: {str(e)}")
            print("\nDicas para correção:")
            print("1. Verifique o encoding do arquivo (salve como UTF-8)")
            print("2. Confira os nomes das colunas na primeira linha")
            print("3. Verifique o separador utilizado (vírgula, ponto-e-vírgula ou tab)")
            print("\nExemplo de formato esperado:")
            print("Data|Valor|Descrição|Identificador")
            print("02/03/2025|100.50|Supermercado|123456")
            return []

    def _categorize_transaction(self, transaction: Dict) -> str:
        """Categoriza uma transação usando o modelo"""
        text = transaction['description']
        X = self.vectorizer.transform([text])
        pred = self.model.predict(X)
        return self.categories[pred[0]]

    def _save_transactions(self, transactions: List[Dict]):
        """Salva transações no banco de dados"""
        with sqlite3.connect(self.db_path) as conn:
            for t in transactions:
                conn.execute('''
                    INSERT INTO transactions (date, description, value, category, account)
                    VALUES (?, ?, ?, ?, ?)
                ''', (t['date'], t['description'], t['value'], t['category'], t['account']))

    def get_spending_by_category(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Retorna gastos por categoria em um período"""
        query = '''
            SELECT category, SUM(value) as total 
            FROM transactions 
            WHERE value < 0
        '''
        params = []
        
        if start_date:
            query += ' AND date >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND date <= ?'
            params.append(end_date)
        
        query += ' GROUP BY category ORDER BY total DESC'
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(query, conn, params=params if params else None)
        
        return df

    def get_transactions(self, limit: int = 100) -> pd.DataFrame:
        """Retorna as últimas transações"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql('''
                SELECT date, description, value, category, account 
                FROM transactions 
                ORDER BY date DESC 
                LIMIT ?
            ''', conn, params=(limit,))
        return df


def main():
    parser = argparse.ArgumentParser(description='Analisador de Extratos Bancários')
    parser.add_argument('file', nargs='?', help='Caminho do arquivo de extrato (PDF, OFX ou CSV)')  # Adicione nargs='?'
    parser.add_argument('--account', help='Nome da conta/correntista')  # Remova required=True
    parser.add_argument('--train', action='store_true', help='Treinar modelo com dados existentes')
    args = parser.parse_args()

    analyzer = FinanceAnalyzer()

    if args.train:
        print("Treinando modelo com transações existentes...")
        with sqlite3.connect(analyzer.db_path) as conn:
            df = pd.read_sql('SELECT description, category FROM transactions WHERE category IS NOT NULL', conn)

        if not df.empty:
            analyzer.vectorizer = TfidfVectorizer(max_features=1000)
            X = analyzer.vectorizer.fit_transform(df['description'])
            y = [analyzer.categories.index(cat) for cat in df['category']]
            analyzer.model.fit(X, y)
            analyzer._save_model()
            print(f"Modelo treinado com {len(df)} exemplos e salvo em {analyzer.model_path}")
        else:
            print("Nenhuma transação categorizada encontrada para treinamento.")
    elif args.file and args.account:
        print(f"Processando arquivo: {args.file}")
        transactions = analyzer.process_file(args.file, args.account)

        print(f"\n{len(transactions)} transações processadas e categorizadas:")
        for t in transactions[:5]:
            print(f"{t['date']} | {t['description'][:50]:<50} | {t['value']:>10.2f} | {t['category']}")

        print("\nResumo por Categoria:")
        summary = analyzer.get_spending_by_category()
        print(summary.to_string(index=False) if not summary.empty else "Nenhuma transação para exibir")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()