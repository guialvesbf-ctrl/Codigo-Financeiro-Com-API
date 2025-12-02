import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from groq import Groq
import os
import sqlite3
from contextlib import contextmanager
from io import BytesIO
import json

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="NeuralFinance Chat 3.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURAÇÃO DO BANCO DE DADOS ---
DB_PATH = "neuralfinance.db"

@contextmanager
def get_db_connection():
    """Context manager para conexões seguras com o banco"""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Inicializa o banco de dados e cria as tabelas necessárias"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Tabela de transações
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transacoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT NOT NULL,
                categoria TEXT NOT NULL,
                tipo TEXT NOT NULL,
                valor REAL NOT NULL,
                descricao TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabela de histórico de chat (opcional)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()

def load_transactions_from_db():
    """Carrega todas as transações do banco de dados"""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query(
                "SELECT data, categoria, tipo, valor, descricao FROM transacoes ORDER BY data DESC",
                conn
            )
            # Renomeia as colunas para manter compatibilidade
            df.columns = ['Data', 'Categoria', 'Tipo', 'Valor', 'Descrição']
            return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame(columns=['Data', 'Categoria', 'Tipo', 'Valor', 'Descrição'])

def save_transaction_to_db(categoria, tipo, valor, descricao, data=None):
    """Salva uma transação no banco de dados"""
    try:
        if data is None:
            data = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transacoes (data, categoria, tipo, valor, descricao)
                VALUES (?, ?, ?, ?, ?)
            """, (data, categoria, tipo, float(valor), descricao if descricao else "Sem descrição"))
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Erro ao salvar transação: {e}")
        return False

def delete_all_transactions():
    """Deleta todas as transações do banco"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transacoes")
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Erro ao limpar dados: {e}")
        return False

def get_transaction_stats():
    """Retorna estatísticas das transações"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    tipo,
                    COUNT(*) as total,
                    SUM(valor) as soma,
                    AVG(valor) as media
                FROM transacoes
                GROUP BY tipo
            """)
            return cursor.fetchall()
    except Exception as e:
        return []

# Inicializa o banco ao carregar o app
init_database()

# --- CONFIGURAÇÃO DA CHAVE GROQ ---
# Tenta primeiro pegar da variável de ambiente, senão usa a chave direta
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_MODEL = "llama-3.1-8b-instant"

# Opção para configurar a chave via interface (mais seguro para produção)
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = bool(GROQ_API_KEY)

# --- CSS FUTURISTA MELHORADO ---
st.markdown("""
    <style>
        /* Background e cores principais */
        .stApp { 
            background: linear-gradient(135deg, #0a0a0a 0%, #1a0a2e 100%);
            color: #e0e0e0; 
        }
        
        [data-testid="stSidebar"] { 
            background: linear-gradient(180deg, #0f0f1e 0%, #1a0a2e 100%);
            border-right: 2px solid #00ff88;
        }
        
        /* Mensagens do Chat */
        .stChatMessage { 
            background: rgba(17, 17, 34, 0.8);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            margin: 10px 0;
        }
        
        /* Conteúdo das mensagens do chat */
        .stChatMessage p,
        .stChatMessage span,
        .stChatMessage div {
            color: #e0e0e0 !important;
            font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
            font-size: 15px !important;
            line-height: 1.6 !important;
        }
        
        /* Texto dentro das mensagens */
        [data-testid="stChatMessageContent"] {
            color: #e0e0e0 !important;
        }
        
        [data-testid="stChatMessageAvatarUser"] { 
            background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        }
        
        [data-testid="stChatMessageAvatarAssistant"] { 
            background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        }
        
        /* Markdown dentro do chat */
        .stChatMessage h1,
        .stChatMessage h2,
        .stChatMessage h3,
        .stChatMessage h4 {
            color: #00ff88 !important;
            margin-top: 10px !important;
            margin-bottom: 5px !important;
        }
        
        .stChatMessage strong {
            color: #00d4ff !important;
            font-weight: 600 !important;
        }
        
        .stChatMessage code {
            background-color: rgba(0, 255, 136, 0.1) !important;
            color: #00ff88 !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-family: 'Courier New', monospace !important;
        }
        
        .stChatMessage ul,
        .stChatMessage ol {
            color: #e0e0e0 !important;
            padding-left: 20px !important;
        }
        
        .stChatMessage li {
            color: #e0e0e0 !important;
            margin: 5px 0 !important;
        }
        
        /* Input de chat */
        .stChatInputContainer {
            border-top: 1px solid rgba(0, 255, 136, 0.3);
            background: rgba(17, 17, 34, 0.8);
        }
        
        .stChatInput textarea {
            color: #e0e0e0 !important;
            background-color: rgba(17, 17, 34, 0.9) !important;
            border: 1px solid #00ff88 !important;
            border-radius: 8px !important;
            font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
        }
        
        .stChatInput textarea::placeholder {
            color: #888 !important;
        }
        .stTextInput>div>div>input, 
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>select { 
            color: #fff;
            background-color: rgba(17, 17, 34, 0.8);
            border: 1px solid #00ff88;
            border-radius: 8px;
        }
        
        .stButton>button {
            background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%);
            color: #000;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background: linear-gradient(90deg, #00cc6a 0%, #00ff88 100%);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 255, 136, 0.4);
        }
        
        /* Botões primários e secundários */
        button[kind="primary"] {
            background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%) !important;
            color: #000 !important;
        }
        
        button[kind="secondary"] {
            background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%) !important;
            color: #fff !important;
        }
        
        /* Estilo do modal/formulário */
        [data-testid="stForm"] {
            background: rgba(17, 17, 34, 0.9);
            border: 2px solid rgba(0, 255, 136, 0.5);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        
        /* Text area */
        .stTextArea textarea {
            color: #e0e0e0 !important;
            background-color: rgba(17, 17, 34, 0.8) !important;
            border: 1px solid #00ff88 !important;
            border-radius: 8px !important;
        }
        
        /* Métricas */
        [data-testid="stMetricValue"] {
            font-size: 28px;
            font-weight: bold;
            background: linear-gradient(90deg, #00ff88, #00d4ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Títulos */
        h1, h2, h3 {
            background: linear-gradient(90deg, #00ff88, #00d4ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        
        /* Tabelas */
        .dataframe {
            background-color: rgba(17, 17, 34, 0.8);
            border: 1px solid rgba(0, 255, 136, 0.3);
        }
    </style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO DE ESTADO ---
# Carrega dados do banco de dados
if 'data' not in st.session_state:
    st.session_state.data = load_transactions_from_db()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Estados para controlar os modais
if 'show_entrada_modal' not in st.session_state:
    st.session_state.show_entrada_modal = False

if 'show_saida_modal' not in st.session_state:
    st.session_state.show_saida_modal = False

# Flag para recarregar dados do banco
if 'reload_data' not in st.session_state:
    st.session_state.reload_data = False

# --- FUNÇÕES MELHORADAS ---
def add_transaction(data, categoria, tipo, valor, descricao):
    """Adiciona uma nova transação ao DataFrame e ao banco de dados"""
    # Salva no banco primeiro
    data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if save_transaction_to_db(categoria, tipo, valor, descricao, data_hora):
        # Adiciona ao DataFrame em memória
        new_row = pd.DataFrame({
            'Data': [data_hora],
            'Categoria': [categoria],
            'Tipo': [tipo],
            'Valor': [float(valor)],
            'Descrição': [descricao if descricao else "Sem descrição"]
        })
        return pd.concat([data, new_row], ignore_index=True)
    return data

def get_financial_summary(df):
    """Gera um resumo financeiro estruturado"""
    if df.empty:
        return "Nenhuma transação registrada ainda."
    
    entrada = df[df['Tipo'] == 'Entrada']['Valor'].sum()
    saida = df[df['Tipo'] == 'Saída']['Valor'].sum()
    saldo = entrada - saida
    
    # Análise por categoria
    gastos_categoria = df[df['Tipo'] == 'Saída'].groupby('Categoria')['Valor'].sum().sort_values(ascending=False)
    
    summary = f"""
📊 RESUMO FINANCEIRO:
- Total de Entradas: R$ {entrada:,.2f}
- Total de Saídas: R$ {saida:,.2f}
- Saldo Atual: R$ {saldo:,.2f}
- Total de Transações: {len(df)}

💸 TOP 3 CATEGORIAS DE GASTOS:
"""
    for i, (cat, val) in enumerate(gastos_categoria.head(3).items(), 1):
        summary += f"{i}. {cat}: R$ {val:,.2f}\n"
    
    return summary

def get_groq_chat_response(messages, df):
    """Obtém resposta do modelo Groq com contexto financeiro"""
    try:
        if not GROQ_API_KEY or GROQ_API_KEY == "":
            return "⚠️ **Erro de Configuração**: A chave da API Groq não está configurada.\n\n💡 **Dica**: Adicione sua chave diretamente no código ou use variável de ambiente."
        
        client = Groq(api_key=GROQ_API_KEY)
        
        # Contexto financeiro estruturado
        financial_context = get_financial_summary(df)
        csv_data = df.to_csv(index=False) if not df.empty else "Sem dados"
        
        system_context = {
            "role": "system",
            "content": f"""Você é o NeuralFinance AI, um consultor financeiro avançado e empático.

DIRETRIZES:
- Responda SEMPRE em Português do Brasil
- Use Markdown para formatação elegante
- Seja direto, prático e motivacional
- Forneça insights acionáveis baseados nos dados
- Use emojis relevantes para melhor visualização
- Quando apropriado, sugira melhorias financeiras

IMPORTANTE - DETECÇÃO DE SOLICITAÇÃO DE PLANILHA:
- Se o usuário pedir "planilha", "orçamento", "relatório", "análise detalhada", "excel" ou "exportar", 
  você DEVE mencionar explicitamente na resposta: "Preparando planilha de [tipo]..."
- Quando detectar essas palavras-chave, inclua no início da resposta a frase exata: 
  "📊 EXPORTAR_PLANILHA: [tipo_da_planilha]"
- Tipos possíveis: "orcamento", "transacoes", "analise_completa"

DADOS FINANCEIROS DO USUÁRIO:
{financial_context}

DADOS DETALHADOS (CSV):
{csv_data}

Base todas as suas análises e recomendações nesses dados reais."""
        }
        
        # Combina contexto + histórico
        final_messages = [system_context] + messages
        
        chat_completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=final_messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"❌ **Erro ao conectar com Groq API**: {str(e)}\n\n💡 Verifique sua chave de API e conexão com a internet."

def create_excel_export(df, filename="transacoes.xlsx"):
    """Cria arquivo Excel para download (com fallback para CSV)"""
    if df.empty:
        st.warning("⚠️ Nenhuma transação para exportar!")
        # Retorna CSV vazio se não houver dados
        output = BytesIO(b"Data,Categoria,Tipo,Valor,Descricao\n")
        return output, 'csv'
    
    output = BytesIO()
    try:
        import openpyxl
        # Tenta usar openpyxl
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Garante que o DataFrame tem dados
            df_export = df.copy()
            df_export.to_excel(writer, index=False, sheet_name='Transações')
            
            # Log de debug
            st.info(f"✅ Excel gerado com {len(df_export)} linhas")
        
        output.seek(0)
        return output, 'xlsx'
    except ImportError:
        st.warning("⚠️ openpyxl não instalado. Exportando como CSV.")
        # Fallback para CSV se openpyxl não estiver instalado
        csv_data = df.to_csv(index=False)
        output = BytesIO(csv_data.encode('utf-8-sig'))
        output.seek(0)
        return output, 'csv'
    except Exception as e:
        st.error(f"❌ Erro ao gerar Excel: {e}")
        # Fallback em caso de erro
        csv_data = df.to_csv(index=False)
        output = BytesIO(csv_data.encode('utf-8-sig'))
        output.seek(0)
        return output, 'csv'

def create_budget_spreadsheet(df):
    """Cria planilha de orçamento detalhada (com fallback para CSV)"""
    if df.empty:
        output = BytesIO(b"Nenhuma transacao registrada ainda.\n")
        return output, 'csv'
    
    output = BytesIO()
    
    try:
        # Tenta criar Excel com múltiplas abas
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Aba 1: Todas as Transações
            df_sorted = df.sort_values('Data', ascending=False)
            df_sorted.to_excel(writer, sheet_name='Transações', index=False)
            
            # Aba 2: Resumo por Categoria
            if len(df) > 0:
                resumo_categoria = df.groupby(['Tipo', 'Categoria']).agg({
                    'Valor': ['sum', 'mean', 'count']
                }).round(2)
                resumo_categoria.columns = ['Total', 'Média', 'Quantidade']
                resumo_categoria.to_excel(writer, sheet_name='Resumo Categorias')
            
            # Aba 3: Resumo Mensal
            df_copy = df.copy()
            df_copy['Data'] = pd.to_datetime(df_copy['Data'], errors='coerce')
            df_copy = df_copy.dropna(subset=['Data'])
            if len(df_copy) > 0:
                df_copy['Mês'] = df_copy['Data'].dt.to_period('M').astype(str)
                resumo_mensal = df_copy.groupby(['Mês', 'Tipo']).agg({
                    'Valor': 'sum'
                }).round(2)
                resumo_mensal.to_excel(writer, sheet_name='Resumo Mensal')
            
            # Aba 4: Dashboard de Indicadores
            entrada_total = df[df['Tipo'] == 'Entrada']['Valor'].sum()
            saida_total = df[df['Tipo'] == 'Saída']['Valor'].sum()
            saldo = entrada_total - saida_total
            
            indicadores = pd.DataFrame({
                'Indicador': ['Total Entradas', 'Total Saídas', 'Saldo', 'Taxa de Poupança', 'Total Transações'],
                'Valor': [
                    f'R$ {entrada_total:,.2f}',
                    f'R$ {saida_total:,.2f}',
                    f'R$ {saldo:,.2f}',
                    f'{(saldo/entrada_total*100) if entrada_total > 0 else 0:.1f}%',
                    len(df)
                ]
            })
            indicadores.to_excel(writer, sheet_name='Indicadores', index=False)
            
            # Aba 5: Top Gastos
            if len(df[df['Tipo'] == 'Saída']) > 0:
                top_gastos = df[df['Tipo'] == 'Saída'].nlargest(min(20, len(df[df['Tipo'] == 'Saída'])), 'Valor')[['Data', 'Categoria', 'Valor', 'Descrição']]
                top_gastos.to_excel(writer, sheet_name='Top 20 Gastos', index=False)
        
        output.seek(0)
        return output, 'xlsx'
    
    except ImportError:
        # Fallback: Cria um relatório TXT completo com todos os dados e análises
        entrada_total = df[df['Tipo'] == 'Entrada']['Valor'].sum()
        saida_total = df[df['Tipo'] == 'Saída']['Valor'].sum()
        saldo = entrada_total - saida_total
        
        report = f"""╔══════════════════════════════════════════════════════════════╗
║        RELATÓRIO FINANCEIRO NEURALFINANCE                   ║
║        Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}                                    ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║                  INDICADORES PRINCIPAIS                      ║
╚══════════════════════════════════════════════════════════════╝

💰 Total Entradas:     R$ {entrada_total:>15,.2f}
💸 Total Saídas:       R$ {saida_total:>15,.2f}
📊 Saldo Atual:        R$ {saldo:>15,.2f}
📈 Taxa de Poupança:   {(saldo/entrada_total*100) if entrada_total > 0 else 0:>14.1f}%
🔢 Total Transações:   {len(df):>18}

╔══════════════════════════════════════════════════════════════╗
║                   RESUMO POR CATEGORIA                       ║
╚══════════════════════════════════════════════════════════════╝

"""
        if len(df) > 0:
            resumo = df.groupby(['Tipo', 'Categoria'])['Valor'].agg(['sum', 'mean', 'count']).round(2)
            resumo.columns = ['Total (R$)', 'Média (R$)', 'Quantidade']
            report += resumo.to_string() + "\n\n"
        
        report += """╔══════════════════════════════════════════════════════════════╗
║                  TODAS AS TRANSAÇÕES                         ║
╚══════════════════════════════════════════════════════════════╝

"""
        report += df.sort_values('Data', ascending=False).to_string(index=False)
        
        output = BytesIO(report.encode('utf-8-sig'))
        output.seek(0)
        return output, 'txt'

def parse_ai_request_for_export(user_message):
    """Detecta se o usuário pediu uma planilha/exportação"""
    keywords = [
        'planilha', 'excel', 'exportar', 'download', 'baixar',
        'orçamento', 'relatório', 'análise detalhada', 'spreadsheet'
    ]
    return any(keyword in user_message.lower() for keyword in keywords)
    """Cria gráfico de linha temporal"""
    fig = go.Figure()
    
    for tipo in df['Tipo'].unique():
        df_tipo = df[df['Tipo'] == tipo].sort_values('Data')
        fig.add_trace(go.Scatter(
            x=df_tipo['Data'],
            y=df_tipo['Valor'],
            mode='lines+markers',
            name=tipo,
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="📈 Timeline de Transações",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,17,34,0.8)',
        font=dict(color='#e0e0e0'),
        hovermode='x unified'
    )
    
    return fig

def create_timeline_chart(df):
    """Cria gráfico de pizza de gastos"""
    df_gastos = df[df['Tipo'] == 'Saída']
    
    fig = px.pie(
        df_gastos,
        values='Valor',
        names='Categoria',
        hole=0.6,
        template='plotly_dark',
        title="💰 Distribuição de Gastos"
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0')
    )
    
    return fig

# --- SIDEBAR (INPUT DE DADOS) ---
st.sidebar.title("🎯 NOVA TRANSAÇÃO")

# Teste de biblioteca Excel
with st.sidebar.expander("🔍 Status do Sistema", expanded=False):
    try:
        import openpyxl
        st.success("✅ openpyxl instalado - Excel disponível")
        st.caption(f"Versão: {openpyxl.__version__}")
    except ImportError:
        st.warning("⚠️ openpyxl não instalado")
        st.caption("Instale com: `pip install openpyxl`")
    
    st.markdown("---")
    st.info(f"📊 Transações no banco: **{len(st.session_state.data)}**")
    st.info(f"💬 Mensagens no chat: **{len(st.session_state.messages)}**")

# Configuração da API (opcional na sidebar)
with st.sidebar.expander("⚙️ Configuração API", expanded=False):
    custom_key = st.text_input(
        "Chave Groq API (opcional)",
        value="",
        type="password",
        help="Cole sua chave API aqui se quiser substituir a padrão"
    )
    if custom_key:
        GROQ_API_KEY = custom_key
        st.success("✅ Chave personalizada configurada!")

# Informações do banco de dados
with st.sidebar.expander("📊 Estatísticas do Banco", expanded=False):
    stats = get_transaction_stats()
    if stats:
        st.markdown("**📈 Resumo Geral:**")
        for tipo, total, soma, media in stats:
            st.markdown(f"**{tipo}s:**")
            st.markdown(f"- Total: {total} transações")
            st.markdown(f"- Soma: R$ {soma:,.2f}")
            st.markdown(f"- Média: R$ {media:,.2f}")
    else:
        st.info("Nenhum dado ainda")
    
    if st.button("🔄 Recarregar do Banco", use_container_width=True):
        st.session_state.data = load_transactions_from_db()
        st.success("✅ Dados recarregados!")
        st.rerun()

st.sidebar.markdown("---")

with st.sidebar.form("entry_form", clear_on_submit=True):
    tipo = st.selectbox("💳 Tipo", ["Saída", "Entrada"], index=0)
    
    categorias_saida = ["Moradia", "Alimentação", "Transporte", "Saúde", "Lazer", "Educação", "Investimentos", "Outros"]
    categorias_entrada = ["Salário", "Freelance", "Investimentos", "Presente", "Outros"]
    
    categoria = st.selectbox(
        "📁 Categoria",
        categorias_saida if tipo == "Saída" else categorias_entrada
    )
    
    valor = st.number_input("💵 Valor (R$)", min_value=0.01, value=100.00, format="%.2f")
    desc = st.text_input("📝 Descrição (opcional)")
    
    submitted = st.form_submit_button("✨ PROCESSAR TRANSAÇÃO", use_container_width=True)
    
    if submitted:
        st.session_state.data = add_transaction(
            st.session_state.data, categoria, tipo, valor, desc
        )
        st.success("✅ Transação adicionada com sucesso!")
        st.balloons()

# Botões de ação adicionais
st.sidebar.markdown("---")
col_a, col_b = st.sidebar.columns(2)

with col_a:
    if st.button("🗑️ Limpar Dados", use_container_width=True):
        if delete_all_transactions():
            st.session_state.data = pd.DataFrame(columns=['Data', 'Categoria', 'Tipo', 'Valor', 'Descrição'])
            st.success("✅ Todos os dados foram deletados do banco!")
            st.rerun()

with col_b:
    if st.button("💬 Limpar Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- DASHBOARD PRINCIPAL ---
st.title("🧠 NEURAL FINANCE CHAT 3.0")
st.markdown("### *Inteligência Artificial para suas Finanças*")

# Botões de ação rápida no topo
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
with col_btn1:
    if st.button("➕ Nova Entrada", use_container_width=True, type="primary"):
        st.session_state.show_entrada_modal = True
        st.rerun()

with col_btn2:
    if st.button("➖ Nova Saída", use_container_width=True, type="secondary"):
        st.session_state.show_saida_modal = True
        st.rerun()

st.markdown("---")

df = st.session_state.data

# --- MODAIS DE ENTRADA E SAÍDA ---
# Modal para ENTRADA
if st.session_state.show_entrada_modal:
    st.markdown("### 💰 Adicionar Nova Entrada")
    with st.form("entrada_modal_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            categoria_entrada = st.selectbox(
                "📁 Categoria",
                ["Salário", "Freelance", "Investimentos", "Presente", "Bônus", "Outros"],
                key="cat_entrada"
            )
            valor_entrada = st.number_input("💵 Valor (R$)", min_value=0.01, value=1000.00, format="%.2f", key="val_entrada")
        
        with col2:
            desc_entrada = st.text_area("📝 Descrição (opcional)", height=100, key="desc_entrada")
        
        col_submit, col_cancel = st.columns(2)
        
        with col_submit:
            if st.form_submit_button("✅ Confirmar Entrada", use_container_width=True):
                st.session_state.data = add_transaction(
                    st.session_state.data, categoria_entrada, "Entrada", valor_entrada, desc_entrada
                )
                st.session_state.show_entrada_modal = False
                st.success(f"✅ Entrada de R$ {valor_entrada:,.2f} adicionada!")
                st.balloons()
                st.rerun()
        
        with col_cancel:
            if st.form_submit_button("❌ Cancelar", use_container_width=True):
                st.session_state.show_entrada_modal = False
                st.rerun()
    
    st.markdown("---")

# Modal para SAÍDA
if st.session_state.show_saida_modal:
    st.markdown("### 💸 Adicionar Nova Saída")
    with st.form("saida_modal_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            categoria_saida = st.selectbox(
                "📁 Categoria",
                ["Moradia", "Alimentação", "Transporte", "Saúde", "Lazer", "Educação", "Investimentos", "Contas", "Outros"],
                key="cat_saida"
            )
            valor_saida = st.number_input("💵 Valor (R$)", min_value=0.01, value=100.00, format="%.2f", key="val_saida")
        
        with col2:
            desc_saida = st.text_area("📝 Descrição (opcional)", height=100, key="desc_saida")
        
        col_submit, col_cancel = st.columns(2)
        
        with col_submit:
            if st.form_submit_button("✅ Confirmar Saída", use_container_width=True):
                st.session_state.data = add_transaction(
                    st.session_state.data, categoria_saida, "Saída", valor_saida, desc_saida
                )
                st.session_state.show_saida_modal = False
                st.success(f"✅ Saída de R$ {valor_saida:,.2f} adicionada!")
                st.rerun()
        
        with col_cancel:
            if st.form_submit_button("❌ Cancelar", use_container_width=True):
                st.session_state.show_saida_modal = False
                st.rerun()
    
    st.markdown("---")

# KPIs e Visualizações
if not df.empty:
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    entrada = df[df['Tipo'] == 'Entrada']['Valor'].sum()
    saida = df[df['Tipo'] == 'Saída']['Valor'].sum()
    saldo = entrada - saida
    total_trans = len(df)
    
    col1.metric("💰 Total Entradas", f"R$ {entrada:,.2f}")
    col2.metric("💸 Total Saídas", f"R$ {saida:,.2f}")
    col3.metric("📊 Saldo Atual", f"R$ {saldo:,.2f}", delta=f"{((saldo/entrada)*100) if entrada > 0 else 0:.1f}%")
    col4.metric("📈 Transações", total_trans)
    
    st.markdown("---")
    
    # Gráficos
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.plotly_chart(create_timeline_chart(df), use_container_width=True)
    
    with chart_col2:
        if not df[df['Tipo'] == 'Saída'].empty:
            st.plotly_chart(create_expenses_chart(df), use_container_width=True)
        else:
            st.info("📊 Adicione gastos para ver a distribuição")
    
    # Tabela de transações recentes
    with st.expander("📋 Ver Todas as Transações", expanded=False):
        st.dataframe(
            df.sort_values('Data', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Botão de exportação rápida
        st.markdown("**📥 Exportar Dados:**")
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            excel_data, file_type = create_excel_export(df)
            file_ext = 'xlsx' if file_type == 'xlsx' else 'csv'
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if file_type == 'xlsx' else "text/csv"
            
            st.download_button(
                label=f"📥 Transações ({file_ext.upper()})",
                data=excel_data,
                file_name=f"neuralfinance_transacoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                mime=mime_type,
                use_container_width=True,
                key="export_transactions"
            )
        
        with col_exp2:
            csv_data = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📄 CSV Simples",
                data=csv_data,
                file_name=f"transacoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="export_csv"
            )
        
        with col_exp3:
            excel_budget, file_type_b = create_budget_spreadsheet(df)
            file_ext_b = 'xlsx' if file_type_b == 'xlsx' else 'txt'
            mime_type_b = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if file_type_b == 'xlsx' else "text/plain"
            
            st.download_button(
                label=f"💼 Orçamento ({file_ext_b.upper()})",
                data=excel_budget,
                file_name=f"neuralfinance_orcamento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext_b}",
                mime=mime_type_b,
                use_container_width=True,
                key="export_budget"
            )
        
        if file_type == 'csv' or file_type_b != 'xlsx':
            st.caption("ℹ️ Para exportar em Excel (.xlsx), instale: `pip install openpyxl`")
else:
    st.info("💡 **Comece agora**: Adicione sua primeira transação na barra lateral para visualizar seus dados financeiros!")

st.markdown("---")

# --- ÁREA DE CHAT INTERATIVO ---
st.subheader("💬 Consultoria Neural Interativa")

# Mensagem inicial se o chat estiver vazio
if not st.session_state.messages and not df.empty:
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"👋 Olá! Sou o **NeuralFinance AI**, seu consultor financeiro inteligente.\n\nAnalisei seus dados e vejo que você tem:\n- **{len(df)} transações** registradas\n- **Saldo atual**: R$ {df[df['Tipo'] == 'Entrada']['Valor'].sum() - df[df['Tipo'] == 'Saída']['Valor'].sum():,.2f}\n\nComo posso ajudar você hoje? 💰"
    })

# Exibir histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Campo de entrada do usuário
if prompt := st.chat_input("💭 Pergunte sobre suas finanças, peça análises ou dicas..."):
    # Adiciona mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Gera resposta da IA
    with st.chat_message("assistant"):
        with st.spinner("🧠 Processando análise neural..."):
            response = get_groq_chat_response(st.session_state.messages, df)
            st.markdown(response)
            
            # Verifica se deve exportar planilha
            if parse_ai_request_for_export(prompt) or "EXPORTAR_PLANILHA" in response:
                st.markdown("---")
                st.markdown("### 📥 Download da Planilha")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Planilha simples de transações
                    excel_simple, file_type1 = create_excel_export(df, "transacoes.xlsx")
                    file_ext1 = 'xlsx' if file_type1 == 'xlsx' else 'csv'
                    mime_type1 = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if file_type1 == 'xlsx' else "text/csv"
                    
                    st.download_button(
                        label=f"📊 Baixar Transações ({file_ext1.upper()})",
                        data=excel_simple,
                        file_name=f"neuralfinance_transacoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext1}",
                        mime=mime_type1,
                        use_container_width=True
                    )
                
                with col2:
                    # Planilha completa de orçamento
                    if not df.empty:
                        excel_budget, file_type2 = create_budget_spreadsheet(df)
                        file_ext2 = 'xlsx' if file_type2 == 'xlsx' else 'csv'
                        mime_type2 = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if file_type2 == 'xlsx' else "text/csv"
                        
                        st.download_button(
                            label=f"💼 Baixar Orçamento Completo ({file_ext2.upper()})",
                            data=excel_budget,
                            file_name=f"neuralfinance_orcamento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext2}",
                            mime=mime_type2,
                            use_container_width=True
                        )
                
                if file_type1 == 'csv' or file_type2 == 'csv':
                    st.info("ℹ️ **Nota**: Biblioteca Excel não instalada. Arquivos exportados em formato CSV. Para Excel, instale: `pip install openpyxl`")
                
                st.success("✅ Planilhas prontas para download!")
    
    # Adiciona resposta ao histórico
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sugestões de perguntas
if df.empty:
    st.info("💡 **Dica**: Adicione transações na barra lateral para começar a conversar sobre suas finanças!")
else:
    st.markdown("#### 🎯 Sugestões de Perguntas:")
    col1, col2, col3 = st.columns(3)
    
    suggestions = [
        "Analise meus gastos",
        "Crie uma planilha de orçamento",
        "Qual meu maior gasto?"
    ]
    
    for col, suggestion in zip([col1, col2, col3], suggestions):
        if col.button(f"💭 {suggestion}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": suggestion})
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>🧠 NeuralFinance Chat 3.0 | Powered by Groq AI</div>",
    unsafe_allow_html=True
)