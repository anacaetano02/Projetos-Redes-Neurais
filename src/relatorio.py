"""
Gerador de relatório PDF do pipeline Lending Club.

Todo número que aparece no PDF vem de `results/run_history.json` (populado via `src.experiment_tracker.log_run`)
ou dos arquivos de curva de perda por época (`results/history_class.json` /
`results/history_reg.json`). Quando um dado não existe, o relatório diz isso
explicitamente ("Dados não disponíveis") em vez de preencher com 1.0000 ou
qualquer outro valor de exemplo.

Como o histórico é acumulativo (append-only), basta rodar este script de novo a cada
vez que você alterar o código e re-treinar: o relatório automaticamente passa a
comparar a nova execução com as anteriores (mesmo model_name) e entre abordagens
diferentes (ex: baseline vs MLP) na execução mais recente de cada uma.
"""

import os
import json
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame,
    Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, Flowable, NextPageTemplate,
)

from src.experiment_tracker import load_history, latest_run_per_model, runs_for_model

# ── 1. Paleta de Cores (Design System) ─────────────────────────────────
COLORS = {
    'heading':   HexColor('#1a2b4c'),
    'body':      HexColor('#2d3748'),
    'accent':    HexColor('#ff6b35'),
    'muted':     HexColor('#718096'),
    'bg_alt':    HexColor('#f7fafc'),
    'bg_header': HexColor('#1a2b4c'),
    'white':     HexColor('#ffffff'),
    'warn':      HexColor('#c53030'),   # vermelho para alertas de dados insuficientes
    'good':      HexColor('#2f855a'),   # verde para melhoria confirmada
}

HEADING_FONT = 'Helvetica-Bold'
BODY_FONT = 'Helvetica'

styles = getSampleStyleSheet()
styles.add(ParagraphStyle('DocTitle', fontName=HEADING_FONT, fontSize=20, textColor=COLORS['heading'], leading=24, spaceAfter=6, alignment=TA_LEFT))
styles.add(ParagraphStyle('DocSubtitle', fontName=BODY_FONT, fontSize=11, textColor=COLORS['muted'], leading=14, spaceAfter=15, alignment=TA_LEFT))
styles.add(ParagraphStyle('CustomH1', fontName=HEADING_FONT, fontSize=14, textColor=COLORS['heading'], leading=18, spaceBefore=14, spaceAfter=8))
styles.add(ParagraphStyle('CustomH2', fontName=HEADING_FONT, fontSize=11, textColor=COLORS['heading'], leading=14, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle('CustomBody', fontName=BODY_FONT, fontSize=10, textColor=COLORS['body'], leading=14, spaceAfter=6, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle('CustomCaption', fontName=BODY_FONT, fontSize=8, textColor=COLORS['muted'], leading=11, spaceAfter=8, alignment=TA_CENTER))
styles.add(ParagraphStyle('CustomTableHead', fontName=HEADING_FONT, fontSize=9, textColor=COLORS['white'], leading=11))
styles.add(ParagraphStyle('CustomTableBody', fontName=BODY_FONT, fontSize=8.5, textColor=COLORS['body'], leading=11))
styles.add(ParagraphStyle('WarnBody', fontName=BODY_FONT, fontSize=10, textColor=COLORS['warn'], leading=14, spaceAfter=6))

DOC_TITLE = "RELATÓRIO TÉCNICO: DEEP LEARNING COM PYTORCH"
DOC_SUBTITLE = "Desenvolvimento, Estabilização e Avaliação de Rede Neural MLP no Dataset Lending Club"

PAGE_SIZE = LETTER
MARGIN = 54
PAGE_W, PAGE_H = PAGE_SIZE
USABLE_W = PAGE_W - 2 * MARGIN

RESULTS_DIR = "results"
HISTORY_PATH = os.path.join(RESULTS_DIR, "run_history.json")
EPOCH_HISTORY_CLASS_PATH = os.path.join(RESULTS_DIR, "history_class.json")
EPOCH_HISTORY_REG_PATH = os.path.join(RESULTS_DIR, "history_reg.json")

# Nomes de modelo esperados no histórico (ajuste se usar outros nomes em log_run)
BASELINE_CLASS_NAME = "baseline_logistic_regression"
MLP_CLASS_NAME = "mlp"
BASELINE_REG_NAME = "baseline_linear_regression"
MLP_REG_NAME = "mlp"


# ── Helpers de formatação segura (sem inventar valor quando ausente) ───────────
def _fmt(value: Optional[float], casas: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{casas}f}"
    return "N/D"


def _get_metric(entry: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not entry:
        return None
    return entry.get("metrics", {}).get(key)


# ── Flowables customizados ──────────────────────────────────────────────
class SectionDivider(Flowable):
    def __init__(self, width, color):
        Flowable.__init__(self)
        self._width = width
        self._color = color
        self._height = 15

    def wrap(self, availWidth, availHeight):
        return self._width, self._height

    def draw(self):
        y = self._height / 2
        self.canv.setStrokeColor(self._color)
        self.canv.setLineWidth(1)
        self.canv.line(0, y, self._width, y)


class CalloutBox(Flowable):
    def __init__(self, text, width, colors, body_style, bar_color=None):
        Flowable.__init__(self)
        self._width = width
        self.colors = colors
        self.bar_color = bar_color or colors['accent']
        self.bar_w = 6
        self.pad = 10
        inner_w = self._width - self.bar_w - 2 * self.pad
        self._para = Paragraph(text, body_style)
        self._para_w, self._para_h = self._para.wrap(inner_w, 10000)
        self._height = self._para_h + 2 * self.pad

    def wrap(self, availWidth, availHeight):
        return self._width, self._height

    def draw(self):
        self.canv.setFillColor(self.colors['bg_alt'])
        self.canv.rect(0, 0, self._width, self._height, fill=1, stroke=0)
        self.canv.setFillColor(self.bar_color)
        self.canv.rect(0, 0, self.bar_w, self._height, fill=1, stroke=0)
        self._para.drawOn(self.canv, self.bar_w + self.pad, self.pad)


def create_report_table(headers, rows, col_widths=None):
    header_row = [Paragraph(str(h), styles['CustomTableHead']) for h in headers]
    data_rows = [[Paragraph(str(cell), styles['CustomTableBody']) for cell in row] for row in rows]
    t = Table([header_row] + data_rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLORS['bg_header']),
        ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [COLORS['white'], COLORS['bg_alt']]),
        ('GRID', (0, 0), (-1, -1), 0.5, COLORS['muted']),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    return t


def on_later_pages(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(COLORS['heading'])
    canvas.setLineWidth(0.6)
    y_rule = PAGE_H - MARGIN + 4
    canvas.line(MARGIN, y_rule, PAGE_W - MARGIN, y_rule)
    canvas.setFont(HEADING_FONT, 7.5)
    canvas.setFillColor(COLORS['heading'])
    canvas.drawString(MARGIN, y_rule + 4, "RELATÓRIO TÉCNICO - DEEP LEARNING COM PYTORCH")
    canvas.drawRightString(PAGE_W - MARGIN, y_rule + 4, "LENDING CLUB PIPELINE")
    y_footer = MARGIN - 20
    canvas.setStrokeColor(COLORS['muted'])
    canvas.setLineWidth(0.3)
    canvas.line(MARGIN, y_footer + 10, PAGE_W - MARGIN, y_footer + 10)
    canvas.drawString(MARGIN, y_footer, "Gerado automaticamente a partir de results/run_history.json")
    canvas.drawRightString(PAGE_W - MARGIN, y_footer, f"Pág. {doc.page}")
    canvas.restoreState()


def on_title_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(COLORS['heading'])
    canvas.rect(0, PAGE_H - 12, PAGE_W, 12, fill=1, stroke=0)
    canvas.restoreState()


# ── 2. Gráficos de curva de perda por época (só se os dados existirem) ─────
def _plot_loss_curve(history_path: str, out_png: str, ylabel: str, titulo: str) -> bool:
    """Gera o PNG da curva de perda a partir de um arquivo real de histórico por época.
    Retorna False (sem gerar nada) se o arquivo não existir — nunca usa dados de exemplo."""
    if not os.path.exists(history_path):
        return False
    with open(history_path, 'r', encoding='utf-8') as f:
        hist = json.load(f)
    if not hist:
        return False

    epochs = list(range(1, len(hist) + 1))
    train_loss = [h.get('train_loss') for h in hist]
    val_loss = [h.get('val_loss') for h in hist]

    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=300)
    ax.plot(epochs, train_loss, label="Treino", color="#1a2b4c", linewidth=2.2, marker='o', markersize=4)
    ax.plot(epochs, val_loss, label="Validação", color="#ff6b35", linewidth=2.2, linestyle="--", marker='s', markersize=4)
    ax.set_title(titulo, fontsize=10, fontweight="bold", color="#1a2b4c", pad=8)
    ax.set_xlabel("Épocas", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, frameon=True, facecolor="white", edgecolor="none")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return True


# ── 3. Texto de discussão dinâmico (nunca afirma sucesso sem checar os números) ──
def gerar_discussao_classificacao(latest: Dict[str, Dict[str, Any]]) -> str:
    baseline = latest.get(BASELINE_CLASS_NAME)
    mlp = latest.get(MLP_CLASS_NAME)

    if not baseline or not mlp:
        faltando = []
        if not baseline:
            faltando.append(f"'{BASELINE_CLASS_NAME}'")
        if not mlp:
            faltando.append(f"'{MLP_CLASS_NAME}'")
        return (
            "<b>DISCUSSÃO E INTERPRETAÇÃO COMERCIAL:</b> Dados insuficientes para comparar "
            f"baseline e MLP — nenhuma execução registrada para {', '.join(faltando)} em "
            f"'{HISTORY_PATH}'. Registre execuções com <code>experiment_tracker.log_run(...)</code> "
            "ao final do treino/avaliação para habilitar esta seção."
        )

    b_recall = _get_metric(baseline, "recall")
    m_recall = _get_metric(mlp, "recall")
    b_f1 = _get_metric(baseline, "f1")
    m_f1 = _get_metric(mlp, "f1")

    if b_recall is None or m_recall is None:
        return (
            "<b>DISCUSSÃO E INTERPRETAÇÃO COMERCIAL:</b> As execuções mais recentes de baseline "
            "e MLP não contêm a métrica 'recall' da classe Charged Off — verifique se "
            "'evaluate_on_test_set' foi executado em modo de classificação."
        )

    diff = m_recall - b_recall
    if diff > 1e-9:
        veredito = (
            f"O MLP ('{MLP_CLASS_NAME}') superou o baseline no recall da classe minoritária "
            f"'Charged Off' ({_fmt(m_recall)} vs {_fmt(b_recall)}), detectando corretamente uma "
            "fração maior dos empréstimos que de fato entraram em inadimplência."
        )
    elif diff < -1e-9:
        veredito = (
            f"O MLP ('{MLP_CLASS_NAME}') apresentou recall <b>inferior</b> ao baseline na classe "
            f"'Charged Off' ({_fmt(m_recall)} vs {_fmt(b_recall)}). Isso indica que o desbalanceamento "
            "de classes provavelmente não foi tratado (ex.: ausência de peso de classe no "
            "critério de perda ou de reamostragem), fazendo o modelo tender a prever a classe "
            "majoritária. Recomenda-se revisar o 'criterion' antes de considerar este modelo "
            "para produção."
        )
    else:
        veredito = (
            f"O MLP ('{MLP_CLASS_NAME}') obteve recall equivalente ao baseline na classe "
            f"'Charged Off' ({_fmt(m_recall)})."
        )

    f1_txt = ""
    if isinstance(b_f1, (int, float)) and isinstance(m_f1, (int, float)):
        f1_txt = f" O F1-score da classe minoritária foi {_fmt(m_f1)} (MLP) contra {_fmt(b_f1)} (baseline)."

    return (
        "<b>DISCUSSÃO E INTERPRETAÇÃO COMERCIAL:</b> A correta classificação de inadimplência "
        "(Classe 1 - Charged Off) é a métrica de maior relevância comercial, pois protege o caixa "
        f"da instituição de perdas diretas por calote. {veredito}{f1_txt} A acurácia agregada não é "
        "um bom indicador neste problema devido ao desbalanceamento entre as classes "
        "(~78% Fully Paid / ~22% Charged Off)."
    )


def gerar_discussao_regressao(latest: Dict[str, Dict[str, Any]]) -> str:
    baseline = latest.get(BASELINE_REG_NAME)
    mlp = latest.get(MLP_REG_NAME)

    if not baseline or not mlp:
        faltando = []
        if not baseline:
            faltando.append(f"'{BASELINE_REG_NAME}'")
        if not mlp:
            faltando.append(f"'{MLP_REG_NAME}'")
        return (
            "<b>DISCUSSÃO (REGRESSÃO):</b> Dados insuficientes para comparar baseline e MLP — "
            f"nenhuma execução registrada para {', '.join(faltando)} em '{HISTORY_PATH}'."
        )

    b_rmse = _get_metric(baseline, "rmse")
    m_rmse = _get_metric(mlp, "rmse")
    if b_rmse is None or m_rmse is None:
        return "<b>DISCUSSÃO (REGRESSÃO):</b> Métrica 'rmse' ausente em uma das execuções mais recentes."

    if m_rmse < b_rmse:
        return (
            f"<b>DISCUSSÃO (REGRESSÃO):</b> O MLP reduziu o RMSE de teste em relação ao baseline "
            f"linear ({_fmt(m_rmse, 6)} vs {_fmt(b_rmse, 6)}), indicando que a rede capturou relações "
            "não-lineares relevantes na predição da taxa de juros."
        )
    elif m_rmse > b_rmse:
        return (
            f"<b>DISCUSSÃO (REGRESSÃO):</b> O MLP apresentou RMSE de teste <b>maior</b> que o baseline "
            f"linear ({_fmt(m_rmse, 6)} vs {_fmt(b_rmse, 6)}), ou seja, não trouxe ganho sobre o modelo "
            "mais simples nesta execução. Vale revisar hiperparâmetros, regularização ou features "
            "antes de substituir o baseline."
        )
    return f"<b>DISCUSSÃO (REGRESSÃO):</b> RMSE equivalente entre MLP e baseline ({_fmt(m_rmse, 6)})."


# ── 4. Tabelas de comparação (sempre a partir do histórico real) ───────────
def tabela_classificacao(latest: Dict[str, Dict[str, Any]]):
    headers = ["Modelo / Abordagem", "Acurácia", "Precision", "Recall", "F1-Score", "Registrado em"]
    rows = []
    for nome_exibicao, model_key in [
        ("Regressão Logística (Baseline)", BASELINE_CLASS_NAME),
        ("MLP", MLP_CLASS_NAME),
    ]:
        entry = latest.get(model_key)
        if entry is None:
            rows.append([nome_exibicao, "N/D", "N/D", "N/D", "N/D", "sem execução registrada"])
            continue
        m = entry.get("metrics", {})
        rows.append([
            nome_exibicao,
            _fmt(m.get("accuracy")),
            _fmt(m.get("precision")),
            _fmt(m.get("recall")),
            _fmt(m.get("f1")),
            entry.get("timestamp", "N/D")[:19].replace("T", " "),
        ])
    return headers, rows


def tabela_regressao(latest: Dict[str, Dict[str, Any]]):
    headers = ["Modelo / Abordagem", "MAE", "MSE", "RMSE", "R²", "Registrado em"]
    rows = []
    for nome_exibicao, model_key in [
        ("Regressão Linear (Baseline)", BASELINE_REG_NAME),
        ("MLP", MLP_REG_NAME),
    ]:
        entry = latest.get(model_key)
        if entry is None:
            rows.append([nome_exibicao, "N/D", "N/D", "N/D", "N/D", "sem execução registrada"])
            continue
        m = entry.get("metrics", {})
        rows.append([
            nome_exibicao,
            _fmt(m.get("mae"), 6),
            _fmt(m.get("mse"), 6),
            _fmt(m.get("rmse"), 6),
            _fmt(m.get("r2")),
            entry.get("timestamp", "N/D")[:19].replace("T", " "),
        ])
    return headers, rows


def tabela_evolucao(task: str, model_name: str, chave_principal: str, casas: int = 4):
    """Mostra todas as execuções históricas de um mesmo modelo, para ver a evolução
    conforme o código vai sendo alterado e re-treinado."""
    runs = runs_for_model(task, model_name)
    if len(runs) < 2:
        return None  # nada a comparar ainda
    headers = ["#", "Registrado em", chave_principal, "Δ vs. execução anterior", "Notas"]
    rows = []
    prev_value = None
    for i, run in enumerate(runs, start=1):
        value = run.get("metrics", {}).get(chave_principal)
        delta_txt = "-"
        if isinstance(value, (int, float)) and isinstance(prev_value, (int, float)):
            delta = value - prev_value
            seta = "▲" if delta > 0 else ("▼" if delta < 0 else "→")
            delta_txt = f"{seta} {delta:+.{casas}f}"
        rows.append([
            str(i),
            run.get("timestamp", "N/D")[:19].replace("T", " "),
            _fmt(value, casas),
            delta_txt,
            run.get("notes") or "-",
        ])
        if isinstance(value, (int, float)):
            prev_value = value
    return headers, rows


# ── 5. Montagem do documento ────────────────────────────────────────────
def build_pdf(output_path: str = "Relatorio_MLP_Lending_Club.pdf") -> None:
    history = load_history(HISTORY_PATH)
    latest_class = latest_run_per_model("classification", HISTORY_PATH)
    latest_reg = latest_run_per_model("regression", HISTORY_PATH)

    has_class_chart = _plot_loss_curve(
        EPOCH_HISTORY_CLASS_PATH, "chart_class.png",
        "Cross-Entropy Loss", "Curvas de Aprendizado: Classificação de Inadimplência",
    )
    has_reg_chart = _plot_loss_curve(
        EPOCH_HISTORY_REG_PATH, "chart_reg.png",
        "MSE Loss", "Curvas de Aprendizado: Regressão de Taxa de Juros",
    )

    doc = BaseDocTemplate(
        output_path, pagesize=PAGE_SIZE,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN + 10, bottomMargin=MARGIN,
    )
    content_frame = Frame(doc.leftMargin, doc.bottomMargin, USABLE_W, PAGE_H - doc.topMargin - doc.bottomMargin, id='main')
    doc.addPageTemplates([
        PageTemplate(id='title_page', frames=content_frame, onPage=on_title_page),
        PageTemplate(id='content', frames=content_frame, onPage=on_later_pages),
    ])

    story = []
    story.append(Spacer(1, 10))
    story.append(Paragraph(DOC_TITLE, styles['DocTitle']))
    story.append(Paragraph(DOC_SUBTITLE, styles['DocSubtitle']))
    story.append(SectionDivider(USABLE_W, COLORS['accent']))
    story.append(Spacer(1, 10))

    if not history:
        story.append(Paragraph(
            f"<b>AVISO:</b> Nenhuma execução encontrada em '{HISTORY_PATH}'. Este relatório está "
            "vazio porque nenhum treino/avaliação foi registrado ainda via "
            "<code>experiment_tracker.log_run(...)</code>. Execute o pipeline de treino e gere o "
            "relatório novamente.",
            styles['WarnBody'],
        ))
        doc.build(story)
        print(f"Relatório gerado em modo de aviso (sem dados): {output_path}")
        return

    resumo = (
        "<b>RESUMO EXECUTIVO:</b> Este relatório é gerado automaticamente a partir de "
        f"{len(history)} execuções registradas em '{HISTORY_PATH}'. Todos os números abaixo "
        "refletem resultados reais de teste cego — nenhum valor é estimado ou de exemplo."
    )
    story.append(CalloutBox(resumo, USABLE_W, COLORS, styles['CustomBody'], COLORS['heading']))
    story.append(Spacer(1, 12))

    # SEÇÃO: Classificação
    if latest_class:
        story.append(Paragraph("1. Classificação de Inadimplência", styles['CustomH1']))
        headers, rows = tabela_classificacao(latest_class)
        col_w = [USABLE_W * 0.28, USABLE_W * 0.14, USABLE_W * 0.14, USABLE_W * 0.14, USABLE_W * 0.14, USABLE_W * 0.16]
        story.append(create_report_table(headers, rows, col_w))
        story.append(Spacer(1, 8))

        if has_class_chart:
            story.append(Paragraph("Figura 1: Curva de Perda — Classificação", styles['CustomH2']))
            story.append(Image("chart_class.png", width=USABLE_W * 0.9, height=USABLE_W * 0.9 * 0.53))
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(
                f"Gráfico de curva de perda não disponível ('{EPOCH_HISTORY_CLASS_PATH}' não encontrado).",
                styles['CustomCaption'],
            ))

        evolucao = tabela_evolucao("classification", MLP_CLASS_NAME, "recall")
        if evolucao:
            story.append(Paragraph(f"Evolução do recall (Charged Off) — modelo '{MLP_CLASS_NAME}'", styles['CustomH2']))
            ev_headers, ev_rows = evolucao
            story.append(create_report_table(ev_headers, ev_rows, [USABLE_W * 0.06, USABLE_W * 0.22, USABLE_W * 0.14, USABLE_W * 0.22, USABLE_W * 0.36]))
            story.append(Spacer(1, 8))

        story.append(CalloutBox(gerar_discussao_classificacao(latest_class), USABLE_W, COLORS, styles['CustomBody'], COLORS['accent']))
        story.append(Spacer(1, 12))
    else:
        story.append(Paragraph(
            "Nenhuma execução de classificação registrada ainda.", styles['WarnBody'],
        ))

    story.append(NextPageTemplate('content'))
    story.append(PageBreak())

    # SEÇÃO: Regressão
    if latest_reg:
        story.append(Paragraph("2. Regressão da Taxa de Juros", styles['CustomH1']))
        headers, rows = tabela_regressao(latest_reg)
        col_w = [USABLE_W * 0.28, USABLE_W * 0.14, USABLE_W * 0.14, USABLE_W * 0.14, USABLE_W * 0.14, USABLE_W * 0.16]
        story.append(create_report_table(headers, rows, col_w))
        story.append(Spacer(1, 8))

        if has_reg_chart:
            story.append(Paragraph("Figura 2: Curva de Perda — Regressão", styles['CustomH2']))
            story.append(Image("chart_reg.png", width=USABLE_W * 0.9, height=USABLE_W * 0.9 * 0.53))
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(
                f"Gráfico de curva de perda não disponível ('{EPOCH_HISTORY_REG_PATH}' não encontrado).",
                styles['CustomCaption'],
            ))

        evolucao_reg = tabela_evolucao("regression", MLP_REG_NAME, "rmse", casas=6)
        if evolucao_reg:
            story.append(Paragraph(f"Evolução do RMSE — modelo '{MLP_REG_NAME}'", styles['CustomH2']))
            ev_headers, ev_rows = evolucao_reg
            story.append(create_report_table(ev_headers, ev_rows, [USABLE_W * 0.06, USABLE_W * 0.22, USABLE_W * 0.14, USABLE_W * 0.22, USABLE_W * 0.36]))
            story.append(Spacer(1, 8))

        story.append(CalloutBox(gerar_discussao_regressao(latest_reg), USABLE_W, COLORS, styles['CustomBody'], COLORS['accent']))
    else:
        story.append(Paragraph(
            "Nenhuma execução de regressão registrada ainda.", styles['WarnBody'],
        ))

    doc.build(story)
    print(f"Relatório PDF compilado com sucesso em: {output_path}")


if __name__ == "__main__":
    build_pdf()
