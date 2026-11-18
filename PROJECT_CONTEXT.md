# Projet Industriel ENSICAEN — STEREOSARC (Radiomique sur scanner)

> Documento de contexto para começar a trabalhar no projeto localmente.
> Fontes: `Stereosarc_indus_Biomed2026.docx` e `STEREOSARC_stage_Master_Biomed_2026_5dias.pptx`.

---

## 1. Identificação

| | |
|---|---|
| **Título (FR)** | Développement d'un pipeline de radiomique sur scanner pour la détection précoce d'une progression métastatique |
| **Título (EN)** | Development of a CT radiomics pipeline for early prediction of metastatic progression |
| **Quadro** | Projet industriel ENSICAEN (sujet originalmente proposto como stage Master Biomédical 2026) |
| **Proponente / orientação clínica** | Pr Juliette Thariat — Centre François Baclesse, Caen |
| **Encadrant ENSICAEN** | Cédric Loiseau |
| **Equipe** | Arthur, Gustavo, Abdel |
| **Áreas** | Radiothérapie · Imagerie biomédicale · Radiomique / IA |

---

## 2. Contexto clínico

- **STEREOSARC** é um estudo de pacientes com **sarcoma oligometastático** (1 a 5 metástases visíveis), tratados principalmente por **radioterapia estereotáxica**.
- Problema clínico: distinguir **cedo** os pacientes cuja doença vai permanecer limitada daqueles que vão desenvolver **novas metástases**.
- Há **scanners (TC) de acompanhamento** disponíveis para esses pacientes.
- **Hipótese:** alterações muito sutis na imagem, **invisíveis ao olho humano**, poderiam **preceder** o aparecimento de novas metástases visíveis.
- A **radiomique** transforma imagens médicas em dados quantitativos (textura, intensidade, forma, heterogeneidade).
- Visão de longo prazo (prevista no protocolo): combinar imagem + dados clínicos + dados biológicos → **predição personalizada** (abordagem multimodal).

---

## 3. Objetivo principal

Desenvolver e validar uma **cadeia reprodutível** que extraia características radiômicas dos scanners de acompanhamento e estudar sua capacidade de detectar **alterações precoces** associadas ao surgimento posterior de novas metástases.

Em outras palavras: uma **prova de conceito** para antecipar a progressão antes que ela seja evidente na imagem.

---

## 4. Dados

- **~300 scanners** (exames de acompanhamento, cohorte real)
- **~500 cortes** por exame → ~150 000 cortes no total
- Formato provável: DICOM (a confirmar com o Centre Baclesse)
- Dados clínicos/biológicos do STEREOSARC existem, mas **não são o foco** desta etapa

---

## 5. Pipeline a construir (6 etapas)

```
1. Importar      →  2. Padronizar + QC  →  3. Excluir zonas já tratadas
                                                   ↓
6. Explorar      ←  5. Estruturar CSV   ←  4. Extrair parâmetros (PyRadiomics)
   variabilidade/sinal
```

| # | Etapa | O que envolve |
|---|---|---|
| 1 | **Importação** | Leitura dos scanners, organização por paciente / data de exame |
| 2 | **Padronização + controle de qualidade** | Reamostragem (voxel spacing), normalização de intensidades (HU), verificação de artefatos, cortes faltando, protocolos de aquisição heterogêneos |
| 3 | **Exclusão das zonas tratadas** | Identificar e excluir as lesões irradiadas e as alterações pós-terapêuticas, de forma **reprodutível** |
| 4 | **Definição das ROIs + extração** | Definir regiões de interesse pertinentes; extração automatizada com **Python + PyRadiomics** |
| 5 | **Base estruturada** | Tabela de features radiômicas, exportável em **CSV** |
| 6 | **Análise exploratória** | Variabilidade e robustez dos parâmetros; primeiras assinaturas |

---

## 6. Pontos metodológicos críticos ⚠️

1. **Não confundir efeito de tratamento com sinal de progressão.**
   As metástases conhecidas e irradiadas sofrem remodelamento pós-radioterapia (inflamação, fibrose, necrose…). Essas zonas **devem ser identificadas e excluídas** por um procedimento documentado e reprodutível.
2. **Excluir exames com novas metástases já claramente visíveis** da análise preditiva principal — o objetivo é encontrar um sinal **antes** da progressão radiológica evidente.
3. **Padronização antes de qualquer análise.** Features radiômicas são sensíveis ao scanner, ao protocolo, à espessura de corte, ao kernel de reconstrução etc. Sem padronização, comparações entre pacientes e ao longo do tempo não fazem sentido.
4. **Robustez/reprodutibilidade das features**: buscar assinaturas estáveis entre pacientes e ao longo do tempo.

---

## 7. Entregáveis esperados

- [ ] Pipeline reprodutível de análise radiômica dos scanners
- [ ] Procedimento documentado de padronização e controle de qualidade
- [ ] Método reprodutível de exclusão das regiões correspondentes às metástases já tratadas
- [ ] Base estruturada de features radiômicas (export CSV)
- [ ] Primeira análise da variabilidade e robustez dos parâmetros

**Critério-chave:** o pipeline deve poder ser **reutilizado em toda a cohorte**.

---

## 8. Perspectiva (fora do escopo atual)

Numa etapa seguinte (ex.: stage de M2), as assinaturas de imagem serão confrontadas com **marcadores tumorais sanguíneos** dos mesmos pacientes:
**Scanner + sangue + IA → predição personalizada.**

---

## 9. Competências mobilizadas

Imagerie biomédicale · Python · traitement d'images · PyRadiomics · analyse de données · statistiques · gestion de données · reproductibilité

---

## 10. Sugestões para começar (não fazem parte do sujet — são propostas)

### Stack técnica
- `pydicom`, `SimpleITK` — leitura DICOM, reamostragem, máscaras
- `pyradiomics` — extração (configurar via arquivo `params.yaml`)
- `numpy`, `pandas` — manipulação e export CSV
- `matplotlib` / `seaborn` — QC visual e exploração
- Opcional: `TotalSegmentator` (segmentação automática de órgãos para definir ROIs), `scikit-learn` (ICC, PCA, clustering), `pingouin` (ICC)
- Versionamento: Git; ambiente: `conda` ou `venv` + `requirements.txt`

> Atenção: PyRadiomics costuma exigir Python ≤ 3.9/3.10 dependendo da versão — verificar compatibilidade antes de montar o ambiente.

### Estrutura de repositório proposta
```
stereosarc-radiomics/
├── README.md
├── PROJECT_CONTEXT.md
├── requirements.txt
├── config/
│   └── pyradiomics_params.yaml
├── data/                # NÃO versionar (dados de saúde)
│   ├── raw/
│   ├── processed/
│   └── masks/
├── src/
│   ├── io_dicom.py          # etapa 1
│   ├── preprocessing.py     # etapa 2 (reamostragem, normalização)
│   ├── quality_control.py   # etapa 2 (QC)
│   ├── exclusion_masks.py   # etapa 3
│   ├── roi.py               # etapa 4 (definição das ROIs)
│   ├── extraction.py        # etapa 4 (PyRadiomics)
│   └── build_dataset.py     # etapa 5 (CSV)
├── notebooks/
│   └── exploration.ipynb    # etapa 6
├── outputs/
│   ├── features.csv
│   └── qc_reports/
└── docs/
    ├── procedure_standardisation.md
    └── procedure_exclusion.md
```

### Dados de saúde
- Dados de pacientes → **nunca** subir para GitHub público nem serviços externos; adicionar `data/` ao `.gitignore`.
- Confirmar com o Centre Baclesse as regras de acesso, anonimização e onde os dados podem ser processados.

---

## 11. Questões em aberto (a esclarecer com Pr Thariat / M. Loiseau)

- [ ] Formato e acesso aos dados (DICOM? já anonimizados? servidor local do Baclesse?)
- [ ] Existem contornos (RTSTRUCT) das lesões irradiadas / dos volumes de radioterapia? → base para a máscara de exclusão
- [ ] Quais ROIs analisar: órgãos inteiros (pulmão, fígado…), parênquima saudável, outra definição?
- [ ] Locais típicos das metástases nesse grupo de sarcomas (pulmão é frequente) → orienta a escolha das ROIs
- [ ] Heterogeneidade dos protocolos de aquisição (scanners diferentes, com/sem contraste, espessura de corte)
- [ ] Como está anotada a data de aparecimento das novas metástases (label para a análise)
- [ ] Recursos computacionais disponíveis (~150 000 cortes)
- [ ] Calendário, jalons e formato dos relatórios do projet industriel
- [ ] Divisão de tarefas entre Arthur, Gustavo e Abdel
