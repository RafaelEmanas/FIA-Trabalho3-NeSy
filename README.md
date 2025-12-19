# Trabalho Final de FIA: Raciocínio Espacial Neuro-Simbólico com LTNtorch

**Disciplina**: Fundamentos de Inteligência Artificial (ICC260)  
**Professor**: Edjard Mota  
**Tópico**: Logic Tensor Networks (LTN) & Raciocínio Espacial  
**Alunos**:
1. Rafael Emanuel Dantas Viana
2. Victor José Nunes Kossmann
3. Caio da Silva Martins
4. Antonio Lucas de Melo Barbosa Mendes Rodrigues
5. Breno Augusto Pinheiro Rodrigues da Silva
6. Gabryella Reis

---

## 📋 Índice

1. Visão Geral
2. Estrutura do Dataset
3. Implementação
4. Resultados

---

## Visão Geral

Este projeto implementa um **agente neuro-simbólico** capaz de entender relações espaciais usando **Logic Tensor Networks (LTN)**. Diferente do Deep Learning tradicional (Input → Output), o LTN permite ensinar à rede **regras lógicas** sobre como objetos se relacionam em um espaço 2D.

### O que é LTN?

**Logic Tensor Networks** é um framework neuro-simbólico que:
- Combina **aprendizado profundo** (redes neurais) com **raciocínio lógico** (lógica de primeira ordem)
- Permite definir **axiomas lógicos** (regras) que guiam o treinamento
- Converte fórmulas lógicas em **funções de perda diferenciáveis**
- Treina predicados para satisfazer conhecimento prévio

### O que é NeSy (Neuro-Simbólico)?

**Neuro-Symbolic AI** é uma abordagem que integra:
- **Componente Neural**: Aprende padrões dos dados (subsimbólico)
- **Componente Simbólico**: Raciocínio explícito com regras lógicas
- **Vantagens**: Explicabilidade, generalização, uso de conhecimento prévio

---

## Estrutura do Dataset

### Dataset CLEVR Simplificado

O projeto utiliza uma versão simplificada do dataset **CLEVR** (Compositional Language and Elementary Visual Reasoning), representando objetos como **vetores de features** ao invés de imagens.

### Formato do Vetor (11 dimensões)

```
[0, 1]       : Posição (x, y) — Coordenadas normalizadas [0.0, 1.0]
[2, 3, 4]    : Cores One-Hot (Vermelho, Verde, Azul)
[5, 6, 7, 8, 9] : Formas One-Hot (Círculo, Quadrado, Cilindro, Cone, Triângulo)
[10]         : Tamanho (0.0 = Pequeno, 1.0 = Grande)
```

**Exemplo de objeto**:
```python
[0.68, 0.64, 1, 0, 0, 0, 1, 0, 0, 0, 1.0]
# Quadrado vermelho grande na posição (0.68, 0.64)
```

---

## Implementação

### Estrutura do Código

O projeto foi desenvolvido em **Python** usando **PyTorch** e **LTNtorch**, seguindo a estrutura recomendada:

```
1. Geração de Dados (CLEVR Simplificado)
2. Definição de Predicados (Redes Neurais)
3. Operadores Lógicos (And, Or, Not, Implies, Forall, Exists)
4. Axiomas (Base de Conhecimento)
5. Treinamento (Otimização via Satisfatibilidade)
6. Validação (5 Cenários Aleatórios)
7. Queries Compostas (Raciocínio Complexo)
8. Explicabilidade (Ponto Extra)
```

---

## 📊 Tarefas Implementadas

### ✅ Tarefa 1: Taxonomia e Formas (2 pontos)

**Predicados Implementados:**
- **Formas**: `IsCircle`, `IsSquare`, `IsCylinder`, `IsCone`, `IsTriangle`
- **Tamanhos**: `IsSmall`, `IsBig`
- **Cores**: `IsRed`, `IsGreen`, `IsBlue`

**Axiomas Implementados:**

1. **Forma Única (Exclusão Mútua)**:
   ```
   ∀x, ¬(IsCircle(x) ∧ IsSquare(x))
   ∀x, ¬(IsCircle(x) ∧ IsCylinder(x))
   ... (todas as combinações)
   ```

2. **Cobertura (Completude)**:
   ```
   ∀x, (IsCircle(x) ∨ IsSquare(x) ∨ IsCylinder(x) ∨ IsCone(x) ∨ IsTriangle(x))
   ```

3. **Tamanho Único**:
   ```
   ∀x, ¬(IsSmall(x) ∧ IsBig(x))
   ∀x, (IsSmall(x) ∨ IsBig(x))
   ```

4. **Cor Única**:
   ```
   ∀x, ¬(IsRed(x) ∧ IsGreen(x))
   ... (todas as combinações)
   ```

---

### ✅ Tarefa 2: Raciocínio Espacial Horizontal (5 pontos)

**Predicados Implementados:**
- `LeftOf(x, y)`: x está à esquerda de y
- `RightOf(x, y)`: x está à direita de y
- `CloseTo(x, y)`: x está próximo de y (Kernel Gaussiano)
- `InBetween(x, y, z)`: x está entre y e z

**Axiomas Implementados:**

1. **Irreflexividade**:
   ```
   ∀x, ¬LeftOf(x, x)
   ```

2. **Assimetria**:
   ```
   ∀x,y, (LeftOf(x, y) ⇒ ¬LeftOf(y, x))
   ```

3. **Inverso**:
   ```
   ∀x,y, (LeftOf(x, y) ⇔ RightOf(y, x))
   ```

4. **Transitividade**:
   ```
   ∀x,y,z, (LeftOf(x, y) ∧ LeftOf(y, z) ⇒ LeftOf(x, z))
   ```

5. **CloseTo (Proximidade)**:
   ```python
   exp(-2 * ||x_pos - y_pos||²)
   ```

6. **InBetween**:
   ```
   (LeftOf(y, x) ∧ RightOf(z, x)) ∨ (LeftOf(z, x) ∧ RightOf(y, x))
   ```

7. **LastOnTheLeft**:
   ```
   ∃x(∀y, ¬Equal(x,y) ⇒ LeftOf(x, y))
   ```

8. **LastOnTheRight**:
   ```
   ∃x(∀y, ¬Equal(x,y) ⇒ RightOf(x, y))
   ```

---

### ✅ Tarefa 3: Raciocínio Espacial Vertical (Parte da Tarefa 2)

**Predicados Implementados:**
- `Below(x, y)`: x está abaixo de y
- `Above(x, y)`: x está acima de y
- `CanStack(x, y)`: x pode ser empilhado sobre y

**Axiomas Implementados:**

1. **Irreflexividade**:
   ```
   ∀x, ¬Below(x, x)
   ```

2. **Assimetria**:
   ```
   ∀x,y, (Below(x, y) ⇒ ¬Below(y, x))
   ```

3. **Inverso**:
   ```
   ∀x,y, (Below(x, y) ⇔ Above(y, x))
   ```

4. **Transitividade**:
   ```
   ∀x,y,z, (Below(x, y) ∧ Below(y, z) ⇒ Below(x, z))
   ```

5. **CanStack**:
   ```
   ∀x,y, (CanStack(x, y) ⇔ (¬IsCone(y) ∧ ¬IsTriangle(y)))
   ```

---

### ✅ Tarefa 4: Raciocínio Composto (2 pontos)

**Query 1**: Filtragem Composta
```
∃x(IsSmall(x) ∧ ∃y(IsCylinder(y) ∧ Below(x, y)) ∧ ∃z(IsSquare(z) ∧ LeftOf(x, z)))
```
_"Existe algum objeto pequeno que esteja abaixo de um cilindro E à esquerda de um quadrado?"_

**Query 2**: Dedução de Posição Absoluta
```
∃x,y,z(IsCone(x) ∧ IsGreen(x) ∧ InBetween(x, y, z))
```
_"Existe um cone verde que está entre dois outros objetos?"_

**Query 3**: Restrição de Proximidade
```
∀x,y((IsTriangle(x) ∧ IsTriangle(y) ∧ CloseTo(x, y)) ⇒ SameSize(x, y))
```
_"Se dois triângulos estão próximos, então devem ter o mesmo tamanho."_

---

## Resultados

### 📊 Validação em 5 Cenários Aleatórios

## 🔹 CENÁRIO 1

### Predicados Unários

| Predicado     | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| IsCircle      | 1.000    | 1.000     | 1.000  | 1.000    |
| IsSquare      | 1.000    | 1.000     | 1.000  | 1.000    |
| IsCylinder    | 0.920    | 1.000     | 0.667  | 0.800    |
| IsCone        | 1.000    | 1.000     | 1.000  | 1.000    |
| IsTriangle    | 1.000    | 1.000     | 1.000  | 1.000    |
| **IsRed**     | **1.000** | **1.000** | **1.000** | **1.000** |
| **IsGreen**   | **1.000** | **1.000** | **1.000** | **1.000** |
| **IsBlue**    | **1.000** | **1.000** | **1.000** | **1.000** |
| IsSmall       | 1.000    | 1.000     | 1.000  | 1.000    |
| IsBig         | 1.000    | 1.000     | 1.000  | 1.000    |

### Predicados Espaciais

| Predicado | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| LeftOf    | 0.915    | 0.875     | 0.960  | 0.916    |
| RightOf   | 0.904    | 0.866     | 0.947  | 0.904    |
| Below     | 0.923    | 0.882     | 0.970  | 0.924    |
| Above     | 0.926    | 0.885     | 0.973  | 0.927    |
| CanStack  | 1.000    | 1.000     | 1.000  | 1.000    |

### Queries Compostas

| Query | Satisfatibilidade |
|------|-------------------|
| Q1: Pequeno abaixo cilindro E esq. quadrado | 0.043 |
| Q2: Cone verde entre dois objetos | 0.106 |
| Q3: Triângulos próximos mesmo tamanho | 0.958 |
| Opcional 1: Existe obj. esq. todos quadrados | 0.807 |
| Opcional 2: Quadrados direita círculos | 0.947 |
| LastOnTheLeft | 0.467 |
| LastOnTheRight | 0.453 |

### Explicações Geradas

```

✓ Q1: Objeto 9 (pequeno) está abaixo do objeto 12 (cilindro)
E à esquerda do objeto 10 (quadrado).

✗ Q2: Cone verde 6 não está entre outros objetos.

✓ Q3: Todos os triângulos próximos têm o mesmo tamanho.

```

---

## 🔹 CENÁRIO 2

### Predicados Unários

| Predicado     | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| IsCircle      | 1.000    | 1.000     | 1.000  | 1.000    |
| IsSquare      | 1.000    | 1.000     | 1.000  | 1.000    |
| IsCylinder    | 0.920    | 1.000     | 0.714  | 0.833    |
| IsCone        | 1.000    | 1.000     | 1.000  | 1.000    |
| IsTriangle    | 0.960    | 0.000     | 0.000  | 0.000    |
| **IsRed**     | **1.000** | **1.000** | **1.000** | **1.000** |
| **IsGreen**   | **1.000** | **1.000** | **1.000** | **1.000** |
| **IsBlue**    | **1.000** | **1.000** | **1.000** | **1.000** |
| IsSmall       | 1.000    | 1.000     | 1.000  | 1.000    |
| IsBig         | 1.000    | 1.000     | 1.000  | 1.000    |

### Predicados Espaciais

| Predicado | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| LeftOf    | 0.936    | 0.896     | 0.980  | 0.936    |
| RightOf   | 0.926    | 0.897     | 0.957  | 0.926    |
| Below     | 0.922    | 0.870     | 0.983  | 0.923    |
| Above     | 0.920    | 0.877     | 0.970  | 0.921    |
| CanStack  | 1.000    | 1.000     | 1.000  | 1.000    |

### Queries Compostas

| Query | Satisfatibilidade |
|------|-------------------|
| Q1: Pequeno abaixo cilindro E esq. quadrado | 0.145 |
| Q2: Cone verde entre dois objetos | 0.172 |
| Q3: Triângulos próximos mesmo tamanho | 0.998 |
| Opcional 1: Existe obj. esq. todos quadrados | 0.714 |
| Opcional 2: Quadrados direita círculos | 0.818 |
| LastOnTheLeft | 0.458 |
| LastOnTheRight | 0.442 |

### Explicações Geradas

```

✓ Q1: Objeto 2 (pequeno) está abaixo do objeto 16 (cilindro)
E à esquerda do objeto 9 (quadrado).

✓ Q2: Objeto 4 é um cone verde em x=0.19, com objetos [5, 11]
à esquerda e [0, 1] à direita.

✓ Q3: Menos de 2 triângulos no cenário.

```

---

## 🔹 CENÁRIO 3

### Predicados Unários

| Predicado     | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| IsCircle      | 1.000    | 1.000     | 1.000  | 1.000    |
| IsSquare      | 1.000    | 1.000     | 1.000  | 1.000    |
| IsCylinder    | 0.960    | 1.000     | 0.800  | 0.889    |
| IsCone        | 1.000    | 1.000     | 1.000  | 1.000    |
| IsTriangle    | 1.000    | 1.000     | 1.000  | 1.000    |
| **IsRed**     | **1.000** | **1.000** | **1.000** | **1.000** |
| **IsGreen**   | **1.000** | **1.000** | **1.000** | **1.000** |
| **IsBlue**    | **1.000** | **1.000** | **1.000** | **1.000** |
| IsSmall       | 1.000    | 1.000     | 1.000  | 1.000    |
| IsBig         | 1.000    | 1.000     | 1.000  | 1.000    |

### Predicados Espaciais

| Predicado | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| LeftOf    | 0.906    | 0.885     | 0.923  | 0.904    |
| RightOf   | 0.902    | 0.892     | 0.907  | 0.899    |
| Below     | 0.918    | 0.876     | 0.967  | 0.919    |
| Above     | 0.928    | 0.888     | 0.973  | 0.928    |
| CanStack  | 1.000    | 1.000     | 1.000  | 1.000    |

### Queries Compostas

| Query | Satisfatibilidade |
|------|-------------------|
| Q1: Pequeno abaixo cilindro E esq. quadrado | 0.170 |
| Q2: Cone verde entre dois objetos | 0.142 |
| Q3: Triângulos próximos mesmo tamanho | 0.999 |
| Opcional 1: Existe obj. esq. todos quadrados | 0.640 |
| Opcional 2: Quadrados direita círculos | 0.808 |
| LastOnTheLeft | 0.447 |
| LastOnTheRight | 0.424 |

### Explicações Geradas

```

✓ Q1: Objeto 2 (pequeno) está abaixo do objeto 1 (cilindro)
E à esquerda do objeto 0 (quadrado).

✓ Q2: Objeto 2 é um cone verde em x=0.06, com objetos [10, 20]
à esquerda e [0, 1] à direita.

✓ Q3: Todos os triângulos próximos têm o mesmo tamanho.

```

---

## 🔹 CENÁRIO 4

### Predicados Unários

| Predicado     | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| IsCircle      | 1.000    | 1.000     | 1.000  | 1.000    |
| IsSquare      | 1.000    | 1.000     | 1.000  | 1.000    |
| IsCylinder    | 1.000    | 1.000     | 1.000  | 1.000    |
| IsCone        | 1.000    | 1.000     | 1.000  | 1.000    |
| IsTriangle    | 0.960    | 1.000     | 0.500  | 0.667    |
| **IsRed**     | **1.000** | **1.000** | **1.000** | **1.000** |
| **IsGreen**   | **1.000** | **1.000** | **1.000** | **1.000** |
| **IsBlue**    | **1.000** | **1.000** | **1.000** | **1.000** |
| IsSmall       | 1.000    | 1.000     | 1.000  | 1.000    |
| IsBig         | 1.000    | 1.000     | 1.000  | 1.000    |

### Predicados Espaciais

| Predicado | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| LeftOf    | 0.917    | 0.885     | 0.950  | 0.916    |
| RightOf   | 0.922    | 0.884     | 0.963  | 0.922    |
| Below     | 0.923    | 0.884     | 0.967  | 0.924    |
| Above     | 0.915    | 0.869     | 0.970  | 0.917    |
| CanStack  | 1.000    | 1.000     | 1.000  | 1.000    |

### Queries Compostas

| Query | Satisfatibilidade |
|------|-------------------|
| Q1: Pequeno abaixo cilindro E esq. quadrado | 0.066 |
| Q2: Cone verde entre dois objetos | 0.220 |
| Q3: Triângulos próximos mesmo tamanho | 0.999 |
| Opcional 1: Existe obj. esq. todos quadrados | 0.713 |
| Opcional 2: Quadrados direita círculos | 0.838 |
| LastOnTheLeft | 0.456 |
| LastOnTheRight | 0.454 |

### Explicações Geradas

```

✓ Q1: Objeto 7 (pequeno) está abaixo do objeto 12 (cilindro)
E à esquerda do objeto 19 (quadrado).

✓ Q2: Objeto 1 é um cone verde em x=0.79, com objetos [0, 2]
à esquerda e [6, 7] à direita.

✓ Q3: Menos de 2 triângulos no cenário.

```

---

## 🔹 CENÁRIO 5

### Predicados Unários

| Predicado     | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| IsCircle      | 1.000    | 1.000     | 1.000  | 1.000    |
| IsSquare      | 1.000    | 1.000     | 1.000  | 1.000    |
| IsCylinder    | 0.960    | 1.000     | 0.667  | 0.800    |
| IsCone        | 1.000    | 1.000     | 1.000  | 1.000    |
| IsTriangle    | 0.960    | 1.000     | 0.857  | 0.923    |
| **IsRed**     | **1.000** | **1.000** | **1.000** | **1.000** |
| **IsGreen**   | **1.000** | **1.000** | **1.000** | **1.000** |
| **IsBlue**    | **1.000** | **1.000** | **1.000** | **1.000** |
| IsSmall       | 1.000    | 1.000     | 1.000  | 1.000    |
| IsBig         | 1.000    | 1.000     | 1.000  | 1.000    |

### Predicados Espaciais

| Predicado | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| LeftOf    | 0.928    | 0.890     | 0.970  | 0.928    |
| RightOf   | 0.920    | 0.891     | 0.950  | 0.919    |
| Below     | 0.922    | 0.881     | 0.967  | 0.922    |
| Above     | 0.925    | 0.878     | 0.980  | 0.926    |
| CanStack  | 1.000    | 1.000     | 1.000  | 1.000    |

### Queries Compostas

| Query | Satisfatibilidade |
|------|-------------------|
| Q1: Pequeno abaixo cilindro E esq. quadrado | 0.089 |
| Q2: Cone verde entre dois objetos | 0.187 |
| Q3: Triângulos próximos mesmo tamanho | 0.934 |
| Opcional 1: Existe obj. esq. todos quadrados | 0.792 |
| Opcional 2: Quadrados direita círculos | 0.869 |
| LastOnTheLeft | 0.457 |
| LastOnTheRight | 0.448 |

### Explicações Geradas

```

✓ Q1: Objeto 1 (pequeno) está abaixo do objeto 5 (cilindro)
E à esquerda do objeto 3 (quadrado).

✓ Q2: Objeto 14 é um cone verde em x=0.26, com objetos [0, 1]
à esquerda e [2, 3] à direita.

✓ Q3: Todos os triângulos próximos têm o mesmo tamanho.

```
---

## 📊 RESUMO CONSOLIDADO (5 CENÁRIOS)

### Predicados Unários – Médias Finais

| Predicado      | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| IsCircle       | 1.000    | 1.000     | 1.000  | 1.000    |
| IsSquare       | 1.000    | 1.000     | 1.000  | 1.000    |
| **IsCylinder** | **0.952** | **1.000** | **0.703** | **0.831** |
| IsCone         | 1.000    | 1.000     | 1.000  | 1.000    |
| **IsTriangle** | **0.976** | **0.800** | **0.671** | **0.704** |
| **IsRed**      | **1.000** | **1.000** | **1.000** | **1.000** |
| **IsGreen**    | **1.000** | **1.000** | **1.000** | **1.000** |
| **IsBlue**     | **1.000** | **1.000** | **1.000** | **1.000** |
| IsSmall        | 1.000    | 1.000     | 1.000  | 1.000    |
| IsBig          | 1.000    | 1.000     | 1.000  | 1.000    |

---

### Predicados Espaciais – Médias Finais

| Predicado | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| LeftOf    | 0.920    | 0.886     | 0.957  | 0.920    |
| RightOf   | 0.915    | 0.886     | 0.945  | 0.914    |
| Below     | 0.922    | 0.879     | 0.971  | 0.922    |
| Above     | 0.923    | 0.879     | 0.973  | 0.924    |
| **CanStack** | **1.000** | **1.000** | **1.000** | **1.000** |

---

### Queries Compostas – Médias Finais

| Query | Média |
|-------|-------|
| **Q1**: Pequeno abaixo cilindro E esq. quadrado | **0.103** |
| **Q2**: Cone verde entre dois objetos | **0.165** |
| **Q3**: Triângulos próximos mesmo tamanho | **0.978** |
| **Opcional 1**: Existe obj. esq. todos quadrados | **0.733** |
| **Opcional 2**: Quadrados direita círculos | **0.856** |
| **LastOnTheLeft** | **0.457** |
| **LastOnTheRight** | **0.444** |
