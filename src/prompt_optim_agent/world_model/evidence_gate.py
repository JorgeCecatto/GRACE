"""
Aceitação por evidência "Lose to Win" (acceptance_mode='evidence' | 'evidence_shadow').

A aceitação de um candidato c contra o incumbente i é um teste PAREADO no
conjunto de validação (eval). Só os itens discordantes importam:
    ganho  g_j = 1 se c acerta e i erra
    perda  l_j = 1 se c erra  e i acerta
    Delta  = soma(g) - soma(l)
Com avaliação completa e margem m = 1, "Delta >= 1" equivale exatamente à regra
atual (acc_c > acc_i), pois acc_c - acc_i = Delta / N.

Este módulo NÃO chama o modelo: recebe uma função `fetch_lot(indices)` que devolve
os acertos (0/1) do candidato nos itens pedidos. Assim a mesma rotina serve ao
modo real (fetch chama o modelo) e ao modo sombra (fetch lê um vetor já avaliado).

Ver docs/lose_to_win.md.
"""

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


class EvidenceGate:
    """
    Histórico de viradas por item + avaliação sequencial com parada precoce.

    Histórico (aprendido dos candidatos OPT anteriores, sem chamadas extras):
    - itens que o incumbente da época ERRAVA: ganhos G_j em W_j tentativas;
    - itens que o incumbente da época ACERTAVA: perdas L_j em R_j tentativas.
    Só contam os itens efetivamente avaliados naquele candidato.

    Posterior por item (prior empírico agrupado, força kappa):
        ganho: Beta(kappa*rg + G_j, kappa*(1-rg) + W_j - G_j),  rg = taxa agrupada de ganho
        perda: Beta(kappa*rl + L_j, kappa*(1-rl) + R_j - L_j),  rl = taxa agrupada de perda
    A taxa agrupada é soma(G)/soma(W) (resp. soma(L)/soma(R)), suavizada com
    (hits + 0.5) / (tries + 1) e limitada a [1e-3, 1 - 1e-3] para manter os
    parâmetros da Beta estritamente positivos.
    """

    def __init__(
        self,
        n_items: int,
        margin: int = 1,
        risk: float = 0.05,
        kappa: float = 5.0,
        warmup: int = 10,
        mc_samples: int = 2000,
        seed: int = 0,
    ):
        assert margin >= 1, "evidence_margin deve ser >= 1 (com 0 o empate seria aceito)"
        assert 0.0 < risk < 0.5, "evidence_delta deve estar em (0, 0.5)"
        assert kappa > 0.0, "evidence_kappa deve ser > 0"
        self.n = int(n_items)
        self.margin = int(margin)
        self.risk = float(risk)
        self.kappa = float(kappa)
        self.warmup = int(warmup)
        self.mc_samples = int(mc_samples)
        self.rng = np.random.default_rng(seed)

        self.G = np.zeros(self.n)
        self.W = np.zeros(self.n)
        self.L = np.zeros(self.n)
        self.R = np.zeros(self.n)
        self.n_seen = 0  # candidatos OPT já incorporados ao histórico

    # ------------------------------------------------------------------ estado
    def in_warmup(self) -> bool:
        """Os primeiros `warmup` candidatos são avaliados por inteiro."""
        return self.n_seen < self.warmup

    @staticmethod
    def _pooled(hits: np.ndarray, tries: np.ndarray) -> float:
        rate = (hits.sum() + 0.5) / (tries.sum() + 1.0)
        return float(np.clip(rate, 1e-3, 1.0 - 1e-3))

    def posterior(self, inc_correct: Sequence[int]):
        """
        Parâmetros (a, b) da Beta de cada item, condicionados ao estado do
        incumbente atual: ganho nos itens que ele erra, perda nos que ele acerta.
        """
        inc = np.asarray(inc_correct, dtype=int)
        rg = self._pooled(self.G, self.W)
        rl = self._pooled(self.L, self.R)
        wrong = inc == 0
        a = np.where(wrong, self.kappa * rg + self.G, self.kappa * rl + self.L)
        b = np.where(
            wrong,
            self.kappa * (1.0 - rg) + self.W - self.G,
            self.kappa * (1.0 - rl) + self.R - self.L,
        )
        return a, b, rg, rl

    def order(self, inc_correct: Sequence[int], a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Primeiro os itens que o incumbente erra (maior prob. posterior de ganho
        primeiro), depois os que acerta (maior prob. posterior de perda primeiro).
        Empates desfeitos pelo índice (ordem estável).
        """
        inc = np.asarray(inc_correct, dtype=int)
        mean = a / (a + b)
        idx = np.arange(self.n)
        wrong = idx[inc == 0]
        right = idx[inc == 1]
        wrong = wrong[np.argsort(-mean[wrong], kind='stable')]
        right = right[np.argsort(-mean[right], kind='stable')]
        return np.concatenate([wrong, right])

    # --------------------------------------------------------------- Monte Carlo
    def prob_reach(self, delta_obs: int, remaining: np.ndarray, inc: np.ndarray,
                   a: np.ndarray, b: np.ndarray) -> float:
        """
        P(Delta_total >= margin | Delta observado, itens restantes).
        Para cada item restante sorteia p_j ~ Beta(a_j, b_j) e o resultado
        ~ Bernoulli(p_j); soma +1 (ganho) nos itens que o incumbente erra e -1
        (perda) nos que acerta.
        """
        if len(remaining) == 0:
            return float(delta_obs >= self.margin)
        sign = np.where(inc[remaining] == 0, 1, -1)
        p = self.rng.beta(a[remaining], b[remaining], size=(self.mc_samples, len(remaining)))
        hit = self.rng.random((self.mc_samples, len(remaining))) < p
        total = delta_obs + (hit * sign).sum(axis=1)
        return float((total >= self.margin).mean())

    # -------------------------------------------------------- avaliação sequencial
    def run(
        self,
        inc_correct: Sequence[int],
        fetch_lot: Callable[[List[int]], Sequence[int]],
        lot_size: int,
        record_all: bool = False,
    ) -> Dict:
        """
        Avalia o candidato em lotes, na ordem do posterior, e decide.

        fetch_lot(indices) -> acertos (0/1) do candidato nesses itens.
        Após cada lote (exceto o último): P = P(Delta_total >= m).
          - P <= risk       -> rejeita cedo ('reject');
          - P >= 1 - risk   -> NÃO aceita parcial: completa a avaliação ('complete')
                               e decide com o Delta exato;
          - senão           -> 'continue'.
        Se todos os itens forem avaliados, aceita sse Delta >= margin.

        record_all=True (modo sombra): continua calculando/registrando P depois de
        'complete', sem alterar a decisão.

        Devolve dict com: accepted, rejected_early, cand_correct (-1 = não avaliado),
        n_evaluated, delta (parcial se rejeitado cedo), steps, est_delta_total
        (Delta esperado se o candidato fosse avaliado até o fim; só é uma ESTIMATIVA
        quando rejected_early) e as taxas agrupadas usadas.
        """
        inc = np.asarray(inc_correct, dtype=int)
        assert len(inc) == self.n, f"incumbente com {len(inc)} itens, esperado {self.n}"
        lot_size = max(1, int(lot_size))
        a, b, rg, rl = self.posterior(inc)
        order = self.order(inc, a, b)

        cand = np.full(self.n, -1, dtype=int)
        delta = 0
        pos = 0
        completing = False
        early = False
        steps = []

        while pos < self.n:
            lot = order[pos:pos + lot_size]
            res = np.asarray(fetch_lot([int(j) for j in lot]), dtype=int)
            assert len(res) == len(lot), "fetch_lot devolveu número de acertos diferente do pedido"
            cand[lot] = res
            delta += int(((res == 1) & (inc[lot] == 0)).sum() - ((res == 0) & (inc[lot] == 1)).sum())
            pos += len(lot)
            remaining = order[pos:]
            if len(remaining) == 0:
                break
            if completing and not record_all:
                continue

            P = self.prob_reach(delta, remaining, inc, a, b)
            if completing:
                action = 'complete'  # só registro (modo sombra)
            elif P <= self.risk:
                action = 'reject'
                early = True
            elif P >= 1.0 - self.risk:
                action = 'complete'
                completing = True
            else:
                action = 'continue'
            steps.append({'n_items': int(pos), 'delta': int(delta), 'P': round(P, 4), 'action': action})
            if early:
                break

        n_eval = int((cand >= 0).sum())
        remaining = order[pos:]
        mean = a / (a + b)
        expected_rest = float(
            np.where(inc[remaining] == 0, mean[remaining], -mean[remaining]).sum()
        ) if len(remaining) else 0.0

        return {
            'accepted': bool((not early) and delta >= self.margin),
            'rejected_early': bool(early),
            'cand_correct': cand,
            'n_evaluated': n_eval,
            'delta': int(delta),
            'steps': steps,
            'est_delta_total': float(delta + expected_rest),
            'rate_gain': rg,
            'rate_loss': rl,
        }

    # ------------------------------------------------------ regra exata (sem MC)
    def run_exact(
        self,
        inc_correct: Sequence[int],
        fetch_lot: Callable[[List[int]], Sequence[int]],
        lot_size: int,
    ) -> Dict:
        """
        Avaliação sequencial EXATA: rejeita cedo só quando é matematicamente
        impossível o candidato atingir a margem. Sem Monte Carlo e sem aquecimento.

        Notação: J0 = itens que o incumbente erra, J1 = itens que acerta.
        Delta_final = ganhos (itens de J0 que o candidato acerta)
                      - perdas (itens de J1 que o candidato erra).
        Após avaliar parte dos itens, com delta_obs o Delta parcial, R0 = itens de J0
        ainda não avaliados e R1 = itens de J1 ainda não avaliados:

            delta_obs - R1  <=  Delta_final  <=  delta_obs + R0.

        Teorema (exatidão): se delta_obs + R0 < m, então Delta_final < m para
        QUALQUER resultado dos itens restantes, pois cada item de J0 restante soma no
        máximo +1 e cada item de J1 restante soma no máximo 0. Logo a rejeição precoce
        nunca rejeita um candidato que a avaliação completa aceitaria (Delta_final >= m),
        qualquer que seja a ordem de avaliação. A ordem (a aprendida por `order`, J0
        primeiro) só afeta QUANTOS itens são poupados, não a decisão.

        Procedimento: avalia em lotes de `lot_size` na ordem; após cada lote (exceto o
        último), se delta_obs + R0 < m, rejeita. NÃO existe aceitação parcial: se não
        rejeitar, avalia até o fim e aceita sse Delta >= m. Com m = 1 reproduz a regra
        estrita do GRACE (acc_c > acc_i).

        Devolve as mesmas chaves de `run()`. Diferenças de significado:
        - `est_delta_total` é o LIMITE SUPERIOR delta_obs + R0 no momento da parada
          (não uma estimativa); se não rejeitou, é o Delta exato.
        - `rate_gain`/`rate_loss` vêm do posterior usado só para ordenar.
        - cada step: {'n_items', 'delta', 'upper_bound', 'lower_bound', 'action'},
          com action em {'continue', 'reject'}.
        """
        inc = np.asarray(inc_correct, dtype=int)
        assert len(inc) == self.n, f"incumbente com {len(inc)} itens, esperado {self.n}"
        lot_size = max(1, int(lot_size))
        a, b, rg, rl = self.posterior(inc)
        order = self.order(inc, a, b)

        cand = np.full(self.n, -1, dtype=int)
        delta = 0
        pos = 0
        early = False
        steps = []
        upper = 0

        while pos < self.n:
            lot = order[pos:pos + lot_size]
            res = np.asarray(fetch_lot([int(j) for j in lot]), dtype=int)
            assert len(res) == len(lot), "fetch_lot devolveu número de acertos diferente do pedido"
            cand[lot] = res
            delta += int(((res == 1) & (inc[lot] == 0)).sum() - ((res == 0) & (inc[lot] == 1)).sum())
            pos += len(lot)
            rest = order[pos:]
            r0 = int((inc[rest] == 0).sum())
            r1 = int((inc[rest] == 1).sum())
            upper = delta + r0
            if len(rest) == 0:
                break
            if upper < self.margin:
                early = True
            steps.append({
                'n_items': int(pos), 'delta': int(delta),
                'upper_bound': int(upper), 'lower_bound': int(delta - r1),
                'action': 'reject' if early else 'continue',
            })
            if early:
                break

        return {
            'accepted': bool((not early) and delta >= self.margin),
            'rejected_early': bool(early),
            'cand_correct': cand,
            'n_evaluated': int((cand >= 0).sum()),
            'delta': int(delta),
            'steps': steps,
            'est_delta_total': float(upper),  # LIMITE SUPERIOR de Delta_final (exato se completo)
            'rate_gain': rg,
            'rate_loss': rl,
        }

    # ------------------------------------------------------------------ histórico
    def update(self, inc_correct: Sequence[int], cand_correct: Sequence[int]):
        """
        Incorpora um candidato OPT ao histórico. cand_correct usa -1 para itens
        não avaliados (rejeição precoce); esses itens não contam.
        """
        inc = np.asarray(inc_correct, dtype=int)
        cand = np.asarray(cand_correct, dtype=int)
        ev = cand >= 0
        wrong = ev & (inc == 0)
        right = ev & (inc == 1)
        self.W[wrong] += 1
        self.G[wrong] += (cand[wrong] == 1)
        self.R[right] += 1
        self.L[right] += (cand[right] == 0)
        self.n_seen += 1

    def summary(self) -> Dict:
        return {
            'n_candidates_seen': int(self.n_seen),
            'rate_gain': self._pooled(self.G, self.W),
            'rate_loss': self._pooled(self.L, self.R),
            'W_total': int(self.W.sum()), 'G_total': int(self.G.sum()),
            'R_total': int(self.R.sum()), 'L_total': int(self.L.sum()),
        }
