"""
Bayesian Difficulty Tracker (curriculum_mode='bayes')

Cada exemplo de treino i tem uma probabilidade de acerto p_i modelada por
Beta-Bernoulli:
- prior: m_i = sigmoid(w0 + w1 * z_i), em que z_i é a atipicidade semântica
  (centroide ou HDBSCAN) e (w0, w1) vêm de uma regressão logística dos
  resultados zero-shot de P0 sobre z (depois reajustada por empirical Bayes);
- evidência: resultados dos prompts da linhagem numa janela deslizante de W;
- posterior: p_hat_i = (kappa * m_i + s_i) / (kappa + n_i).
"""

import json
import logging
from collections import Counter, deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TIERS = ('EASY', 'MIXED', 'HARD')
MIN_FIT_OBS = 10
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _f(x):
    """Float seguro para JSON (None para None/NaN/inf)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return _f(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def trace(logger, tag: str, payload: dict):
    """Emite uma linha 'CURRICULUM_TRACE <tag> <json>'."""
    line = f"CURRICULUM_TRACE {tag} {json.dumps(payload, default=_json_default, ensure_ascii=False)}"
    if logger is not None:
        logger.info(line)
    else:
        print(line)


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic_irls(X, y, l2: float = 1.0, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """
    Regressão logística por IRLS (Newton) com penalização L2.

    X: (N, D) já com a coluna de bias em X[:, 0]; y em {0, 1}.
    Maximiza  sum_i [y_i log mu_i + (1 - y_i) log(1 - mu_i)] - (l2 / 2) * ||w[1:]||^2
    (o intercepto w[0] NÃO é penalizado).
    Passo de Newton: w <- w + (X^T S X + L)^{-1} (X^T (y - mu) - L w),
    com S = diag(mu (1 - mu)) e L = diag(0, l2, ..., l2).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    L = l2 * np.eye(d)
    L[0, 0] = 0.0
    w = np.zeros(d)
    for _ in range(max_iter):
        mu = sigmoid(X @ w)
        s = np.maximum(mu * (1.0 - mu), 1e-10)
        grad = X.T @ (y - mu) - L @ w
        H = (X * s[:, None]).T @ X + L
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        w = w + step
        if np.max(np.abs(step)) < tol:
            break
    return w


def _rankdata(a) -> np.ndarray:
    """Ranks com média nos empates (equivalente a scipy.stats.rankdata 'average')."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    order = np.argsort(a, kind='mergesort')
    _, inv, counts = np.unique(a[order], return_inverse=True, return_counts=True)
    avg = np.bincount(inv, weights=np.arange(n, dtype=float)) / counts
    ranks = np.empty(n, dtype=float)
    ranks[order] = avg[inv]
    return ranks


def spearman(a, b) -> Optional[float]:
    """Correlação de Spearman = Pearson dos ranks. None se indefinida."""
    if len(a) < 3:
        return None
    ra, rb = _rankdata(a), _rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if den < _EPS:
        return None
    return float((ra * rb).sum() / den)


def _zscore(x, ref) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ref = np.asarray(ref, dtype=float)
    sd = ref.std()
    if sd < _EPS:
        return np.zeros_like(x)
    return (x - ref.mean()) / sd


def _normalize(E) -> np.ndarray:
    E = np.asarray(E, dtype=float)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.maximum(norms, _EPS)


def _groups(labels, centroid_mode: str) -> List[Tuple[str, np.ndarray]]:
    labels = np.asarray([str(l) for l in labels])
    if centroid_mode == 'global':
        return [('__all__', np.arange(len(labels)))]
    if centroid_mode == 'class':
        return [(str(c), np.where(labels == c)[0]) for c in sorted(np.unique(labels))]
    raise ValueError(f"centroid_mode inválido: {centroid_mode!r} (use 'class' ou 'global')")


def _centroid_dist(En, member_idx, ref_idx) -> Tuple[np.ndarray, np.ndarray]:
    """Distância de cosseno ao centroide (normalizado) do conjunto de referência."""
    c = En[ref_idx].mean(axis=0)
    c = c / max(np.linalg.norm(c), _EPS)
    return 1.0 - En[member_idx] @ c, 1.0 - En[ref_idx] @ c


def _centroid_group(En, member_idx, ref_idx, typical_percentile):
    d_mem, d_ref = _centroid_dist(En, member_idx, ref_idx)
    z = _zscore(d_mem, d_ref)
    typical = d_mem < np.percentile(d_ref, typical_percentile)
    return z, typical


# ---------------------------------------------------------------------------
# Atipicidade intrínseca
# ---------------------------------------------------------------------------

def atypicality_centroid(
    embeddings,
    labels,
    centroid_mode: str = 'class',
    typical_percentile: float = 50.0,
    min_group_size: int = 10,
):
    """
    Atipicidade = distância de cosseno ao centroide do grupo, em z-score no grupo.
    typical = distância abaixo do percentil `typical_percentile` do grupo.
    Classes com menos de `min_group_size` exemplos usam o conjunto inteiro como
    referência (centroide, média/desvio do z-score e percentil).

    Returns: z (N,), typical (N,) bool, diag {grupo: {...}}
    """
    En = _normalize(embeddings)
    n = len(En)
    all_idx = np.arange(n)
    z = np.zeros(n)
    typical = np.zeros(n, dtype=bool)
    diag = {}
    for g, idx in _groups(labels, centroid_mode):
        use_all = len(idx) < min_group_size
        zg, tg = _centroid_group(En, idx, all_idx if use_all else idx, typical_percentile)
        z[idx] = zg
        typical[idx] = tg
        diag[g] = {
            'size': int(len(idx)),
            'path': 'centroid',
            'reference': 'all' if use_all else 'group',
            'typical_frac': _f(tg.mean()) if len(idx) else None,
        }
    return z, typical, diag


def atypicality_hdbscan(
    embeddings,
    labels,
    centroid_mode: str = 'class',
    typical_percentile: float = 50.0,
    min_group_size: int = 10,
    min_cluster_size: int = 5,
    min_samples: int = 3,
):
    """
    Atipicidade = distância de cosseno ao medoide HDBSCAN mais próximo (inclusive
    para ruído), em z-score no grupo.
    typical = ponto não-ruído com probabilities_ >= percentil `typical_percentile`
    das probabilidades dos pontos não-ruído do grupo.

    Padrões (min_cluster_size=5, min_samples=3): os grupos aqui têm dezenas de
    exemplos (ex.: snarks, ~40 por classe). min_cluster_size=5 permite de 2 a ~8
    clusters com sentido nesse tamanho, sem aceitar pares/trincas como cluster.
    min_samples, por padrão, igual a min_cluster_size, deixa a densidade núcleo
    conservadora demais para N pequeno e marca boa parte do grupo como ruído;
    3 reduz o ruído sem tornar a estimativa de densidade instável.

    Fallback para a versão centroide naquele grupo (motivo registrado em diag) se:
    - grupo menor que min_group_size (ou pequeno demais para os parâmetros do
      HDBSCAN): usa o conjunto inteiro como referência quando < min_group_size;
    - todos os pontos são ruído;
    - existe um único cluster sem nenhum ponto de ruído (sem estrutura útil).
    """
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as e:  # sklearn < 1.3 não tem HDBSCAN
        raise ImportError(
            "intrinsic_mode='hdbscan' requer scikit-learn >= 1.3 (sklearn.cluster.HDBSCAN)."
        ) from e

    En = _normalize(embeddings)
    n = len(En)
    all_idx = np.arange(n)
    z = np.zeros(n)
    typical = np.zeros(n, dtype=bool)
    diag = {}
    min_size = max(min_group_size, min_cluster_size + 1, min_samples + 1)

    for g, idx in _groups(labels, centroid_mode):
        d = {'size': int(len(idx))}
        diag[g] = d

        def _fallback(reason, ref_idx, n_clusters=None, noise_frac=None):
            zg, tg = _centroid_group(En, idx, ref_idx, typical_percentile)
            z[idx] = zg
            typical[idx] = tg
            d.update({
                'path': 'fallback',
                'reason': reason,
                'reference': 'all' if ref_idx is all_idx else 'group',
                'n_clusters': n_clusters,
                'noise_frac': noise_frac,
                'typical_frac': _f(tg.mean()) if len(idx) else None,
                'spearman_vs_centroid': None,
            })

        if len(idx) < min_size:
            _fallback(f'group_size<{min_size}', all_idx if len(idx) < min_group_size else idx)
            continue

        Eg = En[idx]
        D = 1.0 - Eg @ Eg.T
        D = np.clip((D + D.T) / 2.0, 0.0, None)
        np.fill_diagonal(D, 0.0)

        hdb = HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples)
        lab = hdb.fit_predict(D)
        prob = np.asarray(hdb.probabilities_, dtype=float)
        noise = lab == -1
        cluster_ids = sorted(set(lab.tolist()) - {-1})
        noise_frac = _f(noise.mean())

        if not cluster_ids:
            _fallback('all_noise', idx, 0, noise_frac)
            continue
        if len(cluster_ids) == 1 and not noise.any():
            _fallback('single_cluster_no_noise', idx, 1, noise_frac)
            continue

        # Medoide: ponto com menor soma de distâncias aos demais do mesmo cluster.
        medoids = []
        for c in cluster_ids:
            members = np.where(lab == c)[0]
            medoids.append(members[np.argmin(D[np.ix_(members, members)].sum(axis=1))])
        raw = D[:, medoids].min(axis=1)

        tg = (~noise) & (prob >= np.percentile(prob[~noise], typical_percentile))
        z[idx] = _zscore(raw, raw)
        typical[idx] = tg

        d_cent, _ = _centroid_dist(En, idx, idx)
        d.update({
            'path': 'hdbscan',
            'reason': None,
            'reference': 'group',
            'n_clusters': len(cluster_ids),
            'noise_frac': noise_frac,
            'typical_frac': _f(tg.mean()),
            'spearman_vs_centroid': spearman(raw, d_cent),
        })

    return z, typical, diag


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class DifficultyTracker:
    """
    Estado por exemplo (chave = texto da questão): z, typical, m (prior_mean),
    hist (deque(maxlen=W) de (correct, is_p0)), n_total e tier.
    """

    atypicality_centroid = staticmethod(atypicality_centroid)
    atypicality_hdbscan = staticmethod(atypicality_hdbscan)

    def __init__(
        self,
        window: int = 10,
        prior_strength: float = 2.0,
        easy_threshold: float = 0.85,
        hard_threshold: float = 0.15,
        sampling_floor: float = 0.05,
        refit_every: int = 5,
        l2: float = 1.0,
        logger: logging.Logger = None,
    ):
        self.window = int(window)
        self.kappa = float(prior_strength)
        self.easy_threshold = float(easy_threshold)
        self.hard_threshold = float(hard_threshold)
        self.sampling_floor = float(sampling_floor)
        self.refit_every = max(1, int(refit_every))
        self.l2 = float(l2)
        self.logger = logger

        self.w = np.zeros(2)  # (w0, w1)
        self.items: Dict[str, dict] = {}
        self.n_update_calls = 0
        self.n_fits = 0
        self.n_unknown_keys = 0
        self.n_duplicate_keys = 0
        self.migrations = Counter()        # por observação: "DE->PARA"
        self.refit_migrations = Counter()  # causadas por reajuste do prior

    # ----------------------------------------------------------------- modelo
    def _tier_of(self, p: float) -> str:
        if p >= self.easy_threshold:
            return 'EASY'
        if p <= self.hard_threshold:
            return 'HARD'
        return 'MIXED'

    def _prior_mean(self, z):
        return sigmoid(self.w[0] + self.w[1] * np.asarray(z, dtype=float))

    def _posterior(self, it) -> float:
        n = len(it['hist'])
        s = sum(c for c, _ in it['hist'])
        return (self.kappa * it['m'] + s) / (self.kappa + n)

    def _fit_prior(self, z, y, source: str) -> bool:
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)
        X = np.column_stack([np.ones(len(z)), z])
        self.w = fit_logistic_irls(X, y, l2=self.l2)
        self.n_fits += 1
        trace(self.logger, 'prior_fit', {
            'source': source, 'n_obs': int(len(y)), 'pos_rate': _f(y.mean()),
            'w0': _f(self.w[0]), 'w1': _f(self.w[1]), 'n_update_calls': self.n_update_calls,
        })
        return True

    def _refresh_prior(self):
        for it in self.items.values():
            it['m'] = float(self._prior_mean(it['z']))
            new = self._tier_of(self._posterior(it))
            if new != it['tier']:
                self.refit_migrations[f"{it['tier']}->{new}"] += 1
                it['tier'] = new

    # --------------------------------------------------------------- públicos
    def register(self, keys: Sequence[str], z, typical, p0_correct):
        """Ajusta (w0, w1) em P0 ~ z, define m_i e usa P0 como 1ª observação."""
        z = np.asarray(z, dtype=float)
        typical = np.asarray(typical, dtype=bool)
        y = np.asarray(p0_correct, dtype=int)

        n_pos = int(y.sum())
        if 0 < n_pos < len(y):
            self._fit_prior(z, y, source='p0')
        else:
            # Uma só classe em P0: o intercepto divergiria (sem penalização).
            # Usa prior constante com taxa suavizada e w1 = 0.
            rate = (n_pos + 0.5) / (len(y) + 1.0)
            self.w = np.array([np.log(rate / (1.0 - rate)), 0.0])
            trace(self.logger, 'prior_fit', {
                'source': 'p0_degenerate', 'n_obs': int(len(y)), 'pos_rate': _f(y.mean()) if len(y) else None,
                'w0': _f(self.w[0]), 'w1': 0.0, 'n_update_calls': 0,
            })

        self.items = {}
        for k, zi, ti, yi in zip(keys, z, typical, y):
            if k in self.items:
                self.n_duplicate_keys += 1
            it = {
                'id': len(self.items),
                'z': float(zi),
                'typical': bool(ti),
                'm': float(self._prior_mean(zi)),
                'hist': deque(maxlen=self.window),
                'n_total': 0,
                'tier': None,
            }
            # Resultado de P0 como primeira observação (marcado para não entrar no refit).
            it['hist'].append((int(yi), True))
            it['n_total'] = 1
            it['tier'] = self._tier_of(self._posterior(it))
            self.items[k] = it

    def update(self, keys: Sequence[str], correct: Sequence[int]):
        for k, c in zip(keys, correct):
            it = self.items.get(k)
            if it is None:
                self.n_unknown_keys += 1
                continue
            it['hist'].append((int(c), False))
            it['n_total'] += 1
            new = self._tier_of(self._posterior(it))
            if new != it['tier']:
                self.migrations[f"{it['tier']}->{new}"] += 1
                it['tier'] = new

        self.n_update_calls += 1
        if self.n_update_calls % self.refit_every == 0:
            self._refit()

    def _refit(self):
        """Empirical Bayes: reajusta (w0, w1) com todas as observações das janelas (exceto P0)."""
        zs, ys = [], []
        for it in self.items.values():
            for c, is_p0 in it['hist']:
                if not is_p0:
                    zs.append(it['z'])
                    ys.append(c)
        n, n_pos = len(ys), int(sum(ys))
        if n < MIN_FIT_OBS or n_pos in (0, n):
            trace(self.logger, 'prior_refit_skipped', {
                'n_obs': n, 'n_pos': n_pos, 'n_update_calls': self.n_update_calls,
                'reason': f'n_obs<{MIN_FIT_OBS}' if n < MIN_FIT_OBS else 'single_outcome_class',
            })
            return
        self._fit_prior(zs, ys, source='refit')
        self._refresh_prior()

    def p_success(self, key: str) -> Optional[float]:
        it = self.items.get(key)
        return None if it is None else float(self._posterior(it))

    def snapshot_p_hat(self) -> Dict[str, float]:
        """p_hat atual de todos os exemplos (referência congelada por incumbente)."""
        return {k: float(self._posterior(it)) for k, it in self.items.items()}

    def tier(self, key: str) -> Optional[str]:
        it = self.items.get(key)
        return None if it is None else it['tier']

    def regression_test(
        self,
        observations: Sequence[Tuple[str, int, Optional[float]]],
        easy_threshold: Optional[float] = None,
        min_T: int = 3,
    ) -> dict:
        """
        observations: lista de (key, correct, p_hat), com p_hat capturado ANTES
        de a observação entrar na janela.

        Por que p_hat precisa ser anterior ao update: se for posterior, a própria
        observação contamina a expectativa. Um erro em exemplo dominado reduz o
        p_hat desse exemplo, o que aumenta E justamente quando F aumenta. Pode
        ainda derrubar o p_hat abaixo de easy_threshold, e o exemplo sai de T.
        Nos dois casos r encolhe sem nenhum erro ou aviso: o teste perde poder
        silenciosamente.

        easy_threshold: limiar de p_hat para entrar em T. É SEPARADO do
        self.easy_threshold que define os tiers (padrão aqui: o dos tiers).
        O world model passa regression_easy_threshold (padrão 0.80): com
        kappa=2 e P0 como 1ª observação, um exemplo com m=0.7 que P0 acertou
        fica em p_hat=0.80 e ficaria fora de T com 0.85.

        T = observações de exemplos típicos com p_hat >= easy_threshold.
        F = erros em T; E = sum(1 - p_hat); V = sum(p_hat (1 - p_hat));
        r = (F - E) / sqrt(V)   (None se |T| < min_T ou V ~ 0).
        """
        thr = self.easy_threshold if easy_threshold is None else float(easy_threshold)
        T = [
            (k, int(c), float(p)) for k, c, p in observations
            if p is not None and k in self.items and self.items[k]['typical'] and p >= thr
        ]
        F = sum(1 - c for _, c, _ in T)
        E = sum(1.0 - p for _, _, p in T)
        V = sum(p * (1.0 - p) for _, _, p in T)
        regressed = list(dict.fromkeys(k for k, c, _ in T if c == 0))
        r = None if (len(T) < min_T or V < 1e-9) else (F - E) / np.sqrt(V)
        return {
            'r': _f(r), 'F': int(F), 'E': _f(E), 'V': _f(V), 'T': len(T),
            'regressed_keys': regressed,
        }

    def sample_update_batch(
        self,
        wrong_keys: Sequence[str],
        right_keys: Sequence[str],
        n_wrong: int,
        n_right: int,
        rng: np.random.Generator,
        regressed_keys: Optional[Sequence[str]] = None,
        p_hat_map: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Devolve (índices em wrong_keys, índices em right_keys), sem reposição.

        Regra padrão: erros com peso p_hat + floor; acertos com peso
        (1 - p_hat) + floor. O peso é a probabilidade do resultado OPOSTO ao
        observado, ou seja, seleciona onde o prompt atual contradiz a
        expectativa da linhagem (erros em exemplos que ela costuma acertar,
        acertos em exemplos que ela costuma errar). O floor mantém todo
        exemplo amostrável.

        Com regressed_keys não vazio: os erros são escolhidos primeiro entre os
        regredidos (peso p_hat + floor) e as vagas restantes seguem a regra
        padrão. Os acertos não mudam.

        p_hat_map (opcional): p_hat por chave a usar no lugar do estado atual
        (ex.: o p_hat anterior ao update desta iteração).
        """
        floor = self.sampling_floor

        def _p(k):
            if p_hat_map is not None and p_hat_map.get(k) is not None:
                return float(p_hat_map[k])
            p = self.p_success(k)
            return 0.5 if p is None else p

        def _weighted(cands, weights, k):
            cands = list(cands)
            k = min(int(k), len(cands))
            if k <= 0:
                return []
            w = np.maximum(np.asarray(weights, dtype=float), 1e-12)
            pick = rng.choice(len(cands), size=k, replace=False, p=w / w.sum())
            return [int(cands[j]) for j in pick]

        wrong_idx: List[int] = []
        if regressed_keys:
            reg = set(regressed_keys)
            cand = [i for i, k in enumerate(wrong_keys) if k in reg]
            wrong_idx = _weighted(cand, [_p(wrong_keys[i]) + floor for i in cand], n_wrong)
        taken = set(wrong_idx)
        rest = [i for i in range(len(wrong_keys)) if i not in taken]
        wrong_idx += _weighted(rest, [_p(wrong_keys[i]) + floor for i in rest], n_wrong - len(wrong_idx))

        right_idx = _weighted(
            range(len(right_keys)), [(1.0 - _p(k)) + floor for k in right_keys], n_right
        )
        return wrong_idx, right_idx

    def log_state(self, tag: str, with_items: bool = False):
        counts = Counter(it['tier'] for it in self.items.values())
        payload = {
            'tiers': {t: counts.get(t, 0) for t in TIERS},
            'w0': _f(self.w[0]),
            'w1': _f(self.w[1]),
            'n_update_calls': self.n_update_calls,
            'n_fits': self.n_fits,
            'migrations': dict(self.migrations),
            'refit_migrations': dict(self.refit_migrations),
            'n_unknown_keys': self.n_unknown_keys,
            'n_duplicate_keys': self.n_duplicate_keys,
        }
        if with_items:
            payload['items'] = [
                {
                    'id': it['id'],
                    'q': k[:80],
                    'z': _f(it['z']),
                    'typical': it['typical'],
                    'm': _f(it['m']),
                    'p_hat': _f(self._posterior(it)),
                    'window': len(it['hist']),
                    'n_total': it['n_total'],
                    'tier': it['tier'],
                }
                for k, it in self.items.items()
            ]
        trace(self.logger, tag, payload)
